#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Where does ``lambda_k`` settle, as a function of ``lambda_0``?

Reproduces the table of section 11.1 of ``KE_HOACH_IMPLEMENT_IPPO.md``.

Step 3 of Algorithm 1 gives, using ``||u_k - v_k|| = lambda_k ||F(u_k)||``::

    lambda_{k+1} = min(p ||u_k-v_k|| / ||F(u_k)-F(v_k)||, lambda_k)
                 ~ min(p / L_local, lambda_k)

so ``lambda_k`` only shrinks while ``lambda_k > p / L_local`` and settles at
``~ p / L_local``. If ``lambda_0`` is already below that, the adaptive mechanism
never fires and PCVI degenerates into fixed-step extragradient. This script
measures where the knee is on a real task.

WARNING: the defaults below are a fast probe (400 collected frames), not an
experiment. They are ~0.01% of the frames of a default BenchMARL run and use a
single seed. Use them to locate the order of magnitude of ``lambda_0``, not to
compare optimizers.

Usage::

    python examples/pcvi/sweep_lambda0.py
    python examples/pcvi/sweep_lambda0.py --iters 20 --seeds 0 1 2
"""

import argparse
import pathlib
import warnings

import torch

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import (
    AdaptiveConfig,
    AdaptiveExtragradientConfig,
    ExtragradientConfig,
    PcConfig,
    PcviConfig,
)

# The 2x2 grid: {fixed, adaptive} lambda x {Step 5 contraction, beta_k = 1},
# plus `adaptive`, which has a lambda_0 of its own and the same failure mode.
OPTIMIZERS = {
    "extragradient": ExtragradientConfig,
    "adaptive_extragradient": AdaptiveExtragradientConfig,
    "pc": PcConfig,
    "pcvi": PcviConfig,
    "adaptive": AdaptiveConfig,
}


def _output_dir(args) -> pathlib.Path:
    """Every run of every script lands under one folder, created on demand."""
    directory = pathlib.Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_experiment(optimizer_name, lambda_0, seed, args):
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = config.train_device = config.buffer_device = args.device
    config.max_n_iters = args.iters
    config.max_n_frames = None
    config.on_policy_collected_frames_per_batch = args.frames_per_batch
    config.on_policy_n_envs_per_worker = args.n_envs
    config.on_policy_minibatch_size = args.minibatch_size
    config.on_policy_n_minibatch_iters = args.epochs
    config.evaluation = False
    config.render = False
    config.save_folder = str(_output_dir(args))
    config.loggers = list(args.loggers)
    config.create_json = False
    config.checkpoint_interval = 0
    # PCVI refuses to run with gradient clipping: a clipped gradient is not the
    # gradient of anything, which corrupts the Lipschitz estimate driving lambda.
    config.clip_grad_val = None
    # avoids a torchrl 0.11 bug in SyncDataCollector.state_dict() after run()
    config.collect_with_grad = True

    optimizer_config = OPTIMIZERS[optimizer_name].get_from_yaml()
    optimizer_config.lambda_0 = lambda_0

    return Experiment(
        task=task_config_registry[args.task].get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=seed,
        config=config,
    )


def run_one(optimizer_name, lambda_0, seed, args):
    experiment = build_experiment(optimizer_name, lambda_0, seed, args)
    group = args.group or next(iter(experiment.group_map))
    optimizer = experiment.optimizers[group]["loss_objective"]

    betas, corrs = [], []
    original_apply = optimizer.apply

    def spy_apply(*a, **kw):
        info = original_apply(*a, **kw)
        # `adaptive` keeps Adam and has no beta_k at all, and every optimizer
        # prefixes its diagnostics with its own name, so neither key can be
        # assumed to be there.
        if "pcvi_beta_k" in info:
            betas.append(info["pcvi_beta_k"])
        for key in ("pcvi_grad_corr", "adaptive_grad_corr"):
            if key in info:
                corrs.append(info[key])
                break
        return info

    optimizer.apply = spy_apply
    diverged = False
    try:
        experiment.run()
    except (AssertionError, RuntimeError, ValueError) as error:
        # A diverging run poisons the parameters with nan, and the environment
        # then refuses the nan actions. That is a result, not a crash: outside
        # lambda <= p/L there is nothing to pull a fixed lambda back.
        diverged = True
        print(f"    [diverged: {type(error).__name__}]")

    has_beta = bool(betas)
    betas = torch.tensor(betas) if has_beta else torch.zeros(1)
    return {
        "diverged": float(diverged),
        "lambda_end": optimizer.lambda_k,
        "n_steps": optimizer.n_steps,
        # nan, not 0, for a branch that has no beta_k: 0 would read as
        # "the contraction collapsed", which is a different finding.
        "beta_median": betas.median().item() if has_beta else float("nan"),
        "beta_negative_pct": (
            100.0 * (betas < 0).float().mean().item() if has_beta else float("nan")
        ),
        "grad_corr": torch.tensor(corrs).mean().item() if corrs else float("nan"),
        "mean_return": experiment.mean_return,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="vmas/balance",
                        choices=sorted(task_config_registry),
                        help="RUN THIS ON THE TASK YOU WILL ACTUALLY USE: lambda "
                             "settles at ~p/L_local, which is a property of the "
                             "task and the trajectory, not of the algorithm. A "
                             "lambda_0 picked on balance does not transfer to "
                             "simple_tag.")
    parser.add_argument("--lambda0", type=float, nargs="+",
                        default=[1.0, 0.1, 0.01, 1e-3, 1e-4])
    parser.add_argument("--optimizer", type=str, nargs="+", default=["pcvi"],
                        choices=list(OPTIMIZERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--iters", type=int, default=4, help="collection rounds")
    parser.add_argument("--epochs", type=int, default=10,
                        help="on_policy_n_minibatch_iters")
    parser.add_argument("--frames-per-batch", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", default="outputs",
                        help="all runs land in this folder, one subfolder each")
    parser.add_argument("--loggers", nargs="*", default=["csv"],
                        help="[] to log nothing, csv and/or wandb otherwise")
    parser.add_argument("--group", type=str, default=None,
                        help="agent group to watch; default is the task's first "
                             "(balance has only `agents`, simple_tag has "
                             "`adversary` and `agent`)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    minibatches = -(-args.frames_per_batch // args.minibatch_size)
    steps = args.iters * args.epochs * minibatches
    print(
        f"IPPO + {args.task} + MLP | {args.iters} rounds x {args.epochs} epochs x "
        f"{minibatches} minibatches = {steps} update steps per branch "
        f"({args.iters * args.frames_per_batch} collected frames), "
        f"seeds {args.seeds}"
    )
    print(f"{'optimizer':>22} {'lambda_0':>10} {'lambda_end':>12} {'end/0':>8} "
          f"{'beta med':>10} {'beta<0 %':>9} {'return':>9} {'status':>9}")
    print("-" * 96)

    table = {}
    for optimizer_name in args.optimizer:
        for lambda_0 in args.lambda0:
            results = [
                run_one(optimizer_name, lambda_0, seed, args) for seed in args.seeds
            ]
            mean = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
            table[(optimizer_name, lambda_0)] = mean
            status = (
                "ok" if mean["diverged"] == 0.0
                else ("DIVERGED" if mean["diverged"] == 1.0 else "part.div")
            )
            print(
                f"{optimizer_name:>22} {lambda_0:>10.0e} {mean['lambda_end']:>12.3e} "
                f"{mean['lambda_end'] / lambda_0:>8.3f} {mean['beta_median']:>10.4f} "
                f"{mean['beta_negative_pct']:>8.1f}% {mean['mean_return']:>9.3f} "
                f"{status:>9}"
            )
        print()

    print(
        "Read: `end/0` = 1.000 means lambda never moved, i.e. lambda_0 was already\n"
        "below p / L_local and the adaptive step is inert."
    )
    if {"pc", "pcvi"} <= set(args.optimizer):
        print(
            "\nWith both variants: wherever `end/0` is 1.000 for pcvi, pc and pcvi are\n"
            "the SAME algorithm and must agree. The gap between them at the large\n"
            "lambda_0 end is what Step 3 actually buys."
        )
        if len(args.seeds) < 5:
            print(
                f"\nWARNING: {len(args.seeds)} seed(s) and "
                f"{args.iters * args.frames_per_batch} collected frames. Enough to "
                f"locate the knee in lambda, NOT to compare returns."
            )


if __name__ == "__main__":
    main()
