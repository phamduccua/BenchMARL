#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Run the optimizer ablation and print a comparison table.

Same algorithm, same task, same seeds, same everything except the optimizer::

    IPPO              optimizer=adam    baseline
    IPPO + PC         optimizer=pc      contraction only, lambda fixed
    IPPO + Adaptive   optimizer=adaptive  adaptive lambda only, Adam kept
    IPPO + PCVI       optimizer=pcvi    both, i.e. the full Algorithm 1

Usage::

    python examples/pcvi/run_ablation.py --smoke
    python examples/pcvi/run_ablation.py --optimizers adam pc adaptive pcvi \\
        --seeds 0 1 2 3 4 --iters 500 --frames-per-batch 6000 --n-envs 10

Read the caveats in HUONG_DAN_CHAY.md section 5.2 before drawing conclusions:

* pc / pcvi drop Adam's preconditioner, so they need an SGD branch to separate
  "the algorithm is worse" from "there is no per-coordinate normalisation";
* adaptive is in effect a learning-rate decay schedule, so it needs an
  Adam + cosine-decay branch to separate the Lipschitz mechanism from decay;
* everything but adam and lookahead(adam) costs **two** gradients per update, so
  a fixed number of iterations is not a fixed compute budget. ``--half-epochs``
  runs the two-gradient branches at half the epochs to equalise it.
"""

import argparse
import pathlib
import statistics
import sys
import time
import warnings

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.metrics import NashDistanceCallback, WinRateCallback
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import optimizer_config_registry


def _output_dir(args) -> pathlib.Path:
    """Every run of every script lands under one folder, created on demand."""
    directory = pathlib.Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cast(value: str, current):
    """Casts a command-line string to the type the config field already holds."""
    if isinstance(current, bool) or value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none"):
        return None
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _apply_overrides(optimizer_config, overrides, optimizer_name):
    """``--optimizer-overrides beta=1.5 lookahead.inner=pcvi``.

    A bare ``field=value`` goes to every branch that has that field; branches
    without it are left alone, so one command can mix optimizers with different
    knobs. ``name.field=value`` goes only to that optimizer, and is an error if
    the field does not exist there -- that is where a typo gets caught.
    """
    for key, value in overrides.items():
        target, _, field = key.rpartition(".")
        if target and target != optimizer_name:
            continue
        if not hasattr(optimizer_config, field):
            if target:
                raise SystemExit(
                    f"{type(optimizer_config).__name__} has no field {field!r}. "
                    f"Available: {sorted(optimizer_config.__dict__)}"
                )
            continue  # unscoped: this branch simply does not have the knob
        setattr(optimizer_config, field, _cast(value, getattr(optimizer_config, field)))


def build(args, optimizer_name: str, seed: int):
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = config.train_device = config.buffer_device = args.device
    config.max_n_iters = args.iters
    config.max_n_frames = None
    config.on_policy_collected_frames_per_batch = args.frames_per_batch
    config.on_policy_n_envs_per_worker = args.n_envs
    config.on_policy_minibatch_size = args.minibatch_size
    config.on_policy_n_minibatch_iters = args.epochs
    config.evaluation = True
    config.evaluation_interval = args.frames_per_batch * max(1, args.iters // 4)
    config.evaluation_episodes = args.episodes
    config.render = False
    config.save_folder = str(_output_dir(args))
    config.loggers = list(args.loggers)
    config.create_json = False
    config.checkpoint_interval = 0

    optimizer_config = optimizer_config_registry[optimizer_name].get_from_yaml()
    # Overrides FIRST: `lookahead.inner=pcvi` changes whether this branch needs a
    # second gradient and whether clipping has to go, so the capability queries
    # below have to see the final config.
    if optimizer_name in args.lambda0 and hasattr(optimizer_config, "lambda_0"):
        optimizer_config.lambda_0 = args.lambda0[optimizer_name]
    _apply_overrides(optimizer_config, args.optimizer_overrides, optimizer_name)

    two_gradient = optimizer_config.requires_two_gradient_evals()
    if optimizer_config.uses_gradient_as_operator():
        config.clip_grad_val = None
    if args.half_epochs and two_gradient:
        # equal compute rather than equal number of updates
        config.on_policy_n_minibatch_iters = max(1, args.epochs // 2)

    callbacks = []
    if args.task.startswith("matrixgame/"):
        callbacks.append(NashDistanceCallback())
    elif args.task in ("vmas/simple_tag", "vmas/simple_world_comm"):
        callbacks.append(WinRateCallback(predator_group=args.predator_group))

    experiment = Experiment(
        task=task_config_registry[args.task].get_from_yaml(),
        algorithm_config=algorithm_config_registry[args.algorithm].get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=seed,
        config=config,
        callbacks=callbacks,
    )
    return experiment, two_gradient


def run_one(args, optimizer_name: str, seed: int):
    experiment, two_gradient = build(args, optimizer_name, seed)
    started = time.time()
    experiment.run()
    result = {
        "return": experiment.mean_return,
        "seconds": time.time() - started,
        "two_gradient": two_gradient,
    }
    # optimizer-specific diagnostics, if the optimizer exposes them
    optimizer = next(iter(next(iter(experiment.optimizers.values())).values()))
    for attribute in ("lambda_k", "lambda_0", "n_syncs"):
        if hasattr(optimizer, attribute):
            result[attribute] = getattr(optimizer, attribute)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="vmas/simple_tag",
                        choices=sorted(task_config_registry))
    parser.add_argument("--algorithm", default="ippo",
                        choices=sorted(algorithm_config_registry))
    parser.add_argument("--optimizers", nargs="+",
                        default=["adam", "pc", "adaptive", "pcvi"],
                        choices=sorted(optimizer_config_registry))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--frames-per-batch", type=int, default=6000)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--predator-group", default="adversary")
    parser.add_argument("--loggers", nargs="*", default=["csv"])
    parser.add_argument("--output-dir", default="outputs",
                        help="all runs land in this folder, one subfolder each")
    parser.add_argument("--half-epochs", action="store_true",
                        help="halve the epochs of two-gradient optimizers, so every "
                             "branch spends the same number of gradient evaluations")
    parser.add_argument("--lambda0", nargs="*", default=[],
                        metavar="NAME=VALUE",
                        help="per-optimizer lambda_0, e.g. pcvi=0.01 pc=0.001")
    parser.add_argument("--optimizer-overrides", nargs="*", default=[],
                        metavar="FIELD=VALUE",
                        help="optimizer config fields. FIELD=VALUE goes to every branch "
                             "that has it; NAME.FIELD=VALUE only to that "
                             "optimizer. e.g. beta=1.5 lookahead.inner=pcvi")
    parser.add_argument("--zip", action="store_true",
                        help="zip the output folder when the runs finish")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny settings, just to check the pipeline runs")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    args.lambda0 = dict(
        (name, float(value))
        for name, value in (item.split("=") for item in args.lambda0)
    )
    args.optimizer_overrides = dict(
        item.split("=", 1) for item in args.optimizer_overrides
    )
    if args.smoke:
        args.seeds = args.seeds[:2]
        args.iters, args.frames_per_batch, args.n_envs = 3, 300, 2
        args.minibatch_size, args.epochs, args.episodes = 150, 2, 4

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        plus_minus = "±"
    except (AttributeError, ValueError):
        plus_minus = "+/-"

    print(
        f"{args.algorithm.upper()} on {args.task} | seeds {args.seeds} | "
        f"{args.iters} rounds x {args.frames_per_batch} frames"
        + (" | equal-compute (--half-epochs)" if args.half_epochs else "")
    )
    if args.optimizer_overrides:
        for key in args.optimizer_overrides:
            target, _, field = key.rpartition(".")
            if target:
                continue
            if not any(
                hasattr(optimizer_config_registry[name].get_from_yaml(), field)
                for name in args.optimizers
            ):
                raise SystemExit(
                    f"No optimizer in {args.optimizers} has a field {field!r}."
                )
        print(f"  overrides: {args.optimizer_overrides}")
    results = {}
    for optimizer_name in args.optimizers:
        per_seed = []
        for seed in args.seeds:
            result = run_one(args, optimizer_name, seed)
            per_seed.append(result)
            print(
                f"  {optimizer_name:<24} seed {seed:>3}: "
                f"return={result['return']:>9.3f}  {result['seconds']:>6.1f}s",
                flush=True,
            )
        results[optimizer_name] = per_seed

    print()
    print(
        f"{'optimizer':<24}{'return':>20}{'sec/run':>10}{'grad/step':>11}"
        f"{'lambda_end':>13}"
    )
    print("-" * 78)
    for optimizer_name, per_seed in results.items():
        returns = [r["return"] for r in per_seed]
        mean = statistics.fmean(returns)
        std = statistics.stdev(returns) if len(returns) > 1 else 0.0
        seconds = statistics.fmean(r["seconds"] for r in per_seed)
        grads = 2 if per_seed[0]["two_gradient"] else 1
        lambdas = [r["lambda_k"] for r in per_seed if "lambda_k" in r]
        lambda_cell = f"{statistics.fmean(lambdas):.3e}" if lambdas else "-"
        print(
            f"{optimizer_name:<24}{mean:>11.3f} {plus_minus} {std:<6.3f}"
            f"{seconds:>10.1f}{grads:>11}{lambda_cell:>13}"
        )

    print()
    print(
        "Reminders before reading anything into this (HUONG_DAN_CHAY.md section 5.2):\n"
        "  * lambda_end == lambda_0 means the adaptive mechanism never fired.\n"
        "  * grad/step 2 branches cost twice as much per update; rerun with\n"
        "    --half-epochs for an equal-compute comparison.\n"
        "  * add an SGD branch and an Adam+decay branch before attributing any\n"
        "    difference to the algorithm rather than to the preconditioner or to\n"
        "    the learning-rate schedule."
    )
    if len(args.seeds) < 5:
        print(f"  * {len(args.seeds)} seed(s) is not enough to compare returns.")

    if args.zip:
        import subprocess

        print()
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).with_name("zip_results.py")),
             "--output-dir", args.output_dir],
            check=False,
        )


if __name__ == "__main__":
    main()
