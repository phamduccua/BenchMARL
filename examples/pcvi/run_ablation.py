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
from dataclasses import fields
import pathlib
import statistics
import sys
import time
import warnings

import torch

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


def _apply_algorithm_overrides(algorithm_config, overrides):
    """``--algorithm-overrides entropy_coef=0.01 clip_epsilon=0.2``.

    The field that matters most here is ``entropy_coef``: BenchMARL ships it at
    0.0, and on a game whose equilibrium is a fully mixed strategy that lets the
    policy collapse to a pure one, which is the maximally exploitable point.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"--algorithm-overrides takes FIELD=VALUE, got {override!r}"
            )
        field, _, raw = override.partition("=")
        if not hasattr(algorithm_config, field):
            available = ", ".join(sorted(f.name for f in fields(algorithm_config)))
            raise ValueError(
                f"{type(algorithm_config).__name__} has no field {field!r}. "
                f"Available: {available}"
            )
        setattr(algorithm_config, field, _cast(raw, getattr(algorithm_config, field)))
    return algorithm_config


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


def _apply_experiment_overrides(config, overrides):
    """``FIELD=VALUE`` pairs applied to the ExperimentConfig.

    gamma, lmbda and lr live on ExperimentConfig, not on the algorithm or the
    optimizer, so before this flag existed they could only be changed by editing
    base_experiment.yaml. gamma in particular is load-bearing: the matrix game is
    stateless, so the correct discount is 0 and the default 0.99 mixes 31 rounds
    of independent rewards into every advantage.
    """
    for pair in overrides:
        field, _, value = pair.partition("=")
        if not hasattr(config, field):
            raise SystemExit(
                f"ExperimentConfig has no field {field!r}. "
                f"Available: {sorted(f.name for f in fields(config))}"
            )
        setattr(config, field, _cast(value, getattr(config, field)))


def _apply_task_overrides(task, overrides):
    """``KEY=VALUE`` pairs applied to the task config dict (max_steps, ...)."""
    for pair in overrides:
        key, _, value = pair.partition("=")
        if key not in task.config:
            raise SystemExit(
                f"Task {type(task).__name__} has no config key {key!r}. "
                f"Available: {sorted(task.config)}"
            )
        task.config[key] = _cast(value, task.config[key])


def _perturb_initial_policy(experiment, sigma: float, seed: int) -> dict:
    """Moves the starting policy away from the uniform distribution.

    On a matrix game the equilibrium is the uniform mixed strategy, and a freshly
    initialised softmax head with small weights IS approximately uniform: measured
    on rock_paper_scissors, ``nash_conv`` starts at 0.17--0.32 against a ceiling of
    2.0. The run therefore begins next to the answer, and an optimizer that does
    nothing at all scores better than one that learns -- which is not the question
    being asked. Adding N(0, sigma) to the bias of each policy's last linear layer
    starts every branch at the same measured distance from Nash instead.

    Not part of the paper or of BenchMARL: report the sigma used, and report the
    round-0 ``nash_conv`` it produces, in anything written up from these runs.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed + 90210)
    moved = {}
    for group, policy in experiment.group_policies.items():
        layers = [m for m in policy.modules() if isinstance(m, torch.nn.Linear)]
        if not layers or layers[-1].bias is None:
            raise SystemExit(
                f"--init-bias: no final Linear with a bias in the policy of group "
                f"{group!r}; the perturbation would silently do nothing."
            )
        bias = layers[-1].bias
        noise = torch.randn(
            bias.shape, generator=generator, dtype=bias.dtype
        ).to(bias.device)
        with torch.no_grad():
            bias.add_(noise * sigma)
        moved[group] = tuple(bias.shape)
    return moved


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
    # Every `eval_every` collection rounds. The default of iters // 4 gives FOUR
    # evaluation points whatever the length of the run, which is enough to say
    # "it ended higher than it started" and not enough to plot a learning curve
    # of eval_win_rate or eval_nash_conv. Pass --eval-every to make it dense.
    eval_every = args.eval_every or max(1, args.iters // 4)
    config.evaluation_interval = args.frames_per_batch * eval_every
    config.evaluation_episodes = args.episodes
    config.render = False
    config.save_folder = str(_output_dir(args))
    config.loggers = list(args.loggers)
    config.create_json = False
    config.checkpoint_interval = 0
    _apply_experiment_overrides(config, args.experiment_overrides)

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
        # A mixed Nash equilibrium cannot be reached by argmax actions: with
        # BenchMARL's default deterministic evaluation the measured policy is
        # one-hot, and dist_nash / nash_conv sit at their maximum for the whole
        # run whatever the policy learns.
        config.evaluation_deterministic_actions = False
        callbacks.append(NashDistanceCallback())
    elif args.task in ("vmas/simple_tag", "vmas/simple_world_comm"):
        callbacks.append(WinRateCallback(predator_group=args.predator_group))

    task = task_config_registry[args.task].get_from_yaml()
    _apply_task_overrides(task, args.task_overrides)

    experiment = Experiment(
        task=task,
        algorithm_config=_apply_algorithm_overrides(
            algorithm_config_registry[args.algorithm].get_from_yaml(),
            args.algorithm_overrides,
        ),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=seed,
        config=config,
        callbacks=callbacks,
    )
    if args.init_bias:
        moved = _perturb_initial_policy(experiment, args.init_bias, seed)
        print(
            f"  init-bias sigma={args.init_bias} applied to "
            + ", ".join(f"{g}{tuple(shape)}" for g, shape in moved.items())
        )
    return experiment, two_gradient


# The number worth putting in the summary table, per task family. `mean_return`
# is not it: the logger averages the episode reward OVER GROUPS, and simple_tag's
# two groups collect +10 and -10 for the very same collision, so it sits at ~0
# however well the predators learn.
_TASK_METRIC = {
    "vmas/simple_tag": ("catches_per_episode", "catch/ep"),
    "vmas/simple_world_comm": ("catches_per_episode", "catch/ep"),
    "matrixgame/rock_paper_scissors": ("nash_conv", "nash_conv"),
    "matrixgame/matching_pennies": ("nash_conv", "nash_conv"),
}


def run_one(args, optimizer_name: str, seed: int):
    experiment, two_gradient = build(args, optimizer_name, seed)
    started = time.time()
    experiment.run()
    result = {
        "return": experiment.mean_return,
        "seconds": time.time() - started,
        "two_gradient": two_gradient,
    }
    key, _ = _TASK_METRIC.get(args.task, (None, None))
    if key is not None:
        for callback in experiment.callbacks:
            if key in getattr(callback, "last_stats", {}):
                result["metric"] = callback.last_stats[key]
                break
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
    parser.add_argument("--iters", type=int, default=17,
                        help="collection rounds; 17 x 120000 = 2.04M frames")
    parser.add_argument("--frames-per-batch", type=int, default=120000)
    parser.add_argument("--n-envs", type=int, default=1200,
                        help="parallel VMAS environments. 1200 makes "
                             "frames_per_batch/n_envs = 100, which is simple_tag's "
                             "max_steps, so each env completes exactly one full "
                             "episode per collection round and no advantage has to "
                             "be bootstrapped from a truncation")
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=None,
                        metavar="ROUNDS",
                        help="evaluate every ROUNDS collection rounds. Default is "
                             "iters // 4, i.e. four evaluation points however long "
                             "the run is -- too coarse to plot eval_win_rate or "
                             "eval_nash_conv against frames. Costs "
                             "--episodes rollouts each time.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--predator-group", default="adversary")
    parser.add_argument("--loggers", nargs="*", default=["csv"])
    parser.add_argument("--output-dir", default="outputs",
                        help="all runs land in this folder, one subfolder each")
    parser.add_argument("--experiment-overrides", nargs="*", default=[],
                        metavar="FIELD=VALUE",
                        help="fields of ExperimentConfig: gamma, lmbda, lr, ... "
                             "The matrix game is stateless, so gamma=0 is the "
                             "correct discount there, not the default 0.99.")
    parser.add_argument("--task-overrides", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="keys of the task yaml, e.g. max_steps=200")
    parser.add_argument("--init-bias", type=float, default=0.0, metavar="SIGMA",
                        help="add N(0, SIGMA) to the bias of each policy's last "
                             "linear layer before training. On a matrix game the "
                             "default init is already ~uniform = ~Nash, so a "
                             "branch that does not move wins; use e.g. 2.0 to "
                             "start far from the equilibrium. NOT in the paper.")
    parser.add_argument("--half-epochs", action="store_true",
                        help="halve the epochs of two-gradient optimizers, so every "
                             "branch spends the same number of gradient evaluations")
    parser.add_argument("--lambda0", nargs="*", default=[],
                        metavar="NAME=VALUE",
                        help="per-optimizer lambda_0, e.g. pcvi=0.01 pc=0.001")
    parser.add_argument("--algorithm-overrides", nargs="*", default=[],
                        metavar="FIELD=VALUE",
                        help="algorithm config fields, e.g. entropy_coef=0.01. "
                             "BenchMARL ships entropy_coef=0.0, which on a game "
                             "with a fully mixed Nash lets the policy collapse "
                             "to a pure strategy")
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
    _, metric_label = _TASK_METRIC.get(args.task, (None, None))
    metric_label = metric_label or "-"
    print(
        f"{'optimizer':<24}{metric_label:>20}{'return':>10}{'sec/run':>10}"
        f"{'grad/step':>11}{'lambda_end':>13}"
    )
    print("-" * 88)
    for optimizer_name, per_seed in results.items():
        metrics = [r["metric"] for r in per_seed if "metric" in r]
        if metrics:
            mean = statistics.fmean(metrics)
            std = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
            metric_cell = f"{mean:>11.3f} {plus_minus} {std:<6.3f}"
        else:
            metric_cell = f"{'-':>20}"
        seconds = statistics.fmean(r["seconds"] for r in per_seed)
        grads = 2 if per_seed[0]["two_gradient"] else 1
        lambdas = [r["lambda_k"] for r in per_seed if "lambda_k" in r]
        lambda_cell = f"{statistics.fmean(lambdas):.3e}" if lambdas else "-"
        returns = statistics.fmean(r["return"] for r in per_seed)
        print(
            f"{optimizer_name:<24}{metric_cell}{returns:>10.3f}"
            f"{seconds:>10.1f}{grads:>11}{lambda_cell:>13}"
        )

    print()
    print(
        "Reminders before reading anything into this (HUONG_DAN_CHAY.md section 5.2):\n"
        "  * lambda_end == lambda_0 means the adaptive mechanism never fired.\n"
        "  * grad/step 2 branches cost twice as much per update; rerun with\n"
        "    --half-epochs for an equal-compute comparison.\n"
        "  * include `--optimizers sgd adam_cosine` before attributing any\n"
        "    difference to the algorithm rather than to the preconditioner\n"
        "    (sgd) or to the learning-rate schedule (adam_cosine).\n"
        "  * `return` averages the episode reward OVER GROUPS. On simple_tag the\n"
        "    two groups earn +10 and -10 for the same collision, so it sits at\n"
        "    ~0 however well the predators learn: read the first column instead."
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
