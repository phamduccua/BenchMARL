#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Win rate of a trained predator faction against a RANDOM prey.

Trains (or restores) a predator-prey task for several seeds, then evaluates the
trained predator group while the prey group is replaced by a uniform random
policy, and prints::

    TRAINED IPPO-PCVI PREDATOR FACTION VS. RANDOM PREY (10 SEEDS)
    Metric        Value
    Win Rate      <mean> +/- <std>
    Catch Rate    <mean> (<min>-<max>)

The numbers are **computed from the rollouts**, not stored anywhere: what you get
depends on how long you train, which optimizer you pick and which task you run.
Treat the layout as the deliverable, not any particular value.

Definitions, both exact rather than heuristic (see
:class:`~benchmarl.experiment.metrics.WinRateCallback`):

* a **catch** is a predator-prey collision. With ``shape_adversary_rew: False``
  -- the shipped ``vmas/simple_tag`` config -- the adversary reward at a timestep
  is exactly ``10 x (number of collisions)``, so catches are read off the reward
  with no ambiguity.
* an episode is a **win** if the predators land at least ``--min-catches`` catches.
* **catch rate** is the fraction of timesteps in an episode at which a catch
  happened, averaged over episodes.

Why a random prey: evaluating the predators against the prey they co-trained
against measures the pair, not the predators. Freezing the opponent to uniform
random gives a fixed reference that is comparable across seeds and algorithms.

Usage::

    # quick smoke run
    python examples/pcvi/evaluate_win_rate.py --seeds 0 1 --iters 5

    # the real thing
    python examples/pcvi/evaluate_win_rate.py --optimizer pcvi --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --iters 500 --frames-per-batch 6000 --n-envs 10
"""

import argparse
import pathlib
import statistics
import sys
import warnings

import torch
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.metrics import WinRateCallback
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import optimizer_config_registry


def _output_dir(args) -> pathlib.Path:
    """Every run of every script lands under one folder, created on demand."""
    directory = pathlib.Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_experiment(args, seed: int) -> Experiment:
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = config.train_device = config.buffer_device = args.device
    config.max_n_iters = args.iters
    config.max_n_frames = None
    config.on_policy_collected_frames_per_batch = args.frames_per_batch
    config.on_policy_n_envs_per_worker = args.n_envs
    config.on_policy_minibatch_size = args.minibatch_size
    config.on_policy_n_minibatch_iters = args.epochs
    config.evaluation = False  # we run our own evaluation below
    config.render = False
    config.save_folder = str(_output_dir(args))
    config.loggers = list(args.loggers)
    config.create_json = False
    config.checkpoint_interval = 0
    config.restore_file = args.restore_file

    optimizer_config = optimizer_config_registry[args.optimizer].get_from_yaml()
    if optimizer_config.uses_gradient_as_operator():
        # a clipped gradient is not the gradient of anything, which breaks the
        # Lipschitz estimate these optimizers are built on
        config.clip_grad_val = None

    return Experiment(
        task=VmasTask[args.task.upper()].get_from_yaml(),
        algorithm_config=algorithm_config_registry[args.algorithm].get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=seed,
        config=config,
    )


def make_mixed_policy(experiment: Experiment, args):
    """Trained policy for the predators, uniform random for the prey."""
    predator_policy = experiment.group_policies[args.predator_group]
    prey_action_key = (args.prey_group, "action")
    action_spec = experiment.test_env.full_action_spec[prey_action_key]

    def policy(tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = predator_policy(tensordict)
        tensordict.set(prey_action_key, action_spec.rand())
        return tensordict

    return policy


def evaluate(experiment: Experiment, args):
    """Rollouts of the trained predators against a random prey."""
    metric = WinRateCallback(
        predator_group=args.predator_group, min_catches_to_win=args.min_catches
    )
    metric.experiment = experiment
    metric.on_setup()

    policy = make_mixed_policy(experiment, args)
    stats = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(args.episodes):
            rollout = experiment.test_env.rollout(
                max_steps=experiment.max_steps,
                policy=policy,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
            stats.append(metric.episode_stats(rollout))

    n = len(stats)
    return {
        "win_rate": sum(s["won"] for s in stats) / n,
        "catch_rate": sum(s["catch_rate"] for s in stats) / n,
        "catches": sum(s["catches"] for s in stats) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="simple_tag")
    parser.add_argument("--algorithm", default="ippo",
                        choices=sorted(algorithm_config_registry))
    parser.add_argument("--optimizer", default="pcvi",
                        choices=sorted(optimizer_config_registry))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=100,
                        help="evaluation episodes per seed")
    parser.add_argument("--min-catches", type=int, default=1,
                        help="catches needed for an episode to count as a win")
    parser.add_argument("--predator-group", default="adversary")
    parser.add_argument("--prey-group", default="agent")
    parser.add_argument("--iters", type=int, default=100, help="collection rounds")
    parser.add_argument("--frames-per-batch", type=int, default=6000)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs",
                        help="all runs land in this folder, one subfolder each")
    parser.add_argument("--loggers", nargs="*", default=["csv"],
                        help="[] to log nothing, csv and/or wandb otherwise")
    parser.add_argument("--restore-file", default=None,
                        help="skip training and evaluate this checkpoint instead")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    per_seed = []
    for seed in args.seeds:
        experiment = build_experiment(args, seed)
        if args.restore_file is None:
            experiment.run()
        result = evaluate(experiment, args)
        per_seed.append(result)
        print(
            f"  seed {seed:>3}: win_rate={result['win_rate']:.3f}  "
            f"catch_rate={result['catch_rate']:.3f}  "
            f"catches/ep={result['catches']:.2f}",
            flush=True,
        )

    def summary(key):
        values = [r[key] for r in per_seed]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return values, statistics.fmean(values), std

    win, win_mean, win_std = summary("win_rate")
    catch, catch_mean, _ = summary("catch_rate")

    # Windows consoles default to cp1252, which cannot encode the table's symbols
    # and drops the rows silently. Force UTF-8, and fall back to ASCII if the
    # stream will not take it.
    plus_minus, dash = "±", "–"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        plus_minus, dash = "+/-", "-"

    label = f"{args.algorithm.upper()}-{args.optimizer.upper()}"
    print()
    print(
        f"TRAINED {label} PREDATOR FACTION VS. RANDOM PREY "
        f"({len(args.seeds)} SEEDS)"
    )
    print(f"{'Metric':<14}{'Value'}")
    print(f"{'Win Rate':<14}{win_mean:.3f} {plus_minus} {win_std:.3f}")
    print(
        f"{'Catch Rate':<14}{catch_mean:.3f} "
        f"({min(catch):.2f}{dash}{max(catch):.2f})"
    )
    print()
    print(
        f"Task {args.task}, {args.episodes} evaluation episodes per seed, "
        f"{args.iters} collection rounds of {args.frames_per_batch} frames.\n"
        f"A win is >= {args.min_catches} catch(es); a catch is a predator-prey "
        f"collision read exactly off the unshaped adversary reward."
    )


if __name__ == "__main__":
    main()
