#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Measure where ``lambda`` settles, to calibrate ``lr_scale`` for Adaptive.

``lambda ~ p / L`` is a step for the RAW gradient, while Adam steps by
``lr * m/sqrt(v)``, whose magnitude is ~1 per coordinate whatever the gradient
scale. The two are not in the same units, so feeding ``lambda`` straight into
Adam's ``lr`` is a category error (see KE_HOACH_IMPLEMENT_ADAPTIVE.md section 2).

``lr_scale`` reconciles them: it keeps the *shape* of the schedule Step 3
produces while putting it on a scale Adam can use. This script runs Adaptive with
``lr_scale=1.0``, reads the ``lambda`` each loss settles on, and prints the
``lr_scale`` that would put the resulting learning rate at ``--target-lr``.

Usage::

    python examples/pcvi/measure_lr_scale.py
    python examples/pcvi/measure_lr_scale.py --iters 20 --target-lr 5e-5
"""

import argparse
import pathlib
import warnings

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import AdaptiveConfig


def _output_dir(args) -> pathlib.Path:
    """Every run of every script lands under one folder, created on demand."""
    directory = pathlib.Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda0", type=float, default=0.01)
    parser.add_argument("--target-lr", type=float, default=5e-5,
                        help="the Adam lr you would otherwise have used")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--frames-per-batch", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", default="outputs",
                        help="all runs land in this folder, one subfolder each")
    parser.add_argument("--loggers", nargs="*", default=["csv"],
                        help="[] to log nothing, csv and/or wandb otherwise")
    parser.add_argument("--lipschitz-from", type=str, default="probe",
                        choices=["probe", "iterates"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

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
    config.clip_grad_val = None  # required: clipping distorts F
    config.collect_with_grad = True

    optimizer_config = AdaptiveConfig.get_from_yaml()
    optimizer_config.lambda_0 = args.lambda0
    optimizer_config.lr_scale = 1.0  # measure the raw lambda
    optimizer_config.lipschitz_from = args.lipschitz_from

    experiment = Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=args.seed,
        config=config,
    )
    experiment.run()

    minibatches = -(-args.frames_per_batch // args.minibatch_size)
    print(
        f"\nIPPO + vmas/balance + MLP | lipschitz_from={args.lipschitz_from} | "
        f"{args.iters * args.epochs * minibatches} update steps | seed {args.seed}"
    )
    print(f"{'group':>10} {'loss':>16} {'lambda_0':>10} {'lambda_end':>12} "
          f"{'end/0':>8} {'lr_scale':>10}")
    print("-" * 72)
    for group, optimizers in experiment.optimizers.items():
        for loss_name, optimizer in optimizers.items():
            lam = optimizer.lambda_k
            print(
                f"{group:>10} {loss_name:>16} {optimizer.lambda_0:>10.1e} "
                f"{lam:>12.4e} {lam / optimizer.lambda_0:>8.3f} "
                f"{args.target_lr / lam:>10.4f}"
            )
    print(
        f"\n`lr_scale` = {args.target_lr:g} / lambda_end would put the learning rate "
        f"at {args.target_lr:g}.\nIf `end/0` is 1.000 the adaptation never fired: "
        f"raise --lambda0 before reading anything into these numbers."
    )


if __name__ == "__main__":
    main()
