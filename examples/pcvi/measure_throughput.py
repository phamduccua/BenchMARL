#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Find the fastest (device, n_envs) for THIS machine, by running it.

The hours-per-Mframe numbers quoted in ``QUY_TRINH_CHAY.md`` were measured on one
particular machine (12-core CPU, Quadro P1000) and do **not** transfer: a
different GPU, a different core count or a different container changes them by
several times. Run this before committing days of compute, and use its answer.

**Why ``--n-envs`` is the lever that matters.** VMAS is a *vectorised* simulator:
all ``n_envs`` environments step as one batched tensor operation, and the policy
sees a batch of ``n_envs`` observations per forward pass. The networks here are
small MLPs, so at ``n_envs=10`` a GPU spends its time waiting on kernel launches
and transfers rather than computing, and a CPU wins. The GPU only pays off once
the batch is big enough to fill it, which means **hundreds to thousands** of
environments. Raising ``n_envs`` at a fixed ``--frames-per-batch`` does not
change how much data is collected -- it changes how much of it is collected in
parallel::

    frames_per_batch / n_envs = sequential environment steps per collection round

    120000 / 200  = 600 sequential steps  <- the current default
    120000 / 1200 = 100 sequential steps  <- same data, 6x less serialisation

Usage::

    # the sweep that answers "cpu or cuda, and how many envs?"
    python examples/pcvi/measure_throughput.py

    # narrower, e.g. once you know it is cuda
    python examples/pcvi/measure_throughput.py --devices cuda --n-envs 200 600 1200

⚠️ ``--n-envs`` is **not** a free speed knob: it changes the composition of each
collected batch, and with it the gradient noise. Whatever you pick, use the SAME
value for every branch of the ablation and report it in the write-up. Pick it
once, here, then never vary it again.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_ABLATION = _HERE / "run_ablation.py"


def _run(args, device: str, n_envs: int, optimizer: str, scratch: pathlib.Path):
    command = [
        sys.executable, str(_ABLATION),
        "--task", args.task,
        "--algorithm", args.algorithm,
        "--optimizers", optimizer,
        "--seeds", "0",
        "--iters", str(args.iters),
        "--frames-per-batch", str(args.frames_per_batch),
        "--n-envs", str(n_envs),
        "--minibatch-size", str(args.minibatch_size),
        "--epochs", str(args.epochs),
        "--episodes", "2",
        "--eval-every", str(args.iters),  # one evaluation, so it is not timed away
        "--device", device,
        "--output-dir", str(scratch),
        "--loggers",  # empty: writing csv is not what we are timing
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if device != "cpu":
        # The compute is on the GPU; letting torch also spin up one thread per
        # core (128 of them on the rented box) only adds contention. This is what
        # the project's earlier BenchMARL_2 runs did too.
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.time()
    completed = subprocess.run(
        command, cwd=str(_HERE.parent.parent), env=env,
        capture_output=True, text=True, errors="replace",
    )
    elapsed = time.time() - started
    shutil.rmtree(scratch, ignore_errors=True)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        return None, (tail[-1][:110] if tail else "failed")
    return elapsed, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="vmas/simple_tag")
    parser.add_argument("--algorithm", default="ippo")
    parser.add_argument("--optimizers", nargs="+", default=["adam", "pcvi"],
                        help="one 1-gradient and one 2-gradient branch is enough")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--n-envs", type=int, nargs="+",
                        default=[10, 60, 200, 600, 1200])
    parser.add_argument("--iters", type=int, default=2,
                        help="collection rounds per probe; 2 is enough and the "
                             "first one carries the start-up cost")
    parser.add_argument("--frames-per-batch", type=int, default=120000)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--scratch", default="outputs/_throughput")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    try:
        import torch
        if "cuda" in args.devices and not torch.cuda.is_available():
            print("cuda khong kha dung tren may nay, bo qua.")
            args.devices = [d for d in args.devices if d != "cuda"]
        elif torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    print(f"CPU cores: {os.cpu_count()}")
    frames = args.iters * args.frames_per_batch
    minibatches = -(-args.frames_per_batch // args.minibatch_size)
    updates = minibatches * args.epochs
    print(f"{args.algorithm.upper()} on {args.task} | {args.iters} rounds x "
          f"{args.frames_per_batch} frames | minibatch {args.minibatch_size} "
          f"x {args.epochs} epochs")
    print(f"= {updates} buoc gradient moi vong.")
    # The two configs the run book actually uses, from QUY_TRINH_CHAY.md.
    known = {
        450: "vmas/* (120000 frame/batch, minibatch 4096, 15 epoch)",
        150: "matrixgame/* (10000 frame/batch, minibatch 1000, 15 epoch)",
    }
    if updates in known:
        print(f"   = cau hinh {known[updates]} -- dung cai se chay that.")
    else:
        print(f"!! Khong khop cau hinh nao cua QUY_TRINH_CHAY.md "
              f"({' hoac '.join(map(str, known))} buoc/vong).")
        print(f"!! Ban dang do MOT CAU HINH KHAC voi cai se chay that.")
    print()

    print(f"{'device':>7} {'n_envs':>7} {'steps/round':>12} "
          + "".join(f"{o:>14}" for o in args.optimizers))
    print("-" * (28 + 14 * len(args.optimizers)))

    best = {}
    for device in args.devices:
        for n_envs in args.n_envs:
            cells, row_ok = [], False
            for optimizer in args.optimizers:
                scratch = pathlib.Path(args.scratch).resolve() / f"{device}_{n_envs}"
                elapsed, error = _run(args, device, n_envs, optimizer, scratch)
                if elapsed is None:
                    cells.append(f"{'FAIL':>14}")
                    continue
                row_ok = True
                hours = elapsed / frames * 1e6 / 3600
                cells.append(f"{hours:>13.2f}h")
                key = optimizer
                if key not in best or hours < best[key][0]:
                    best[key] = (hours, device, n_envs)
            steps = -(-args.frames_per_batch // n_envs)
            print(f"{device:>7} {n_envs:>7} {steps:>12} " + "".join(cells), flush=True)
            if not row_ok and error:
                print(f"{'':>28}{error}")

    print("\nGio tren 1M frame, thap hon la tot hon.\n")
    if not best:
        print("Khong cau hinh nao chay duoc.")
        return
    for optimizer, (hours, device, n_envs) in best.items():
        print(f"  Nhanh nhat cho {optimizer}: --device {device} --n-envs {n_envs} "
              f"=> {hours:.2f} h/1M frame")
    print("\nDoi chieu, may Windows 12 loi cua du an o DUNG cau hinh nay:")
    print("  cpu  n_envs=200                : adam 0.24 h, pcvi 0.42 h")
    print("  cuda n_envs=200 (Quadro P1000) : adam 0.35 h, pcvi 0.42 h\n")
    print("Chon --workers cho run_campaign.py: moi o la mot tien trinh rieng. Tren")
    print("CPU nhieu loi, thu --workers = so_loi / threads-per-worker. Tren GPU cac")
    print("o dung chung mot GPU nen phai do rieng, dung suy tu so loi CPU.\n")
    print("LUU Y: --n-envs thay doi thanh phan cua moi batch thu thap, nen no KHONG")
    print("phai mot num tang toc mien phi. Chon MOT gia tri, dung cho MOI nhanh cua")
    print("ablation, va ghi ro trong luan van.")


if __name__ == "__main__":
    main()
