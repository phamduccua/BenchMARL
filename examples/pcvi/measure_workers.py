#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""How many cells should run at once on THIS machine? Measure it.

``measure_throughput.py`` answers "how fast is one cell"; this answers "how many
cells at a time", which on a rented box is the bigger lever. It runs the *same*
set of cells at several ``--workers`` values and reports the wall clock, the
speedup over one worker, and -- on a GPU -- the peak memory and utilisation
sampled from ``nvidia-smi`` while it ran.

Why it cannot be reasoned out. One cell was seen using 2.8 GB of a 24 GB 3090 and
11% of it, which suggests eight cells would fit. But cells contend for streaming
multiprocessors, for the PCIe bus and for the CPU feeding them, and VRAM per cell
grows with ``--n-envs``. The only honest number is a measured one, and it is
specific to this machine.

Usage::

    # the default sweep, on the config the campaign will actually run
    python examples/pcvi/measure_workers.py

    # narrower, if you already know 8 is too many
    python examples/pcvi/measure_workers.py --workers 1 2 4

Each probe runs ``--cells`` cells of ``--iters`` rounds. The default is
deliberately short: it is measuring contention, not training anything. Nothing it
writes is kept.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

_HERE = pathlib.Path(__file__).resolve().parent
_CAMPAIGN = _HERE / "run_campaign.py"


class GpuSampler(threading.Thread):
    """Polls nvidia-smi in the background and keeps the peaks."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak_memory = 0
        self.peak_util = 0
        self.samples = 0
        self._stop = threading.Event()

    def run(self):
        query = ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                out = subprocess.run(query, capture_output=True, text=True,
                                     timeout=10).stdout.strip().splitlines()[0]
                memory, util = (int(x.strip()) for x in out.split(","))
                self.peak_memory = max(self.peak_memory, memory)
                self.peak_util = max(self.peak_util, util)
                self.samples += 1
            except Exception:
                pass  # no nvidia-smi, or a transient failure: just report nothing
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()


def _probe(args, workers: int, scratch: pathlib.Path):
    command = [
        sys.executable, str(_CAMPAIGN),
        "--task", args.task,
        "--optimizers", args.optimizer,
        "--seeds", *[str(s) for s in range(args.cells)],
        "--iters", str(args.iters),
        "--frames-per-batch", str(args.frames_per_batch),
        "--n-envs", str(args.n_envs),
        "--minibatch-size", str(args.minibatch_size),
        "--epochs", str(args.epochs),
        "--episodes", "2",
        "--eval-every", str(args.iters),
        "--device", args.device,
        "--workers", str(workers),
        "--output-dir", str(scratch),
    ]
    shutil.rmtree(scratch, ignore_errors=True)
    sampler = GpuSampler()
    if args.device != "cpu":
        sampler.start()
    started = time.time()
    completed = subprocess.run(
        command, cwd=str(_HERE.parent.parent),
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        capture_output=True, text=True, errors="replace",
    )
    elapsed = time.time() - started
    sampler.stop()

    done = len(list(scratch.glob("*/done.json"))) if scratch.is_dir() else 0
    shutil.rmtree(scratch, ignore_errors=True)
    error = ""
    # Only a missing cell is a failure. run_campaign prints its own summary and
    # can exit non-zero for reasons that do not mean the probe was invalid.
    if done < args.cells:
        tail = (completed.stdout or completed.stderr).strip().splitlines()
        error = next((l for l in reversed(tail) if "Error" in l or "FAIL" in l),
                     tail[-1] if tail else "failed")[:100]
    return elapsed, done, sampler, error


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="vmas/simple_tag")
    parser.add_argument("--optimizer", default="pcvi",
                        help="one branch is enough; pcvi is the expensive kind")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--cells", type=int, default=8,
                        help="cells per probe; should be >= the largest --workers")
    parser.add_argument("--iters", type=int, default=2,
                        help="rounds per cell. Short on purpose: this measures "
                             "contention, not training")
    parser.add_argument("--frames-per-batch", type=int, default=120000)
    parser.add_argument("--n-envs", type=int, default=1200)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scratch", default="outputs/_workers")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if max(args.workers) > args.cells:
        raise SystemExit(
            f"--cells {args.cells} is fewer than the largest --workers "
            f"{max(args.workers)}; the extra workers would sit idle and the "
            f"measurement would be meaningless."
        )

    print(f"{args.optimizer} on {args.task} | {args.cells} o x {args.iters} vong x "
          f"{args.frames_per_batch} frame | n_envs={args.n_envs} device={args.device}")
    print(f"Moi lan do chay LAI dung {args.cells} o do, chi khac so worker.\n")

    header = f"{'workers':>8} {'wall clock':>12} {'/ 1 worker':>12} {'o xong':>8}"
    if args.device != "cpu":
        header += f" {'VRAM dinh':>11} {'GPU dinh':>9}"
    print(header)
    print("-" * len(header))

    baseline = None
    best = (0.0, None)
    scratch = pathlib.Path(args.scratch).resolve()
    for workers in sorted(args.workers):
        elapsed, done, sampler, error = _probe(args, workers, scratch)
        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed if elapsed > 0 else 0.0
        row = (f"{workers:>8} {elapsed:>11.1f}s {speedup:>11.2f}x "
               f"{done:>4}/{args.cells}")
        if args.device != "cpu":
            memory = f"{sampler.peak_memory} MiB" if sampler.samples else "-"
            util = f"{sampler.peak_util}%" if sampler.samples else "-"
            row += f" {memory:>11} {util:>9}"
        print(row + (f"   {error}" if error else ""), flush=True)
        if done == args.cells and speedup > best[0]:
            best = (speedup, workers)

    print()
    if best[1] is None:
        print("Khong cau hinh nao chay tron ven. Giam --n-envs hoac --cells.")
        return
    print(f"=> Nhanh nhat: --workers {best[1]} ({best[0]:.2f}x so voi 1 worker)")
    print()
    print("Doc bang tren:")
    print("  * 'o xong' < so o  => het VRAM hoac loi khac. Giam --workers.")
    print("  * speedup ngung tang => da bao hoa; them worker chi ton VRAM.")
    print("  * VRAM dinh gan 24000 MiB => nguy hiem, giam --workers mot bac.")
    print()
    print("Dua so nay vao --workers cua moi lenh trong QUY_TRINH_CHAY.md.")


if __name__ == "__main__":
    main()
