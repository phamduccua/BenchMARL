#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Run the full real-scale ablation as a resumable, parallel campaign.

``run_ablation.py`` runs every (optimizer, seed) pair in one process, in order.
At the real scale that is measured in days, and a single crash or Ctrl-C loses
everything. This script runs the same grid as **one subprocess per cell**, so
that

* a finished cell is never recomputed -- rerun the same command to resume;
* several cells run at once, which is what actually shortens the wall clock,
  because a single run does not saturate the machine;
* one cell dying (a NaN divergence, an OOM) does not take the campaign with it.

Each cell writes ``<output-dir>/<optimizer>_seed<seed>/`` containing ``log.txt``
(the full stdout of that run) and, on success, ``done.json`` with the mean
return and the wall clock. ``done.json`` is the resume marker: delete it to
force a recompute.

Usage::

    # the real thing: 3M frames = 500 rounds x 6000 frames, 5 seeds
    python examples/pcvi/run_campaign.py --seeds 0 1 2 3 4 --iters 500 --workers 4

    # what it would cost, without running anything
    python examples/pcvi/run_campaign.py --dry-run

Threads: each worker is pinned to ``--threads-per-worker`` CPU threads, because
N unpinned torch processes on one machine oversubscribe the cores and end up
slower than running them one at a time.
"""

import argparse
import json
import os
import pathlib
import queue
import statistics
import subprocess
import sys
import threading
import time

from benchmarl.optimizers import optimizer_config_registry

_HERE = pathlib.Path(__file__).resolve().parent
_ABLATION = _HERE / "run_ablation.py"
# Hours per 1M frames, keyed by device then by gradients per update. All measured
# with measure_throughput.py at the CURRENT default config -- frames_per_batch
# 120000, minibatch 4096, 15 epochs, i.e. 450 updates per 120000 frames.
#
# On the RTX 3090 the numbers fall until n_envs 2400 and then stop: 200 -> 0.12,
# 600 -> 0.07, 1200 -> 0.05, 2400/4800/9600 -> 0.04 (adam). 1200 is the default
# because it is the largest value that still collects one whole episode per env
# per round, and the 20% left on the table above it is not worth truncating every
# batch mid-episode for.
#
# For scale: the old config (10 envs, 675 updates per 6000 frames) cost 1.91 and
# 3.09 h/Mframe on a 12-core CPU, and 5.92 / 9.52 on the rented box's CPU.
_HOURS_PER_MFRAME = {
    "cuda": {  # RTX 3090, n_envs 1200
        "vmas/simple_tag": {1: 0.05, 2: 0.08},
    },
    "cpu": {  # 12-core Windows box, n_envs 200
        "vmas/simple_tag": {1: 0.24, 2: 0.42},
    },
}
_UNMEASURED_TASK = "vmas/simple_tag"  # what an unmeasured task is quoted at
# 4 cells at 3 threads each finished in 209.3 s against 414.8 s one at a time on
# 12 cores. Not 4x: the cells contend. Re-measure on a different machine.
_MEASURED_SPEEDUP = 1.98
# The config the rates above were measured at: 15 * ceil(120000/4096) = 450
# updates per 120000 frames. A different config is rescaled by this ratio, which
# only corrects the gradient part -- the environment simulation is unaffected by
# it -- so the further you move from here, the rougher the estimate.
_REFERENCE_UPDATES_PER_FRAME = 450 / 120000


def _cell_dir(args, optimizer: str, seed: int) -> pathlib.Path:
    return pathlib.Path(args.output_dir).resolve() / f"{optimizer}_seed{seed}"


def _is_done(cell: pathlib.Path) -> bool:
    return (cell / "done.json").is_file()


def _two_gradient(optimizer: str) -> bool:
    config = optimizer_config_registry[optimizer].get_from_yaml()
    return config.requires_two_gradient_evals()


def _command(args, optimizer: str, seed: int, cell: pathlib.Path):
    command = [
        sys.executable, str(_ABLATION),
        "--task", args.task,
        "--algorithm", args.algorithm,
        "--optimizers", optimizer,
        "--seeds", str(seed),
        "--iters", str(args.iters),
        "--frames-per-batch", str(args.frames_per_batch),
        "--n-envs", str(args.n_envs),
        "--minibatch-size", str(args.minibatch_size),
        "--epochs", str(args.epochs),
        "--episodes", str(args.episodes),
        "--eval-every", str(args.eval_every),
        "--device", args.device,
        "--output-dir", str(cell),
    ]
    if args.half_epochs:
        command.append("--half-epochs")
    if args.lambda0:
        # One flag with every value: run_ablation declares --lambda0 as nargs="*",
        # so repeating the flag REPLACES the list instead of extending it, and
        # every branch but the last would silently lose its lambda_0.
        command.append("--lambda0")
        command += args.lambda0
    if args.optimizer_overrides:
        command.append("--optimizer-overrides")
        command += args.optimizer_overrides
    return command


def _run_cell(args, optimizer: str, seed: int) -> dict:
    cell = _cell_dir(args, optimizer, seed)
    cell.mkdir(parents=True, exist_ok=True)
    threads = str(args.threads_per_worker)
    env = dict(
        os.environ,
        PYTHONPATH=str(_HERE.parent.parent),
        PYTHONIOENCODING="utf-8",
        # N unpinned torch processes oversubscribe the cores and end up slower
        # than running the cells one at a time.
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
    )
    command = _command(args, optimizer, seed, cell)
    started = time.time()
    with open(cell / "log.txt", "w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started

    record = {
        "optimizer": optimizer,
        "seed": seed,
        "seconds": elapsed,
        "returncode": completed.returncode,
        "command": " ".join(command),
    }
    if completed.returncode == 0:
        record.update(_parse_log(cell / "log.txt"))
        (cell / "done.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    else:
        # No done.json is written, so the next run of this script retries the cell.
        record["error"] = _last_error_line(cell / "log.txt")
    return record


def _parse_log(path: pathlib.Path) -> dict:
    """Pull the per-seed line run_ablation prints:

    ``  <optimizer> seed   N: return=X.XXX   YY.Ys``
    """
    parsed = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "return=" in line and "seed" in line:
            try:
                parsed["return"] = float(line.split("return=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return parsed


def _last_error_line(path: pathlib.Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    for line in reversed(lines):
        if "Error" in line:
            return line[:300]
    return lines[-1][:300] if lines else ""


def _estimate_hours(args, optimizer: str) -> float:
    frames = args.iters * args.frames_per_batch
    grads = 2 if _two_gradient(optimizer) else 1
    epochs = args.epochs
    if args.half_epochs and grads == 2:
        epochs = max(1, epochs // 2)
    by_task = _HOURS_PER_MFRAME.get(args.device, _HOURS_PER_MFRAME["cpu"])
    rates = by_task.get(args.task, by_task[_UNMEASURED_TASK])
    # Cost tracks gradient steps per frame, not epochs: the reference measurement
    # did epochs * ceil(batch / minibatch) updates for every `batch` frames, and
    # so does this config, but with wildly different numbers.
    updates = epochs * -(-args.frames_per_batch // args.minibatch_size)
    per_frame = updates / args.frames_per_batch
    return rates[grads] * frames / 1e6 * (per_frame / _REFERENCE_UPDATES_PER_FRAME)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="vmas/simple_tag")
    parser.add_argument("--algorithm", default="ippo")
    parser.add_argument("--optimizers", nargs="+",
                        default=["adam", "adam_cosine", "sgd", "pc", "adaptive", "pcvi"],
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
    parser.add_argument("--eval-every", type=int, default=1, metavar="ROUNDS",
                        help="evaluate every ROUNDS collection rounds. run_ablation's "
                             "own default is iters // 4, which gives four points "
                             "however long the run is and cannot be plotted against "
                             "frames, so the campaign asks for something denser.")
    parser.add_argument("--device", default="cpu",
                        help="cpu is FASTER than cuda on this project's GPU "
                             "(Quadro P1000, tiny MLPs); measure before changing it")
    parser.add_argument("--half-epochs", action="store_true")
    parser.add_argument("--lambda0", nargs="*", default=[], metavar="NAME=VALUE")
    parser.add_argument("--optimizer-overrides", nargs="*", default=[],
                        metavar="FIELD=VALUE")
    parser.add_argument("--output-dir", default="outputs/campaign")
    parser.add_argument("--workers", type=int, default=1,
                        help="cells to run at once")
    parser.add_argument("--threads-per-worker", type=int, default=None,
                        help="CPU threads per cell. Default 1 on cuda (the "
                             "compute is on the GPU, so extra threads only "
                             "contend) and 3 on cpu.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the grid and the estimated cost, run nothing")
    args = parser.parse_args()
    if args.threads_per_worker is None:
        # On a GPU the work is on the device; one torch thread per cell is
        # enough to feed it, and N cells x 128 threads only thrash the box.
        args.threads_per_worker = 1 if args.device != "cpu" else 3

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        plus_minus = "±"
    except (AttributeError, ValueError):
        plus_minus = "+/-"

    cells = [(optimizer, seed)
             for optimizer in args.optimizers
             for seed in args.seeds]
    todo = [cell for cell in cells if not _is_done(_cell_dir(args, *cell))]
    done_already = len(cells) - len(todo)

    frames = args.iters * args.frames_per_batch
    print(f"{args.algorithm.upper()} on {args.task} | "
          f"{len(args.optimizers)} branches x {len(args.seeds)} seeds = {len(cells)} runs | "
          f"{args.iters} rounds x {args.frames_per_batch} frames = "
          f"{frames / 1e6:.2f}M frames each")
    if done_already:
        print(f"{done_already} already finished, resuming the remaining {len(todo)}")

    serial = sum(_estimate_hours(args, optimizer) for optimizer, _ in todo)
    by_task = _HOURS_PER_MFRAME.get(args.device, {})
    measured = args.task in by_task
    where = "RTX 3090" if args.device == "cuda" else "12-core CPU"
    print(f"\nEstimated cost (from a 2-round measurement on a {where}, not a guarantee"
          + ("):" if measured else f"; {args.task} was never measured, quoting "
                                   f"{_UNMEASURED_TASK} rates):"))
    if args.device not in _HOURS_PER_MFRAME:
        print(f"  !! --device {args.device} was never measured; quoting cpu rates.")
    updates = args.epochs * -(-args.frames_per_batch // args.minibatch_size)
    ratio = (updates / args.frames_per_batch) / _REFERENCE_UPDATES_PER_FRAME
    if ratio < 0.5 or ratio > 2.0:
        print(f"  !! This config does {ratio:.2f}x the gradient steps per frame of the "
              f"measured one\n"
              f"     ({updates} updates per {args.frames_per_batch} frames against 450 "
              f"per 120000). Only the\n"
              f"     gradient part is rescaled, not the environment simulation, so this "
              f"is a rough bound.")
    if args.device == "cuda" and args.n_envs != 1200:
        print(f"  !! The cuda rates were measured at --n-envs 1200; you asked for "
              f"{args.n_envs}.\n"
              f"     On the RTX 3090 the cost fell until 2400 and was flat after: "
              f"200 -> 0.12 h/Mframe,\n"
              f"     600 -> 0.07, 1200 -> 0.05, 2400+ -> 0.04 (adam).")
    for optimizer in args.optimizers:
        each = _estimate_hours(args, optimizer)
        remaining = sum(1 for name, _ in todo if name == optimizer)
        print(f"  {optimizer:<16} {each:6.1f} h/seed x {remaining} remaining "
              f"= {each * remaining:7.1f} h")
    workers = max(1, args.workers)
    print(f"  {'TOTAL':<16} {serial:6.1f} h serial")
    if workers > 1:
        # Perfect scaling is an upper bound that never happens: the cells contend
        # for cores and memory bandwidth. Reporting serial/workers alone would
        # promise a wall clock the machine cannot deliver.
        print(f"  {'':<16} {serial / workers:6.1f} h at {workers} workers "
              f"IF they scaled perfectly (they do not)")
        print(f"  {'':<16} {serial / _MEASURED_SPEEDUP:6.1f} h at the speedup actually "
              f"measured on this project's machine ({_MEASURED_SPEEDUP}x at 4 workers)")

    if args.dry_run:
        print("\n--dry-run: nothing was run.")
        return

    if not todo:
        print("\nNothing to do: every cell already has a done.json.")
        _summarise(args, cells, plus_minus)
        return

    print(f"\nStarting {args.workers} worker(s), {args.threads_per_worker} threads each. "
          f"Rerun this exact command to resume after an interrupt.\n", flush=True)

    work = queue.Queue()
    for cell in todo:
        work.put(cell)
    results = []
    lock = threading.Lock()
    started = time.time()

    def worker():
        while True:
            try:
                optimizer, seed = work.get_nowait()
            except queue.Empty:
                return
            record = _run_cell(args, optimizer, seed)
            with lock:
                results.append(record)
                if record["returncode"] == 0:
                    status = "OK"
                else:
                    status = "FAIL " + record.get("error", "")[:80]
                got = record.get("return")
                shown = "-" if got is None else f"{got:.3f}"
                print(f"[{len(results)}/{len(todo)}] {optimizer:<14} seed {seed} "
                      f"{record['seconds'] / 3600:5.2f} h  return={shown}  {status}",
                      flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [record for record in results if record["returncode"] != 0]
    print(f"\nCampaign finished in {(time.time() - started) / 3600:.2f} h. "
          f"{len(results) - len(failed)} ok, {len(failed)} failed.")
    for record in failed:
        cell = _cell_dir(args, record["optimizer"], record["seed"])
        print(f"  FAILED {record['optimizer']} seed {record['seed']}: "
              f"{record.get('error', '')}")
        print(f"         log: {cell / 'log.txt'}")
    _summarise(args, cells, plus_minus)


def _summarise(args, cells, plus_minus="+/-"):
    """Aggregate every done.json on disk into the comparison table."""
    print(f"\n{'optimizer':<20}{'return':>18}{'seeds':>8}{'h/seed':>9}{'grad/step':>11}")
    print("-" * 66)
    for optimizer in args.optimizers:
        records = []
        for _, seed in [cell for cell in cells if cell[0] == optimizer]:
            path = _cell_dir(args, optimizer, seed) / "done.json"
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
        returns = [record["return"] for record in records if "return" in record]
        grads = 2 if _two_gradient(optimizer) else 1
        if not returns:
            print(f"{optimizer:<20}{'(no finished run)':>18}{0:>8}{'-':>9}{grads:>11}")
            continue
        mean = statistics.fmean(returns)
        std = statistics.stdev(returns) if len(returns) > 1 else 0.0
        hours = statistics.fmean(record["seconds"] for record in records) / 3600
        print(f"{optimizer:<20}{mean:>9.3f} {plus_minus} {std:<6.3f}"
              f"{len(returns):>8}{hours:>9.2f}{grads:>11}")
    print("\nRead HUONG_DAN_CHAY.md section 5.2 before drawing conclusions: the "
          "two-gradient branches cost twice as much per update, so equal iterations "
          "is not equal compute (rerun with --half-epochs).")


if __name__ == "__main__":
    main()
