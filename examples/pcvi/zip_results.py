#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Package the contents of ``outputs/`` into one zip, with a summary inside.

Every run writes one folder under ``outputs/``, and every metric inside it is a
separate two-column ``step,value`` csv. That is fine to train against and awful
to hand to anyone. This script rolls the whole thing into a single archive and
adds two files that make it readable without unpacking every csv:

``manifest.csv``
    one row per run: folder, path, algorithm, optimizer, task, model, number of
    metrics, size, modification time. The optimizer column is why
    ``experiment.py`` puts the optimizer in the run name -- without it, an
    IPPO+Adam run and an IPPO+PCVI run are indistinguishable on disk. The
    ``path`` column is what carries the **seed**, since it holds the
    ``<optimizer>_seed<N>/`` cell folder that ``run_campaign.py`` writes; the
    run name itself has no seed in it.
``summary.csv``
    one row per (run, metric) with the **first, last, min, max** value and the
    number of points. This is the table to open first: the last value of
    ``train_<group>_pcvi_lambda_loss_objective`` next to ``pcvi_lambda_0``
    immediately says whether the adaptive step ever fired.

Checkpoints are excluded by default -- they dominate the size and are not
results. Pass ``--with-checkpoints`` if you need them.

Usage::

    python examples/pcvi/zip_results.py
    python examples/pcvi/zip_results.py --output-dir outputs --dest results.zip
    python examples/pcvi/zip_results.py --only pcvi pc      # runs matching these
    python examples/pcvi/zip_results.py --with-checkpoints
"""

import argparse
import csv
import datetime
import io
import pathlib
import sys
import zipfile

SCALAR_DIR = "scalars"
CHECKPOINT_DIR = "checkpoints"


def run_folders(output_dir: pathlib.Path, only):
    """The per-run folders under ``output_dir``, optionally filtered by name.

    A run folder is the one holding ``config.pkl``, found at whatever depth it
    sits. Looking only one level down would break on the two layouts that are
    not flat:

    * ``run_campaign.py`` puts each cell in ``<optimizer>_seed<N>/<run name>/``,
    * hydra puts each run in ``outputs/<date>/<time>/<run name>/``,

    and in both cases the one-level-down folder has no name to parse, so the
    manifest would lose the optimizer column -- the column the run naming exists
    for in the first place.
    """
    folders = sorted({p.parent for p in output_dir.rglob("config.pkl")})
    if not folders:  # a run whose logger wrote nothing but scalars
        folders = sorted(
            {p.parent.parent for p in output_dir.rglob(f"{SCALAR_DIR}/*.csv")}
        )
    if not folders:  # nothing recognisable: fall back to the flat layout
        folders = sorted(p for p in output_dir.iterdir() if p.is_dir())
    if only:
        folders = [p for p in folders if any(token in p.name for token in only)]
    return folders


def parse_name(name: str):
    """``ippo_pcvi_simple_tag_mlp__<hash>_<date>`` -> its parts.

    The name is built by ``Experiment._setup_name`` as
    ``<algorithm>_<optimizer>_<task>_<model>__<hash>_<date>``, and the task name
    itself may contain underscores, so the model is taken from the end.
    """
    head, _, tail = name.partition("__")
    parts = head.split("_")
    if len(parts) < 3:
        return {"algorithm": head, "optimizer": "", "task": "", "model": ""}
    return {
        "algorithm": parts[0],
        "optimizer": parts[1],
        "task": "_".join(parts[2:-1]),
        "model": parts[-1],
        "run_id": tail,
    }


def scalar_files(folder: pathlib.Path):
    """The metric csvs of a run, wherever the logger nested them."""
    return sorted(folder.glob(f"**/{SCALAR_DIR}/*.csv"))


def read_series(path: pathlib.Path):
    """A ``step,value`` csv with no header -> the list of values."""
    values = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except ValueError:
                continue  # a header line, if some logger ever writes one
    return values


def build_summary(folders, output_dir):
    rows = []
    for folder in folders:
        parsed = parse_name(folder.name)
        for csv_path in scalar_files(folder):
            values = read_series(csv_path)
            if not values:
                continue
            rows.append(
                {
                    "run": folder.name,
                    "path": str(folder.relative_to(output_dir)),
                    "algorithm": parsed["algorithm"],
                    "optimizer": parsed["optimizer"],
                    "task": parsed["task"],
                    "metric": csv_path.stem,
                    "n_points": len(values),
                    "first": values[0],
                    "last": values[-1],
                    "min": min(values),
                    "max": max(values),
                }
            )
    return rows


def build_manifest(folders, output_dir):
    rows = []
    for folder in folders:
        parsed = parse_name(folder.name)
        files = list(folder.rglob("*"))
        size = sum(f.stat().st_size for f in files if f.is_file())
        rows.append(
            {
                "run": folder.name,
                "path": str(folder.relative_to(output_dir)),
                "algorithm": parsed["algorithm"],
                "optimizer": parsed["optimizer"],
                "task": parsed["task"],
                "model": parsed["model"],
                "n_metrics": len(scalar_files(folder)),
                "size_mb": round(size / 1e6, 3),
                "modified": datetime.datetime.fromtimestamp(
                    folder.stat().st_mtime
                ).isoformat(timespec="seconds"),
            }
        )
    return rows


def to_csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs",
                        help="folder holding the per-run folders")
    parser.add_argument("--dest", default=None,
                        help="archive path (default results_<timestamp>.zip)")
    parser.add_argument("--only", nargs="*", default=[],
                        metavar="TOKEN",
                        help="keep only runs whose folder name contains a token, "
                             "e.g. --only pcvi pc")
    parser.add_argument("--with-checkpoints", action="store_true",
                        help="include checkpoints; they dominate the size")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        raise SystemExit(
            f"{output_dir} does not exist. Run an experiment first, or pass "
            f"--output-dir."
        )
    folders = run_folders(output_dir, args.only)
    if not folders:
        raise SystemExit(f"No run folders in {output_dir} matching {args.only or 'any'}.")

    dest = pathlib.Path(
        args.dest
        or f"results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    ).resolve()
    if dest.is_relative_to(output_dir):
        raise SystemExit(
            f"The archive would be written inside the folder being archived "
            f"({dest}). Pass a --dest outside {output_dir}."
        )

    manifest = build_manifest(folders, output_dir)
    summary = build_summary(folders, output_dir)

    written = skipped = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                if not args.with_checkpoints and CHECKPOINT_DIR in path.parts:
                    skipped += 1
                    continue
                archive.write(path, path.relative_to(output_dir))
                written += 1
        archive.writestr("manifest.csv", to_csv_bytes(manifest))
        if summary:
            archive.writestr("summary.csv", to_csv_bytes(summary))

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print(f"{dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(folders)} run(s), {written} file(s)"
          + (f", {skipped} checkpoint file(s) skipped" if skipped else ""))
    print(f"  + manifest.csv, summary.csv ({len(summary)} metric rows)")
    print()
    print(f"{'optimizer':<22}{'algorithm':<12}{'task':<18}{'metrics':>8}{'MB':>8}")
    print("-" * 68)
    for row in manifest:
        print(
            f"{row['optimizer']:<22}{row['algorithm']:<12}{row['task']:<18}"
            f"{row['n_metrics']:>8}{row['size_mb']:>8.2f}"
        )
    if not args.with_checkpoints and skipped:
        print("\nCheckpoints were left out; pass --with-checkpoints to include them.")


if __name__ == "__main__":
    main()
