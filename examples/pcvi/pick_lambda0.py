#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Read a ``sweep_lambda0.py`` table and pick ``lambda_0`` for every branch.

The sweep measures where ``lambda_k`` settles as a function of ``lambda_0``.
Turning that table into a number is a judgement call that is easy to get wrong
in a way nothing later complains about, so this script makes the rule explicit
and applies it the same way every time.

**The rule.** A row is *usable* when the run did not diverge, ``lambda``
actually moved (``lambda_end / lambda_0 < 1 - tol``), and ``beta_k`` went
negative on no more than ``--max-beta-neg`` percent of the steps.

Among the usable rows the default choice is the **smallest** ``lambda_0``, i.e.
the one sitting just above the knee. Note that "the largest ``lambda_0`` at
which lambda still shrinks" -- the phrasing this project used at first -- is not
a criterion at all: lambda shrinks for *every* ``lambda_0`` above the knee, and
shrinks harder the larger it is, so that rule just returns the largest value you
happened to try. The knee is the point where shrinkage *stops*, so the value
that identifies it is the smallest one that still moves.

Why just above the knee rather than far above it: Step 3 pulls ``lambda`` down
to ``~p/L_local`` on its own, so every usable ``lambda_0`` ends at the same
place and they differ only in the transient. The first steps are taken at
``lambda_0`` before Step 3 has reacted, and a large one makes them violent --
measured on this project, ``beta_k`` went negative on ~1.3% of steps at
``lambda_0 >= 0.1`` and never at ``lambda_0 <= 0.01``.

``--aggressive`` picks the largest usable row instead. That is the right choice
when the point is to *demonstrate* Step 3 doing work: the gap between
``lambda_0`` and ``lambda_end`` is the mechanism made visible. Both are
defensible; the script prints the other one either way so the choice is
deliberate.

The knee itself is reported as the median ``lambda_end`` over the usable rows.
It is only approximately a constant: ``lambda`` measures a *directional*
Lipschitz ratio along the trajectory, and a larger ``lambda_0`` walks through
higher-curvature regions, so the settling point drifts a little with
``lambda_0``.

**Derivation for the branches not swept.** ``p/L_local`` is a property of the
task, the trajectory and the *metric*, not of the branch, so one representative
per metric is enough:

* ``pcvi`` measures the knee in the Euclidean metric -> also fixes ``pc``,
  because Step 3 is the only difference and, below the knee, ``pc`` and ``pcvi``
  are the same algorithm bit for bit. ``pc`` freezes ``lambda``, so it gets
  **half the knee**: the convergence proof needs ``lambda <= p/L``, and ``p/L``
  is the boundary, not a safe value.
* ``pcvi_plus`` measures the knee under ``precond=true``, i.e. in Adam's metric
  -> also fixes ``pc_plus``, the same way.
* ``adaptive``/``adaptive_plus`` keep Adam and have ``lr_scale`` instead; use
  ``measure_lr_scale.py``. Their ``lambda_0`` follows ``pcvi``'s knee.

A derived value is marked ``suy ra`` in the output and is an assumption, not a
measurement: it holds while the two branches see the same local curvature, which
is true at the start (they begin from the same weights) and drifts later. If a
derived branch shows a flat ``pcvi_lambda`` in the real campaign, sweep it
directly.

Usage::

    python examples/pcvi/pick_lambda0.py --results outputs/sweep/sweep_results.csv
    python examples/pcvi/pick_lambda0.py --results outputs/*/sweep_results.csv
    python examples/pcvi/pick_lambda0.py --results R.csv --no-derive
"""

import argparse
import csv
import glob
import math
import pathlib
import statistics
import sys

# branch -> (representative that measures its knee, factor applied to the pick)
# 1.0 keeps the representative's own choice; 0.5 halves it for the fixed-lambda
# branches, whose proof needs lambda <= p/L rather than lambda ~ p/L.
DERIVED = {
    "pc": ("pcvi", 0.5),
    "pc_plus": ("pcvi_plus", 0.5),
    "adaptive": ("pcvi", 1.0),
    "adaptive_plus": ("pcvi", 1.0),
}

MOVED_TOL = 0.02  # lambda_end/lambda_0 must be below 1 - this to count as moved


def beta_neg(row):
    """How often beta_k went negative, BEFORE beta_fallback masked it.

    A branch with ``beta_fallback`` set never reports a negative ``pcvi_beta_k``
    -- the fallback has already replaced it -- so reading the plain column would
    say 0% on exactly the branches built to hide the problem. Fall back to the
    masked column only for older sweep tables that lack the raw one.
    """
    raw = row.get("beta_raw_negative_pct", float("nan"))
    if raw == raw:
        return raw
    return row.get("beta_negative_pct", float("nan"))


def load(patterns):
    rows, files = [], []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and pathlib.Path(pattern).exists():
            matches = [pattern]
        if not matches:
            raise SystemExit(f"No file matches {pattern!r}.")
        files += matches
    for path in sorted(set(files)):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key in ("lambda_0", "lambda_end", "diverged",
                            "beta_negative_pct", "beta_raw_negative_pct",
                            "mean_return"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                row["_file"] = path
                rows.append(row)
    if not rows:
        raise SystemExit("The sweep table(s) contained no rows.")
    return rows, sorted(set(files))


def usable(row, max_beta_neg):
    """Did the mechanism actually fire on this row, without misbehaving?"""
    if row.get("diverged", 0.0) > 0.0:
        return False, "phan ky"
    ratio = row["lambda_end"] / row["lambda_0"]
    if not math.isfinite(ratio):
        return False, "khong huu han"
    if ratio > 1.0 - MOVED_TOL:
        return False, f"lambda dung yen (end/0={ratio:.3f})"
    neg = beta_neg(row)
    if neg == neg and neg > max_beta_neg:
        return False, f"beta_k am {neg:.1f}% > {max_beta_neg:g}%"
    return True, f"end/0={ratio:.3f}"


def choose(rows, max_beta_neg, aggressive):
    """The usable row just above the knee -- or the largest, if aggressive."""
    good = [r for r in rows if usable(r, max_beta_neg)[0]]
    if not good:
        return None
    safe = min(good, key=lambda r: r["lambda_0"])
    bold = max(good, key=lambda r: r["lambda_0"])
    pick = bold if aggressive else safe
    other = safe if aggressive else bold
    knee = statistics.median(r["lambda_end"] for r in good)
    return {
        "row": pick,
        "other": other,
        "knee": knee,
        "n_usable": len(good),
        "n_total": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="CSV",
                        help="sweep_results.csv file(s); globs allowed")
    parser.add_argument("--no-derive", action="store_true",
                        help="report only branches actually swept")
    parser.add_argument("--task", default=None,
                        help="keep only rows of this task")
    parser.add_argument("--aggressive", action="store_true",
                        help="pick the LARGEST usable lambda_0 instead of the "
                             "one just above the knee; shows Step 3 doing more "
                             "work, at the cost of a violent transient")
    parser.add_argument("--max-beta-neg", type=float, default=5.0, metavar="PCT",
                        help="reject a row whose beta_k went negative on more "
                             "than PCT%% of steps (default 5). beta_k < 0 means "
                             "the step moves AGAINST d_k")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    rows, files = load(args.results)
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]
        if not rows:
            raise SystemExit(f"No rows for task {args.task!r}.")

    tasks = sorted({r.get("task", "?") for r in rows})
    print(f"Doc {len(files)} file, {len(rows)} dong, task: {', '.join(tasks)}")
    if len(tasks) > 1:
        print("  !! Nhieu task trong mot bang. lambda_0 KHONG dung chung duoc "
              "giua cac task;\n     chay lai voi --task de tach ra.")
    print()

    by_optimizer = {}
    for row in rows:
        by_optimizer.setdefault(row["optimizer"], []).append(row)

    mode = "LON NHAT dung duoc (--aggressive)" if args.aggressive else \
           "nho nhat dung duoc, tuc ngay tren dau goi"
    print(f"Quy tac chon: {mode}; loai dong co beta_k am > {args.max_beta_neg:g}%")
    print()
    print(f"{'nhanh':<16}{'lambda_0 chon':>15}{'dau goi p/L':>14}"
          f"{'dung duoc':>11}{'beta<0':>9}  lua chon con lai / ghi chu")
    print("-" * 96)

    picks, knees = {}, {}
    for name in sorted(by_optimizer):
        result = choose(by_optimizer[name], args.max_beta_neg, args.aggressive)
        if result is None:
            reasons = {usable(r, args.max_beta_neg)[1].split("(")[0].strip()
                       for r in by_optimizer[name]}
            print(f"{name:<16}{'KHONG CO':>15}{'-':>14}"
                  f"{'0/' + str(len(by_optimizer[name])):>11}{'-':>9}  "
                  f"!! {', '.join(sorted(reasons))}")
            continue
        row, knee = result["row"], result["knee"]
        picks[name] = row["lambda_0"]
        knees[name] = knee
        neg = beta_neg(row)
        neg_txt = "-" if neg != neg else f"{neg:.1f}%"
        other = result["other"]["lambda_0"]
        note = "" if other == row["lambda_0"] else f"{other:g}"
        if result["n_usable"] == 1:
            note += "  (chi 1 dong dung duoc -> quet them diem)"
        print(f"{name:<16}{row['lambda_0']:>15.3g}{knee:>14.3g}"
              f"{str(result['n_usable']) + '/' + str(result['n_total']):>11}"
              f"{neg_txt:>9}  {note}")

    if not args.no_derive:
        derived = {}
        for name, (source, factor) in DERIVED.items():
            if name in picks or source not in picks:
                continue
            derived[name] = picks[source] * factor
            print(f"{name:<16}{derived[name]:>15.3g}{knees[source]:>14.3g}"
                  f"{'suy ra':>11}{'-':>9}  tu `{source}` x {factor}")
        picks.update(derived)

    if not picks:
        raise SystemExit(
            "\nKhong nhanh nao co lambda_0 dung duoc. Moi dong deu phan ky hoac "
            "lambda dung yen:\nquet lai voi dai GIA TRI CAO HON (lambda dung yen) "
            "hoac THAP HON (phan ky)."
        )

    print()
    print("Dan thang vao run_campaign.py:")
    print()
    fragment = " ".join(f"{k}={v:g}" for k, v in sorted(picks.items()))
    print(f"    --lambda0 {fragment}")
    print()
    print("Kiem tra sau khi chay chien dich: cot `pcvi_lambda` trong csv PHAI co "
          "lai.\nPhang li dung bang lambda_0 = lua chon nay sai, quet lai dai cao hon.")


if __name__ == "__main__":
    main()
