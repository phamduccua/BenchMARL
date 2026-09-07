#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Rank optimizers by how well they actually approach the Nash equilibrium.

The last logged ``nash_conv`` does not answer that question on its own. In a
two-player zero-sum game the gradient field is purely rotational, so the iterate
*orbits* the equilibrium; where the orbit happens to be when the run stops is
close to arbitrary. Three runs that end at the same ``nash_conv`` can be
converging, cycling at a fixed radius, or spiralling outward -- and only the
last one is a failure. Simulated on this project's own payoff matrix:

    kich ban      nc cuoi   nc t.binh   do doc log   bien do
    hoi tu          0.034       0.160      -0.0976     0.464
    quay vong       0.177       0.164      +0.0078     0.270
    xoay ra         0.745       0.254      +0.1014     0.423

``nc cuoi`` puts them in an order that has nothing to do with which is better.
The slope separates them cleanly. So this script reports four numbers per run:

``nc_last``
    exploitability of the final policy. What a naive reading uses.
``nc_avg``
    exploitability of the **time-averaged policy**. This is the classical
    quantity: for gradient play on a zero-sum game the *average* iterate
    converges to Nash even while the last iterate cycles forever, so a method
    can be doing everything right and still show a large ``nc_last``. Computed
    here by averaging the logged action distributions over evaluations and
    evaluating exploitability once, which is NOT the same as averaging
    ``nash_conv`` (that would be the mean of a convex function, always larger).
``spiral``
    slope of ``log nash_conv`` over the second half of the run, per evaluation.
    **Negative = pulling in, ~0 = orbiting, positive = spiralling out.** This is
    the number that answers "does this algorithm damp the rotation".
``wobble``
    std/mean of ``nash_conv`` over the second half: the orbit's radius relative
    to its mean. Large with a flat slope means a wide stable cycle.

Read them together: an algorithm is better at reaching Nash when ``spiral`` is
more negative, and among methods with the same slope, when ``nc_avg`` is lower.
``nc_last`` is a tiebreaker, not the headline.

Usage::

    python examples/pcvi/analyse_nash.py --output-dir outputs/matrix_rps
    python examples/pcvi/analyse_nash.py --output-dir outputs/matrix_rps --csv rank.csv
"""

import argparse
import collections
import csv
import math
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarl.environments.matrixgame.matrix_game import (  # noqa: E402
    ACTION_NAMES,
    PAYOFFS,
    PLAYERS,
)

PROB_RE = re.compile(r"eval_action_prob_(player_\d+)_([a-z]+)\.csv$")


def read_series(path):
    """A ``step,value`` csv -> {step: value}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                out[int(float(row[0]))] = float(row[1])
            except ValueError:
                continue
    return out


def nash_conv(payoff, x, y):
    """``max_i (A y)_i - min_j (x A)_j`` -- 0 iff (x, y) is a Nash equilibrium.

    Player 0 maximises ``x A y``, player 1 minimises it. Verified against the
    two corners: uniform/uniform gives 0, pure/pure gives the maximum (2.0 on
    rock-paper-scissors).
    """
    ay = [sum(payoff[i][j] * y[j] for j in range(len(y))) for i in range(len(x))]
    xa = [sum(x[i] * payoff[i][j] for i in range(len(x))) for j in range(len(y))]
    return max(ay) - min(xa)


def policies_over_time(run_dir, game):
    """-> steps, {player: [distribution at each step]}, or None if not logged."""
    found = collections.defaultdict(dict)
    for path in run_dir.rglob("eval_action_prob_*.csv"):
        match = PROB_RE.search(path.name)
        if match:
            found[match.group(1)][match.group(2)] = read_series(path)
    if set(found) != set(PLAYERS):
        return None, None
    names = ACTION_NAMES[game]
    steps = sorted(
        set.intersection(
            *[set(found[p][a]) for p in PLAYERS for a in names if a in found[p]]
        )
    )
    if not steps:
        return None, None
    series = {}
    for player in PLAYERS:
        if set(found[player]) != set(names):
            return None, None
        series[player] = [[found[player][a][s] for a in names] for s in steps]
    return steps, series


def analyse(run_dir, game, tail_frac=0.5):
    steps, series = policies_over_time(run_dir, game)
    if steps is None or len(steps) < 4:
        return None
    payoff = PAYOFFS[game]
    p0, p1 = series[PLAYERS[0]], series[PLAYERS[1]]
    curve = [nash_conv(payoff, x, y) for x, y in zip(p0, p1)]

    # time-averaged POLICY, then exploitability once -- not the mean of the curve
    n = len(steps)
    avg0 = [sum(v[i] for v in p0) / n for i in range(len(p0[0]))]
    avg1 = [sum(v[i] for v in p1) / n for i in range(len(p1[0]))]

    tail = curve[int(n * (1 - tail_frac)) :]
    logs = [math.log(max(v, 1e-12)) for v in tail]
    mean_x = (len(logs) - 1) / 2
    mean_y = statistics.mean(logs)
    denom = sum((i - mean_x) ** 2 for i in range(len(logs)))
    slope = (
        sum((i - mean_x) * (y - mean_y) for i, y in enumerate(logs)) / denom
        if denom
        else 0.0
    )
    tail_mean = statistics.mean(tail)

    # policy movement between consecutive evaluations, averaged over the tail.
    # This is what separates "stable because it converged" from "stable because
    # it stopped": a branch whose lambda has collapsed sits still and would
    # otherwise look like the best-damped one.
    def l1(a, b):
        return sum(abs(p - q) for p, q in zip(a, b))

    start = int(n * (1 - tail_frac))
    moves = [
        0.5 * (l1(p0[i], p0[i - 1]) + l1(p1[i], p1[i - 1]))
        for i in range(max(start, 1), n)
    ]

    return {
        "nc_last": curve[-1],
        "nc_avg": nash_conv(payoff, avg0, avg1),
        "spiral": slope,
        "wobble": (statistics.pstdev(tail) / tail_mean) if tail_mean > 0 else 0.0,
        "move": statistics.mean(moves) if moves else 0.0,
        "n_evals": n,
    }


def optimizer_of(run_dir):
    """``ippo_pcvi_rock_paper_scissors_mlp__hash_date`` -> ``pcvi``."""
    for part in (run_dir.name, *(p.name for p in run_dir.parents)):
        bits = part.split("_")
        if len(bits) > 2 and bits[0] in ("ippo", "mappo"):
            return bits[1]
        if part.endswith(tuple(f"seed{i}" for i in range(10))):
            return part.rsplit("_seed", 1)[0]
    return run_dir.name


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True,
                        help="campaign folder, e.g. outputs/matrix_rps")
    parser.add_argument("--game", default="rock_paper_scissors",
                        choices=sorted(PAYOFFS))
    parser.add_argument("--tail-frac", type=float, default=0.5,
                        help="fraction of the run used for spiral/wobble")
    parser.add_argument("--csv", default=None, help="also write the table here")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    root = pathlib.Path(args.output_dir)
    if not root.is_dir():
        raise SystemExit(f"{root} khong ton tai.")

    runs = sorted({p.parent for p in root.rglob("eval_action_prob_*.csv")})
    if not runs:
        raise SystemExit(
            f"Khong tim thay eval_action_prob_*.csv trong {root}.\n"
            f"Chi task matrixgame/* moi co, va can NashDistanceCallback."
        )

    by_opt = collections.defaultdict(list)
    for scalars_dir in runs:
        result = analyse(scalars_dir.parent, args.game, args.tail_frac)
        if result:
            by_opt[optimizer_of(scalars_dir.parent)].append(result)

    if not by_opt:
        raise SystemExit("Khong doc duoc run nao (can >= 4 diem eval moi run).")

    rows = []
    for name, results in by_opt.items():
        row = {"optimizer": name, "n_seeds": len(results)}
        for key in ("nc_last", "nc_avg", "spiral", "wobble", "move"):
            values = [r[key] for r in results]
            row[key] = statistics.mean(values)
            row[key + "_sd"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    # best = pulls in hardest; ties broken by the averaged-policy exploitability
    rows.sort(key=lambda r: (r["spiral"], r["nc_avg"]))

    print(f"{args.game} | {len(runs)} run | doc tu action_prob, "
          f"duoi {args.tail_frac:.0%} chuoi cho spiral/wobble\n")
    print(f"{'optimizer':<16}{'seed':>5}{'nc_last':>16}{'nc_avg':>16}"
          f"{'spiral':>18}{'wobble':>9}{'move':>9}  ket luan")
    print("-" * 114)
    for row in rows:
        if row["move"] < 1e-3:
            # sitting still is not damping: check pcvi_lambda before reading
            # anything into a flat curve
            verdict = "DUNG IM (kiem pcvi_lambda!)"
        elif row["spiral"] < -0.005:
            verdict = "KEO VAO  <-- tot"
        elif row["spiral"] > 0.005:
            verdict = "XOAY RA  <-- xau"
        else:
            verdict = "quay vong (khong hoi tu)"
        print(
            f"{row['optimizer']:<16}{row['n_seeds']:>5}"
            f"{row['nc_last']:>10.3f}±{row['nc_last_sd']:<5.3f}"
            f"{row['nc_avg']:>10.3f}±{row['nc_avg_sd']:<5.3f}"
            f"{row['spiral']:>+12.4f}±{row['spiral_sd']:<5.4f}"
            f"{row['wobble']:>9.2f}{row['move']:>9.4f}  {verdict}"
        )

    print()
    print("Doc bang: `spiral` = do doc cua log(nash_conv) tren nua cuoi.")
    print("  am  = bien do vong xoay GIAM      -> thuat toan dang dap tat vong xoay")
    print("  ~0  = quay vong ban kinh co dinh  -> khong hoi tu, cung khong te di")
    print("  duong = xoay RA                   -> ngay cang xa Nash")
    print("`nc_avg` la exploitability cua CHINH SACH TRUNG BINH theo thoi gian:")
    print("  voi tro choi tong-bang-khong, iterate trung binh hoi tu ve Nash du")
    print("  iterate cuoi cu quay vong mai. nc_avg thap + spiral ~0 van la ket qua tot.")
    print("`nc_last` la so de so sanh sau cung, khong phai so dan dau.")
    print("`move` = |pi_t - pi_(t-1)|_1 trung binh: chinh sach con DI CHUYEN bao nhieu.")
    print("  ~0 nghia la nhanh do DUNG IM. Duong cong phang khi do KHONG phai la")
    print("  dap tat vong xoay -- kiem `pcvi_lambda`: lambda sup thi no dong bang.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDa ghi {args.csv}")


if __name__ == "__main__":
    main()
