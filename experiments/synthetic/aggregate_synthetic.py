#!/usr/bin/env python3
"""Mean and standard error of WSC and length over the 100 repetitions of the Figure 1 grid."""
import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIZES = [100, 200, 300, 500, 1000, 2000, 5000, 8000]
METHODS = ["CHR", "G2-Residual", "MCR", "Norm-SC", "RBC-CHR", "RLR-CHR"]
REPS = list(range(1, 101))


def se(v):
    """Standard error of the mean."""
    # Use sqrt(variance) for the summary's SD rounding convention.
    return math.sqrt(st.variance(v)) / math.sqrt(len(v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPO / "results"), help="results root")
    args = p.parse_args()
    root = Path(args.out) / "synthetic"

    vals = defaultdict(lambda: defaultdict(dict))   # n -> method -> repetition -> row
    for f in sorted((root / "grid").glob("n*_reps*-*.csv")):
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                n, method, rep = int(r["n_data"]), r["method"], int(r["rep"])
                if rep in vals[n][method]:
                    raise SystemExit(f"repetition {rep} for n={n} {method} appears twice "
                                     f"(second copy in {f.name})")
                vals[n][method][rep] = r

    rows, missing = [], []
    if sorted(vals) != SIZES:
        missing.append(f"sample sizes {sorted(vals)}, expected {SIZES}")
    for n in sorted(vals):
        if sorted(vals[n]) != METHODS:
            missing.append(f"n={n}: methods {sorted(vals[n])}, expected {METHODS}")
        for method in sorted(vals[n]):
            reps = sorted(vals[n][method])
            if reps != REPS:
                missing.append(
                    f"n={n} {method}: {len(reps)} repetitions, range {reps[0]}..{reps[-1]}"
                )
                continue
            wsc = [float(vals[n][method][rep]["wsc"]) for rep in reps]
            length = [float(vals[n][method][rep]["length"]) for rep in reps]
            coverage = [float(vals[n][method][rep]["coverage"]) for rep in reps]
            rows.append({"n_data": n, "method": method, "reps": len(reps),
                         "wsc_mean": st.mean(wsc), "wsc_se": se(wsc),
                         "len_mean": st.mean(length), "len_se": se(length),
                         "cov_mean": st.mean(coverage)})
    if missing:
        raise SystemExit("the grid is incomplete (nothing written):\n  " + "\n  ".join(missing))
    out = root / "fig1_summary.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
