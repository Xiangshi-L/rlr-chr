#!/usr/bin/env python3
"""Table 18: the skew-reversal design, WSC, length and coverage on the two halves of x."""
import csv
import os
import sys
from collections import defaultdict

from common import closest, mean, num, parse_args, se, shortest
from table_bodies import write_body

SKEW_METHODS = ["CHR", "Norm-SC", "G2-Residual", "MCR", "RLR-CHR"]
N_TABLE = 3000   # sample size shown in the table
STATS = ("wsc", "length", "coverage", "coverage_xlt05", "coverage_xge05")


def main():
    args = parse_args(__doc__)
    reps = defaultdict(lambda: defaultdict(list))     # n -> method -> repetitions
    wsc_rlr = defaultdict(list)                       # n -> RLR-CHR WSC per repetition
    values = defaultdict(lambda: defaultdict(list))   # method -> column -> values at N_TABLE
    with open(os.path.join(args.results, "synthetic", "skew_long.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            method, n = r["method"], int(float(r["n_data"]))
            reps[n][method].append(int(float(r["rep"])))
            if method == "RLR-CHR":
                wsc_rlr[n].append(num(r["wsc"]))
            if n == N_TABLE:
                for col in STATS:
                    values[method][col].append(num(r[col]))
    same_reps = len({tuple(sorted(reps[N_TABLE][method])) for method in SKEW_METHODS}) == 1
    if sorted(values) != sorted(SKEW_METHODS) or not same_reps:
        sys.exit(f"n={N_TABLE}: expected the same repetitions for {', '.join(SKEW_METHODS)}")
    avg = {method: {col: mean(v) for col, v in values[method].items()} for method in SKEW_METHODS}
    covs = [avg[method]["coverage"] for method in SKEW_METHODS]
    print(f"marginal coverage at n={N_TABLE}: {min(covs):.4f} to {max(covs):.4f}")
    for n in sorted(wsc_rlr):
        print(f"RLR-CHR WSC at n={n}: {mean(wsc_rlr[n]):.4f}")
    for method in SKEW_METHODS:
        means = avg[method]
        print(f"{method}: WSC {means['wsc']:.4f}, length {means['length']:.3f}, "
              f"coverage x<0.5 {means['coverage_xlt05']:.4f}, "
              f"x>=0.5 {means['coverage_xge05']:.4f}")
    # Bold: the WSC closest to 0.9 and the shortest average length, ties at the printed precision.
    best_wsc = closest({method: avg[method]["wsc"] for method in SKEW_METHODS}, 0.9, nd=4)
    best_len = shortest({method: avg[method]["length"] for method in SKEW_METHODS}, nd=3)
    out = ["\\small", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
           "Method & WSC$_{0.1}$ & Average length & Coverage, $x<0.5$ & Coverage, $x\\ge0.5$ \\\\",
           "\\midrule"]
    for method in SKEW_METHODS:
        means = avg[method]
        wsc = f"{means['wsc']:.4f} ({se(values[method]['wsc']):.4f})"
        length = f"{means['length']:.3f} ({se(values[method]['length']):.3f})"
        if method in best_wsc:
            wsc = f"\\textbf{{{wsc}}}"
        if method in best_len:
            length = f"\\textbf{{{length}}}"
        out.append(f"{method} & {wsc} & {length} & {means['coverage_xlt05']:.4f} & "
                   f"{means['coverage_xge05']:.4f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    write_body(os.path.join(args.out, "skew_table_body.tex"), out, se="paren")


if __name__ == "__main__":
    main()
