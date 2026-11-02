#!/usr/bin/env python3
"""Table 13: coverage of RLR-CHR and CHR in covariate-density strata, D1 (sparsest) to Dn."""
import csv
import math
import os
import sys

from common import DAGGER, MAIN_DATASETS, is_degenerate, mean, mean_with_se, num, parse_args
from table_bodies import write_body

DENSITY_METHODS = ["RLR-CHR", "CHR"]


def load(results, key):
    """Coverage by (method, rep, stratum), number of strata, repetitions, degenerate repetitions."""
    cov = {}
    with open(os.path.join(results, "density", f"{key}_strata.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            cell = (r["method"], int(float(r["rep"])), r["stratum"])
            if cell in cov:
                sys.exit(f"{key}: more than one strata row for {cell}")
            cov[cell] = num(r["coverage"])
    engine = {}
    with open(os.path.join(results, "density", f"{key}_engine_long.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            engine[(r["method"], int(float(r["rep"])))] = r
    n_strata = max(int(s) for _, _, s in cov if s != "all")
    reps = sorted({rep for method, rep, _ in cov if method == "CHR"})
    if sorted({rep for method, rep, _ in cov if method == "RLR-CHR"}) != reps:
        sys.exit(f"{key}: RLR-CHR and CHR do not share the same repetitions")
    chr_len = {rep: num(engine[("CHR", rep)]["length"]) for rep in reps}
    if not all(math.isfinite(v) and v > 0 for v in chr_len.values()):
        sys.exit(f"{key}: a CHR length is not finite and positive")
    # Degenerate against the CHR length of this run in the same repetition.
    bad = {}
    for method in DENSITY_METHODS:
        bad[method] = [r for r in reps
                       if is_degenerate(method, num(engine[(method, r)]["length"]), chr_len[r])]
        if len(reps) - len(bad[method]) < 2:
            sys.exit(f"{key}: {method} has fewer than two non-degenerate repetitions")
    return {"cov": cov, "n_strata": n_strata, "reps": reps, "bad": bad}


def stratum_cov(d, method, s, all_reps=False):
    """Coverage of a method in stratum s per repetition (non-degenerate ones unless all_reps)."""
    return [d["cov"][(method, r, str(s))] for r in d["reps"]
            if all_reps or r not in d["bad"][method]]


def table(data):
    """Columns: the datasets; rows: stratum x method, a rule between strata."""
    n = data[0][1]["n_strata"]
    out = ["\\scriptsize", "\\setlength{\\tabcolsep}{5pt}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{@{}ll" + "r" * len(data) + "@{}}", "\\toprule",
           "\\textbf{Stratum} & \\textbf{Method}"
           + "".join(f" & \\textbf{{{name}}}" for name, _ in data) + " \\\\", "\\midrule"]
    # One pair of rows per stratum, RLR-CHR above CHR.
    for s in range(1, n + 1):
        if s > 1:
            out.append("\\midrule")
        for i, method in enumerate(DENSITY_METHODS):
            stratum = (f"D{s}" if i == 0 else
                       "(sparsest)" if s == 1 else "(densest)" if s == n else "")
            cells = [mean_with_se(stratum_cov(d, method, s)) + (DAGGER if d["bad"][method] else "")
                     for _, d in data]
            out.append(f"{stratum} & {method} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}%", "}"]
    return out


def main():
    args = parse_args(__doc__)
    data = [(name, load(args.results, key)) for key, name in MAIN_DATASETS]
    if len({d["n_strata"] for _, d in data}) != 1:
        sys.exit("the datasets were run with different numbers of strata")
    # Coverage over all repetitions of the cells with degenerate repetitions.
    for name, d in data:
        for method in DENSITY_METHODS:
            if d["bad"][method]:
                cov_all = [mean(stratum_cov(d, method, s, all_reps=True))
                           for s in range(1, d["n_strata"] + 1)]
                print(f"{name}, {method}: {len(d['bad'][method])} of {len(d['reps'])} repetitions "
                      f"degenerate {d['bad'][method]}; coverage over all repetitions, D1 to "
                      f"D{d['n_strata']}: " + ", ".join(f"{v:.3f}" for v in cov_all))
    write_body(os.path.join(args.out, "density_strata_table_body.tex"), table(data), se="paren")


if __name__ == "__main__":
    main()
