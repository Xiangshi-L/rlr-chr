#!/usr/bin/env python3
"""Tables 14-16: sensitivity to the score levels T, the histogram bins and alpha = 0.05."""
import csv
import math
import os
import sys
from collections import defaultdict

from common import (ALPHA_DATASETS, DAGGER, is_degenerate, mean, mean_with_se, median, num,
                    parse_args, se)
from table_bodies import write_body

# Result-file column and slab mass delta of each WSC.
WSC_DELTA = (("wsc_005", "0.05"), ("wsc", "0.10"), ("wsc_020", "0.20"))


def read_long(path):
    """Rows of a result file."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def bins_of(row):
    """Number of histogram bins of a row; None for CQR, which has no histogram."""
    bins = num(row["bins"])
    return None if math.isnan(bins) else int(bins)


def column(cell, col, reps):
    """Values of a result-file column over the repetitions reps of a cell {rep: row}."""
    return [num(cell[r][col]) for r in reps]


def cells_by(rows, cell_of, where):
    """{cell: {rep: row}} with one row per (cell, rep) and the same repetitions in every cell."""
    cells = defaultdict(dict)
    for r in rows:
        cell, rep = cell_of(r), int(float(r["rep"]))
        if rep in cells[cell]:
            sys.exit(f"{where}: more than one row for cell {cell} rep {rep}")
        cells[cell][rep] = r
    if len({tuple(sorted(v)) for v in cells.values()}) != 1:
        sys.exit(f"{where}: the cells do not share the same repetitions")
    return cells


def split_degenerate(cell, ref_len, method, where):
    """(kept, degenerate) repetitions of a cell {rep: row}, against ref_len {rep: CHR length}."""
    reps = sorted(cell)
    # CQR's share of empty intervals, in the files that carry the column frac_empty_cqr.
    bad = [r for r in reps if is_degenerate(method, num(cell[r]["length"]), ref_len[r],
                                            num(cell[r].get("frac_empty_cqr")))]
    keep = [r for r in reps if r not in bad]
    if len(keep) < 2:
        sys.exit(f"{where}: fewer than two non-degenerate repetitions")
    return keep, bad


def level_cells(cell, ref_len, method, where):
    """Coverage, WSC and length with SEs over the non-degenerate repetitions of a cell."""
    keep, bad = split_degenerate(cell, ref_len, method, where)
    if bad:
        reps = sorted(cell)
        length = column(cell, "length", reps)
        print(f"{where}: {len(bad)} of {len(reps)} repetitions degenerate {bad}; "
              f"all repetitions: coverage {mean(column(cell, 'coverage', reps)):.4f}, "
              f"WSC {mean(column(cell, 'wsc', reps)):.4f}, "
              f"average length {mean(length):.2f} (SE {se(length):.2f})")
    dg = DAGGER if bad else ""
    return [f"{mean(v):.{nd}f} ({se(v):.{nd}f}){dg}"
            for v, nd in ((column(cell, "coverage", keep), 4), (column(cell, "wsc", keep), 4),
                          (column(cell, "length", keep), 2))]


def bins_table(results):
    """CQR (no histogram) and CHR, RLR-CHR at 100 and 1,000 bins on BlogFeedback."""
    rows = read_long(os.path.join(results, "ablations", "bins_blog_long.csv"))
    cells = cells_by(rows, lambda r: (r["method"], bins_of(r)), "bins")
    out = ["\\small", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
           "Method & Bins & Coverage & WSC$_{0.1}$ & Average length \\\\", "\\midrule"]
    for method, b in (("CQR", None), ("CHR", 100), ("CHR", 1000), ("RLR-CHR", 100),
                      ("RLR-CHR", 1000)):
        # Reference: CHR at the same number of bins (for CQR, 1,000) in the same repetition.
        ref = {rep: num(x["length"]) for rep, x in cells[("CHR", b or 1000)].items()}
        where = method if b is None else f"{method} at {b} bins"
        bins = "---" if b is None else (f"\\phantom{{0}}{b}" if b == 100 else "1{,}000")
        out.append(f"{method} & {bins} & "
                   + " & ".join(level_cells(cells[(method, b)], ref, method, where)) + " \\\\")
    return out + ["\\bottomrule", "\\end{tabular}"]


def t_table(results):
    """RLR-CHR on SGEMM GPU at each number of score levels T."""
    rows = read_long(os.path.join(results, "ablations", "t_sgemm_long.csv"))
    cells = cells_by(rows, lambda r: int(float(r["T"])), "T")
    out = ["\\small", "\\begin{tabular}{@{}rrrr@{}}", "\\toprule",
           "$T$ & Coverage & WSC$_{0.1}$ & Average length \\\\", "\\midrule"]
    ks = []
    for T in sorted(cells):
        cell = cells[T]
        # Reference: the length of CHR at the default T = 1,000 in the same repetition.
        ref = {rep: num(x["chr_length"]) for rep, x in cell.items()}
        k = sorted({int(float(x["k"])) for x in cell.values()})
        ks.append(f"T={T}: k={','.join(map(str, k))}")
        out.append(f"{T:,}".replace(",", "{,}") + " & "
                   + " & ".join(level_cells(cell, ref, "RLR-CHR", f"RLR-CHR at T={T}")) + " \\\\")
    print("neighbors per T: " + ", ".join(ks))
    return out + ["\\bottomrule", "\\end{tabular}"]


def alpha_table(results):
    """WSC of CHR and RLR-CHR at three slab masses and RLR-CHR's relative length at alpha = 0.05."""
    n = len(WSC_DELTA)
    heads = " & ".join(f"$\\delta={d}$" for _, d in WSC_DELTA)
    out = ["\\scriptsize", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}ll" + "r" * (2 * n + 1) + "@{}}", "\\toprule",
           f" & & \\multicolumn{{{n}}}{{c}}{{CHR WSC}} & \\multicolumn{{{n}}}{{c}}{{RLR-CHR WSC}} "
           "& RLR-CHR \\\\",
           f"\\cmidrule(lr){{3-{2 + n}}}\\cmidrule(lr){{{3 + n}-{2 + 2 * n}}}",
           f"$\\alpha$ & Dataset & {heads} & {heads} & rel.\\ length \\\\", "\\midrule"]
    for key, name in ALPHA_DATASETS:
        where = f"alpha=0.05 {name}"
        rows = read_long(os.path.join(results, "ablations", f"bins_alpha005_{key}_long.csv"))
        per = cells_by(rows, lambda r: r["method"], where)
        chr_cell, rlr_cell = per["CHR"], per["RLR-CHR"]
        ref = {rep: num(x["length"]) for rep, x in chr_cell.items()}
        keep, bad = split_degenerate(rlr_cell, ref, "RLR-CHR", where)
        reps = sorted(chr_cell)
        rel = {r: num(rlr_cell[r]["length"]) / ref[r] for r in reps}
        gaps = [mean(x - y for x, y in zip(column(rlr_cell, col, keep),
                                           column(chr_cell, col, keep)))
                for col, _ in WSC_DELTA]
        print(f"{where}: RLR-CHR minus CHR WSC at delta = 0.05, 0.10, 0.20: "
              + ", ".join(f"{g:+.3f}" for g in gaps))
        if bad:
            print(f"{where}, RLR-CHR: {len(bad)} of {len(reps)} repetitions degenerate {bad}; "
                  "all repetitions: WSC "
                  + ", ".join(f"{mean(column(rlr_cell, col, reps)):.4f}" for col, _ in WSC_DELTA)
                  + f", relative length {mean(rel[r] for r in reps):.3f} "
                  f"(median {median(rel[r] for r in reps):.3f})")
        # CHR over all repetitions, RLR-CHR over its non-degenerate ones.
        dg = DAGGER if bad else ""
        row = [mean_with_se(column(chr_cell, col, reps)) for col, _ in WSC_DELTA]
        row += [mean_with_se(column(rlr_cell, col, keep)) + dg for col, _ in WSC_DELTA]
        row.append(mean_with_se(rel[r] for r in keep) + dg)
        out.append(f"0.05 & {name} & " + " & ".join(row) + " \\\\")
    return out + ["\\bottomrule", "\\end{tabular}"]


def main():
    args = parse_args(__doc__)
    write_body(os.path.join(args.out, "t_ablation_table_body.tex"), t_table(args.results),
               se="paren")
    write_body(os.path.join(args.out, "bins_ablation_table_body.tex"), bins_table(args.results),
               se="paren")
    write_body(os.path.join(args.out, "alpha_table_body.tex"), alpha_table(args.results),
               se="paren")


if __name__ == "__main__":
    main()
