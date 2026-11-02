#!/usr/bin/env python3
"""Table 17: all eight methods at alpha = 0.20 on SGEMM GPU and Bike Sharing."""
import os

from common import (ALPHA_DATASETS, METHODS, ADDITIONAL_WSC, cell_tex, closest, load_cells, parse_args,
                    print_degenerate, shortest)
from table_bodies import write_body

ALPHA = 0.20
# Per dataset block: WSC at delta = 0.05, 0.10, 0.20, relative length, coverage.
WSC_STATS = (("wsc005", "0.05"), ("wsc", "0.10"), ("wsc020", "0.20"))
STATS = [s for s, _ in WSC_STATS] + ["rel", "cov"]


def table(cells):
    """One block of WSC, relative length and coverage per dataset."""
    n = len(WSC_STATS)
    out = ["\\scriptsize", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}l" + "r" * len(STATS) + "@{}}", "\\toprule"]
    for b, (key, name) in enumerate(ALPHA_DATASETS):
        # Bold: the WSC closest to 1 - alpha in each delta column and the shortest relative length.
        bold = {s: closest({method: cells[key][method][s] for method in METHODS}, 1 - ALPHA)
                for s, _ in WSC_STATS}
        bold["rel"] = shortest({method: cells[key][method]["rel"] for method in METHODS})
        if b:
            out.append("\\midrule[\\heavyrulewidth]")
        out += [f"\\textbf{{{name}}} & \\multicolumn{{{n}}}{{c}}{{WSC}}" + " &" * (len(STATS) - n)
                + " \\\\",
                f"\\cmidrule(lr){{2-{1 + n}}}",
                "Method" + "".join(f" & $\\delta={d}$" for _, d in WSC_STATS)
                + " & Rel.\\ length & Coverage \\\\",
                "\\midrule"]
        for method in METHODS:
            row = [cell_tex(cells[key][method], s, method in bold.get(s, ())) for s in STATS]
            out.append(f"{method} & " + " & ".join(row) + " \\\\")
    return out + ["\\bottomrule", "\\end{tabular}"]


def main():
    args = parse_args(__doc__)
    cells = {key: load_cells(os.path.join(args.results, "ablations", f"alpha020_{key}_long.csv"),
                             ADDITIONAL_WSC, ALPHA)[0]
             for key, _ in ALPHA_DATASETS}
    print_degenerate(cells, ALPHA_DATASETS)
    write_body(os.path.join(args.out, "alpha_a020_table_body.tex"), table(cells), se="paren")


if __name__ == "__main__":
    main()
