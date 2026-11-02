#!/usr/bin/env python3
"""Tables 11-12: the additional datasets, and their WSC at slab masses 0.05 and 0.20."""
import csv
import os

from common import (METHODS, ADDITIONAL_DATASETS, ADDITIONAL_WSC, cell_tex, closest, load_cells,
                    paired_table, parse_args, print_degenerate, print_share_above_T)
from table_bodies import write_body


def zero_share(data_dir, key):
    """Percent of responses equal to zero in the whole processed dataset."""
    with open(os.path.join(data_dir, f"{key}.csv"), newline="") as fh:
        ys = [float(r["target"]) for r in csv.DictReader(fh)]
    return 100 * sum(1 for y in ys if y == 0) / len(ys)


def delta_table(cells):
    """WSC at delta = 0.05 (upper block) and 0.20 (lower block), all datasets in one row."""
    n = len(ADDITIONAL_DATASETS)
    out = ["\\scriptsize", "\\setlength{\\tabcolsep}{5pt}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{@{}l" + "r" * n + "@{}}", "\\toprule"]
    for i, (heading, stat) in enumerate((("WSC at $\\delta=0.05$", "wsc005"),
                                         ("WSC at $\\delta=0.20$", "wsc020"))):
        if i:
            out.append("\\midrule[\\heavyrulewidth]")
        out += [f"& \\multicolumn{{{n}}}{{c}}{{{heading}}} \\\\",
                f"\\cmidrule(lr){{2-{1 + n}}}",
                "\\textbf{Method}" + "".join(f" & \\textbf{{{name}}}" for _, name in ADDITIONAL_DATASETS)
                + " \\\\",
                "\\midrule"]
        # Bold: the WSC closest to 0.9.
        bold = {key: closest({method: cells[key][method][stat] for method in METHODS}, 0.9)
                for key, _ in ADDITIONAL_DATASETS}
        for method in METHODS:
            out.append(f"{method} & "
                       + " & ".join(cell_tex(cells[key][method], stat, method in bold[key])
                                    for key, _ in ADDITIONAL_DATASETS) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}%", "}"]
    return out


def main():
    args = parse_args(__doc__, data=True)
    files = {key: os.path.join(args.results, "additional", f"{key}_long.csv")
             for key, _ in ADDITIONAL_DATASETS}
    cells = {key: load_cells(path, ADDITIONAL_WSC)[0] for key, path in files.items()}
    print_degenerate(cells, ADDITIONAL_DATASETS)
    print_share_above_T({name: files[key] for key, name in ADDITIONAL_DATASETS}, "additional datasets")
    zeros = {key: f"{zero_share(args.data, key):.1f}" for key, _ in ADDITIONAL_DATASETS}
    write_body(os.path.join(args.out, "additional_table_merged_body.tex"),
               paired_table(cells, ADDITIONAL_DATASETS, zeros=zeros))
    write_body(os.path.join(args.out, "additional_delta_table_body.tex"), delta_table(cells),
               se="paren")


if __name__ == "__main__":
    main()
