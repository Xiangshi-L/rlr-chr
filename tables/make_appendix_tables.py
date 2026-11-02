#!/usr/bin/env python3
"""Tables 3-10: complete results for each of the eight main datasets."""
import os
from decimal import Decimal

from common import DAGGER, MAIN_DATASETS, METHODS, load_main, mean, num, parse_args
from table_bodies import write_body


def sig4(v):
    """Four significant figures in plain notation (11.8975 -> 11.90, 0.0035858 -> 0.003586)."""
    return format(Decimal(f"{v:.3e}"), "f")


def label(method):
    """A method name whose hyphens do not break across lines."""
    return method.replace("-", "\\mbox{-}")


def table(cells, rlcp, rows):
    """One dataset: coverage, length and WSC over the non-degenerate repetitions."""
    out = ["\\scriptsize", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}lrrrrrrr@{}}",
           "\\toprule",
           "\\textbf{Method} & Coverage & SE & Length & SD(Len.) & Rel.\\ length & "
           "WSC & SE(WSC) \\\\", "\\midrule"]
    for method in METHODS:
        cell = cells[method]
        dg = DAGGER if cell["bad"] else ""
        out.append(f"{label(method)} & {cell['cov']:.4f}{dg} & {cell['cov_se']:.4f}{dg} & "
                   f"{sig4(cell['len'])}{dg} & {sig4(cell['len_sd'])}{dg} & "
                   f"{cell['rel']:.4f}{dg} & {cell['wsc']:.4f}{dg} & {cell['wsc_se']:.4f}{dg} \\\\")
    # RLCP: Length and SD(Len.) summarize the per-repetition median widths.
    out += ["\\midrule",
            f"RLCP & {rlcp['cov']:.4f} & {rlcp['cov_se']:.4f} & {sig4(rlcp['len'])} & "
            f"{sig4(rlcp['len_sd'])} & {rlcp['rel']:.4f} & {rlcp['wsc']:.4f} & "
            f"{rlcp['wsc_se']:.4f} \\\\",
            "\\bottomrule", "\\end{tabular}%", "}"]
    return out


def main():
    args = parse_args(__doc__)
    cells, rlcp, rows = load_main(args.results)
    for key, _ in MAIN_DATASETS:
        # File names end in the first word of the dataset key (sarcos_torque1 -> sarcos).
        slug = key.split("_")[0]
        write_body(os.path.join(args.out, f"appendix_tables_real-complete-{slug}_body.tex"),
                   table(cells[key], rlcp[key], rows[key]), se="columns")


if __name__ == "__main__":
    main()
