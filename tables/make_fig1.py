#!/usr/bin/env python3
"""Figure 1: worst-slab coverage and average length on the synthetic design, with SEs."""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import SMALL_FIGURE_STYLE, num, parse_args, save_figure

ORDER = ["CHR", "RLR-CHR", "RBC-CHR", "Norm-SC", "G2-Residual", "MCR"]
# Color, marker, line style, hollow marker. RLR-CHR and RBC-CHR share a color (solid and
# dashed), since the two are asymptotically equivalent.
PAIR = "#953F2F"
STYLE = {"CHR": ("#496079", "s", "-", False),
         "RLR-CHR": (PAIR, "o", "-", False),
         "RBC-CHR": (PAIR, "o", (0, (4, 2)), True),
         "Norm-SC": ("#6F8F79", "^", "-", False),
         "G2-Residual": ("#CCA45E", "D", "-", False),
         "MCR": ("#2E2E2E", "v", "-", False)}
MARKER_SIZE = {"s": 3.2, "o": 3.4, "^": 3.6, "D": 2.9, "v": 3.6}
NOMINAL = 0.9


def main():
    args = parse_args(__doc__)
    # Columns n_data, method, reps, wsc_mean, wsc_se, len_mean, len_se (aggregate_synthetic.py).
    src = os.path.join(args.results, "synthetic", "fig1_summary.csv")
    data = defaultdict(dict)
    reps = set()
    with open(src, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(float(r["n_data"]))
            data[r["method"]][n] = (num(r["wsc_mean"]), num(r["wsc_se"]),
                                    num(r["len_mean"]), num(r["len_se"]))
            reps.add(int(float(r["reps"])))
    ns = sorted({n for by_n in data.values() for n in by_n})
    xs = list(range(len(ns)))
    if any(sorted(data.get(method, {})) != ns for method in ORDER):
        sys.exit(f"{src}: every method needs every sample size")

    # STIX mathematics, Times-like as the text, for the label n.
    plt.rcParams.update({**SMALL_FIGURE_STYLE, "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(5.5, 2.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06)
    axes = fig.subplots(1, 2, sharex=True)
    for ax, (i_mu, i_se, ylab) in zip(axes, [(0, 1, "Worst-slab coverage"),
                                             (2, 3, "Average interval length")]):
        if i_mu == 0:
            ax.axhline(NOMINAL, color="#8C8C8C", linestyle=(0, (1, 1.5)), linewidth=0.8,
                       zorder=1)
        for i, method in enumerate(ORDER):
            color, marker, line, hollow = STYLE[method]
            ys = [data[method][n][i_mu] for n in ns]
            es = [data[method][n][i_se] for n in ns]
            ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, linestyle=line,
                        linewidth=0.9, markersize=MARKER_SIZE[marker], markeredgewidth=0.7,
                        markerfacecolor="none" if hollow else color,
                        elinewidth=0.6, capsize=1.0, capthick=0.4,
                        zorder=10 - i)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Sample size $n$")
    # Sample sizes equally spaced, labeled with their values.
    axes[0].set_xticks(xs, [str(n) for n in ns])
    axes[0].set_xlim(-0.3, len(ns) - 0.7)
    handles = []
    for method in ORDER:
        color, marker, line, hollow = STYLE[method]
        handles.append(Line2D([], [], color=color, marker=marker, linestyle=line, linewidth=0.9,
                              markersize=MARKER_SIZE[marker], markeredgewidth=0.7,
                              markerfacecolor="none" if hollow else color, label=method))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(ORDER), frameon=False,
               handlelength=2.4, columnspacing=1.4, handletextpad=0.5)
    save_figure(fig, os.path.join(args.out, "synthetic_wsc_length_with_se.pdf"), dpi=200)
    print(f"repetitions per point: {sorted(reps)}")
    for method in ORDER:
        wsc, wsc_se = data[method][ns[-1]][:2]
        print(f"{method} at n={ns[-1]}: WSC {wsc:.4f} (SE {wsc_se:.4f})")


if __name__ == "__main__":
    main()
