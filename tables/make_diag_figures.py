#!/usr/bin/env python3
"""Figures 2-3: SGEMM diagnostics by fitted-scale tertile and by neighborhood multiplier c."""
import csv
import glob
import gzip
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import WIDE_FIGURE_STYLE, num, parse_args, save_figure

GROUPS = ["Low Z", "Mid Z", "High Z"]   # tertiles of Z = log sigma_hat(X)
SCALE_COLS = ("q10", "q50", "q90", "score_cdf_09", "coverage", "rel_length")
K_COLS = ("coverage", "wsc", "rel_length", "length", "chr_length", "frac_top_level_rlr",
          "mean_t", "coverage_lowZ", "coverage_midZ", "coverage_highZ", "rel_length_lowZ",
          "rel_length_midZ", "rel_length_highZ", "inference_time")


def multiplier(row):
    """The multiplier c of a row, rounded so that it can serve as a key."""
    return round(num(row["multiplier"]), 6)


def read_csv(path):
    """Rows of a CSV file, gzipped or not."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def load(diag_dir):
    """Rows of the chunks sgemm_diag_reps<a>-<b>_*: per (rep, c), tertile, test point, rep."""
    metas = sorted(glob.glob(os.path.join(diag_dir, "sgemm_diag_reps*-*_meta.csv")))
    if not metas:
        sys.exit(f"no diagnostic results in {diag_dir}")
    k_rows, group_rows, test_rows, meta_rows = [], [], [], []
    for p in metas:
        stub = p[:-len("_meta.csv")]
        k_rows += read_csv(stub + "_k_long.csv")
        group_rows += read_csv(stub + "_scale_group_long.csv")
        test_rows += read_csv(stub + "_scale_test_long.csv.gz")
        meta_rows += read_csv(p)
    rep_c = [(int(float(r["rep"])), multiplier(r)) for r in k_rows]
    rep_group = [(int(float(r["rep"])), r["group"]) for r in group_rows]
    if len(set(rep_c)) != len(rep_c) or len(set(rep_group)) != len(rep_group):
        sys.exit("more than one row for a (rep, multiplier) or (rep, tertile)")
    mults = sorted({c for _, c in rep_c})
    # The number k of neighbors at each multiplier, which must not vary across repetitions.
    kmap = {}
    for r in k_rows:
        c, k = multiplier(r), int(float(r["k"]))
        if kmap.setdefault(c, k) != k:
            sys.exit(f"k at c={c} varies across repetitions")
    return k_rows, group_rows, test_rows, meta_rows, mults, kmap


def scale_stats(group_rows):
    """Per tertile, the per-repetition values of SCALE_COLS, and the range of the tertile means."""
    by = defaultdict(lambda: defaultdict(list))
    for r in group_rows:
        for col in SCALE_COLS:
            by[r["group"]][col].append(num(r[col]))
    by = {group: {col: np.array(v) for col, v in cols.items()} for group, cols in by.items()}
    ranges = {col: float(max(by[group][col].mean() for group in GROUPS)
                         - min(by[group][col].mean() for group in GROUPS)) for col in SCALE_COLS}
    return by, ranges


def k_stats(k_rows, mults):
    """Per multiplier, the per-repetition values of K_COLS, and the range of the means."""
    by = defaultdict(lambda: defaultdict(list))
    for r in k_rows:
        c = multiplier(r)
        for col in K_COLS:
            by[c][col].append(num(r[col]))
    by = {c: {col: np.array(v) for col, v in cols.items()} for c, cols in by.items()}
    ranges = {col: float(max(by[c][col].mean() for c in mults)
                         - min(by[c][col].mean() for c in mults))
              for col in ("coverage", "wsc", "rel_length")}
    return by, ranges


def mean_se(v):
    """Mean, standard error and standard deviation of an array, in numpy arithmetic."""
    std = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return float(v.mean()), std / math.sqrt(v.size), std


def summary_rows(group_rows, k_rows, meta_rows, mults, kmap, gby, granges, kby, kranges):
    """Rows (section, quantity, level, mean, se, sd, n, note) of the diagnostics summary."""
    rows = []

    def add(section, quantity, level, v=None, est=None, note=""):
        if v is None:
            rows.append([section, quantity, level, est, "", "", "", note])
        else:
            v = np.asarray(v, dtype=float)
            rows.append([section, quantity, level, *mean_se(v), v.size, note])

    # Per tertile of Z, then High Z minus Low Z within each repetition.
    for grp in GROUPS:
        for col in SCALE_COLS:
            add("scale", col, grp, gby[grp][col])
    per = defaultdict(dict)
    for r in group_rows:
        per[int(float(r["rep"]))][r["group"]] = r
    for grp in GROUPS:
        cr = [num(per[i][grp]["q90"]) - num(per[i][grp]["q10"]) for i in sorted(per)]
        add("scale", "q90_minus_q10", grp, cr, note="central residual range")
        for col in ("frac_test_score_gt_T", "chr_coverage", "length", "chr_length"):
            add("scale", col, grp, [num(per[i][grp][col]) for i in sorted(per)])
    for col, v in granges.items():
        add("scale", f"range_{col}", "max-min of group means", est=v)
    for col in ("score_cdf_09", "coverage", "rel_length", "q90", "q10"):
        dv = [num(per[i]["High Z"][col]) - num(per[i]["Low Z"][col]) for i in sorted(per)]
        add("scale_paired", f"high_minus_low_{col}", "High Z - Low Z", dv,
            note="paired by repetition (same fits)")
    # Per multiplier c, then each c minus c = 1 within each repetition.
    for c in mults:
        for col in K_COLS:
            add("k", col, f"c={c:.2f} k={kmap[c]}", kby[c][col])
    for col, v in kranges.items():
        add("k", f"range_{col}", "max-min of c means", est=v)
    perk = defaultdict(dict)
    for r in k_rows:
        perk[int(float(r["rep"]))][multiplier(r)] = r
    for c in mults:
        if c == 1.0:
            continue
        for col in ("coverage", "wsc", "rel_length"):
            dv = [num(perk[i][c][col]) - num(perk[i][1.0][col]) for i in sorted(perk)]
            add("k_paired", f"{col}_minus_c1", f"c={c:.2f} k={kmap[c]}", dv,
                note="paired by repetition (same fits)")
    for col in ("fit_seconds", "sweep_seconds", "total_seconds"):
        add("timing", col, "per rep", [num(r[col]) for r in meta_rows])
    return rows


def style_axis(ax):
    """Dashed grid, no top or right spine."""
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="#b0b0b0", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=4, width=0.8)


def ecdf_curve(values, grid):
    """Empirical CDF of the finite values at the grid points."""
    v = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if v.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    return np.searchsorted(v, grid, side="right") / v.size


def make_figures(test_rows, kby, mults, kmap, out):
    """Figure 2, the scale-stratified ECDFs, and Figure 3, the k-sensitivity panels."""
    plt.rcParams.update(WIDE_FIGURE_STYLE)
    style = {"Low Z": ("#0072B2", "-"), "Mid Z": ("#E69F00", "--"), "High Z": ("#009E73", "-.")}
    grp = np.array([r["group"] for r in test_rows])
    res = np.array([num(r["std_residual"]) for r in test_rows])
    score = np.array([num(r["score_norm"]) for r in test_rows])
    # Pooled ECDFs over all test points: residuals on their 1%-99% range, E/T on [0, 1].
    lo, hi = np.nanquantile(res, [0.01, 0.99])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        lo, hi = np.nanmin(res), np.nanmax(res)
    xg = np.linspace(lo, hi, 400)
    ug = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), dpi=300)
    handles = []
    ax = axes[0]
    for gname in GROUPS:
        col, ls = style[gname]
        (ln,) = ax.plot(xg, ecdf_curve(res[grp == gname], xg), color=col, linestyle=ls,
                        linewidth=1.5, label=gname)
        handles.append(ln)
    ax.set_title("Residual ECDF by Fitted Scale", pad=8, fontweight="semibold")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1)
    style_axis(ax)
    ax = axes[1]
    ax.axvline(0.9, color="black", linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.75, zorder=0)
    for gname in GROUPS:
        col, ls = style[gname]
        ax.plot(ug, ecdf_curve(score[grp == gname], ug), color=col, linestyle=ls, linewidth=1.5)
    ax.set_title("Score ECDF by Fitted Scale", pad=8, fontweight="semibold")
    ax.set_xlabel(r"Normalized residual CHR score $E/T$")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axis(ax)
    fig.legend(handles, GROUPS, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.095, 1, 1], w_pad=2.2)
    save_figure(fig, os.path.join(out, "scale_stratified_residual_score_diagnostic.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # WSC and relative length versus c, +/- 1 SE, with k under each tick.
    x = np.array(mults, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), dpi=300, sharex=True)
    panels = (("wsc", r"WSC$_{0.1}$", r"WSC versus $k$-multiplier", 0.9),
              ("rel_length", "Relative length", r"Relative length versus $k$-multiplier", 1.0))
    for ax, (col, ylab, title, href) in zip(axes, panels):
        y, err, _ = np.array([mean_se(kby[c][col]) for c in mults]).T
        err = np.where(np.isfinite(err), err, 0.0)
        ax.errorbar(x, y, yerr=err, marker="s", linestyle="-",
                    color="#222222", linewidth=1.5, markersize=4.0, markeredgewidth=0.8,
                    capsize=2.5, elinewidth=1.0, label="RLR-CHR")
        ax.axhline(href, color="black", linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.75, zorder=0)
        ax.axvline(1.0, color="black", linestyle=(0, (2, 3)), linewidth=1.0, alpha=0.55, zorder=0)
        ax.set_title(title, pad=8, fontweight="semibold")
        ax.set_xlabel(r"Multiplier $c$ ($k$)")
        ax.set_ylabel(ylab)
        ymin, ymax = min(np.nanmin(y - err), href), max(np.nanmax(y + err), href)
        pad = max(0.005, (ymax - ymin) * 0.18)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:g}\n({kmap[c]})" for c in mults])
        style_axis(ax)
    fig.legend([axes[0].containers[0]], ["RLR-CHR ($\\pm 1$ SE)"], frameon=False,
               loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.005), handlelength=2.2)
    fig.tight_layout(rect=[0, 0.095, 1, 1], w_pad=2.2)
    save_figure(fig, os.path.join(out, "k_sensitivity_local_rank_diagnostic.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args(__doc__)
    k_rows, group_rows, test_rows, meta_rows, mults, kmap = load(
        os.path.join(args.results, "diagnostics"))
    gby, granges = scale_stats(group_rows)
    kby, kranges = k_stats(k_rows, mults)
    # The numbers that the text quotes from these diagnostics.
    os.makedirs(args.out, exist_ok=True)
    summary = os.path.join(args.out, "diagnostics_summary.csv")
    with open(summary, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "quantity", "level", "mean", "se", "sd", "n", "note"])
        writer.writerows(summary_rows(group_rows, k_rows, meta_rows, mults, kmap,
                                      gby, granges, kby, kranges))
    print(f"wrote {summary}")
    make_figures(test_rows, kby, mults, kmap, args.out)


if __name__ == "__main__":
    main()
