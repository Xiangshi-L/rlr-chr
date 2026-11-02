#!/usr/bin/env python3
"""Figure 4: the worked example of local-rank calibration, its neighborhood and p-value p(e)."""
import os
import sys
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from common import SMALL_FIGURE_STYLE, parse_args, save_figure

# The example: m = 5 calibration points, test point 6, T = 4 levels, k = 2, alpha = 0.4.
Z_CAL = [0.10, 0.30, 0.50, 0.85, 1.40]   # perturbed localization coordinates Z_1..Z_5
E_CAL = [2, 1, 3, 2, 4]                  # residual CHR scores E_1..E_5
Z_TEST = 0.60                            # Z_6
T, K, ALPHA = 4, 2, Fraction(2, 5)

INK = "#222222"
MUTED = "#777777"
GRAY_LINE = "#8C8C8C"
SHADE = "#EEEEEE"


def neighborhoods(z, k):
    """N_j: j and its k nearest neighbors in Z (1-based); a tie at the k-th neighbor is an error."""
    # Rounded distances, so that floating-point noise cannot reorder equal gaps.
    idx = range(1, len(z) + 1)
    out = {}
    for j in idx:
        d = sorted((round(abs(z[ell - 1] - z[j - 1]), 10), ell) for ell in idx if ell != j)
        if d[k - 1][0] == d[k][0]:
            sys.exit(f"tie at the k-NN boundary for j={j}: {d[:k + 1]}")
        out[j] = {j} | {ell for _, ell in d[:k]}
    return out


def local_rank_path(z_cal, e_cal, z_test, k, levels):
    """N and {e: (A_1..A_{m+1}, p(e))}, with E_{m+1} = e."""
    # A_j(e) = sum_{ell in N_j} 1{E_ell(e) <= E_j(e)} and
    # p(e) = (1 + sum_{i <= m} 1{A_i(e) >= A_{m+1}(e)}) / (m + 1).
    m = len(z_cal)
    test = m + 1
    N = neighborhoods(z_cal + [z_test], k)
    path = {}
    for e in levels:
        E = dict(enumerate(e_cal, start=1))
        E[test] = e
        A = {j: sum(E[ell] <= E[j] for ell in N[j]) for j in N}
        p = Fraction(1 + sum(A[i] >= A[test] for i in range(1, m + 1)), m + 1)
        path[e] = ([A[j] for j in sorted(A)], p)
    return N, path


def main():
    args = parse_args(__doc__, results=False)
    m = len(Z_CAL)
    test = m + 1
    levels = list(range(1, T + 1))
    N, path = local_rank_path(Z_CAL, E_CAL, Z_TEST, K, levels)
    # The p-value p(e) is nonincreasing in e, so the accepted levels are {1, ..., e*}.
    accepted = [e for e in levels if path[e][1] > ALPHA]
    e_star = max(accepted) if accepted else 1
    for e in sorted(levels, reverse=True):
        A, p = path[e]
        print(f"e={e}: A_1..A_{test} = {A}  p(e) = {p * (m + 1)}/{m + 1}"
              f" = {float(p):.3f}  {'accept' if p > ALPHA else 'reject'}")
    print(f"N_{test} = {sorted(N[test])},  e* = {e_star}")

    nbr = sorted(N[test] - {test})                 # calibration members of N_6
    lo = min(Z_CAL[i - 1] for i in nbr)
    hi = max(Z_CAL[i - 1] for i in nbr)
    lo, hi = min(lo, Z_TEST), max(hi, Z_TEST)      # Z-span of N_6

    plt.rcParams.update(SMALL_FIGURE_STYLE)
    fig = plt.figure(figsize=(5.5, 2.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.08)
    ax_a, ax_b = fig.subplots(1, 2)

    # (a) The neighborhood of the test point.
    ax_a.axvspan(lo, hi, color=SHADE, linewidth=0, zorder=0)
    ax_a.axvline(Z_TEST, color=INK, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    for i, (z, e) in enumerate(zip(Z_CAL, E_CAL), start=1):
        color = INK if i in nbr else MUTED
        ax_a.plot(z, e, marker="o", markersize=3.6, color=color, markeredgewidth=0.7,
                  markerfacecolor=INK if i in nbr else "white", linestyle="none", zorder=3)
        ax_a.annotate(str(i), (z, e), xytext=(3.0, 2.5), textcoords="offset points",
                      fontsize=6.5, color=INK, ha="left", va="bottom")
    ax_a.text(Z_TEST + 0.025, 4.42, "test point", color=INK, fontsize=7,
              ha="left", va="top")
    ax_a.text((Z_TEST + hi) / 2, 0.62, r"$\mathcal{N}_6$", color=INK, fontsize=8,
              ha="center", va="bottom")
    ax_a.set_xlim(0.0, 1.5)
    ax_a.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0, 1.5]))
    ax_a.set_ylim(0.5, 4.5)
    ax_a.yaxis.set_major_locator(FixedLocator(levels))
    ax_a.set_xlabel(r"Localization coordinate $\widetilde Z$")
    ax_a.set_ylabel("Residual CHR score $E_i$")
    ax_a.set_title("(a)", loc="left", fontsize=8, pad=3)

    # (b) The p-value over the candidate levels.
    ax_b.axhline(float(ALPHA), color=GRAY_LINE, linestyle=(0, (1, 1.5)),
                 linewidth=0.8, zorder=1)
    ps = [float(path[e][1]) for e in levels]
    ax_b.plot(levels, ps, color=INK, linewidth=0.9, zorder=2)
    for e, p in zip(levels, ps):
        ok = path[e][1] > ALPHA
        ax_b.plot(e, p, marker="o", markersize=3.6, color=INK, markeredgewidth=0.7,
                  markerfacecolor=INK if ok else "white", linestyle="none", zorder=3)
    ax_b.plot(e_star, float(path[e_star][1]), marker="o", markersize=7.5,
              markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.6,
              linestyle="none", zorder=3)
    ax_b.annotate(rf"$e^\star={e_star}$", (e_star, float(path[e_star][1])),
                  xytext=(6, 5), textcoords="offset points", fontsize=7.5,
                  color=INK, ha="left", va="bottom")
    ax_b.text(levels[-1] + 0.38, float(ALPHA) + 0.015, rf"$\alpha={float(ALPHA):g}$", color=INK,
              fontsize=7, ha="right", va="bottom")
    ax_b.set_xlim(0.6, levels[-1] + 0.4)
    ax_b.xaxis.set_major_locator(FixedLocator(levels))
    ax_b.set_ylim(0.0, 1.08)
    yt = [Fraction(j, m + 1) for j in range(0, m + 2)]
    ax_b.yaxis.set_major_locator(FixedLocator([float(f) for f in yt]))
    ax_b.yaxis.set_major_formatter(FixedFormatter(
        ["0"] + [f"{j}/{m + 1}" for j in range(1, m + 2)]))
    ax_b.set_xlabel("Candidate level $e$")
    ax_b.set_ylabel("Conformal $p$-value $p(e)$")
    ax_b.set_title("(b)", loc="left", fontsize=8, pad=3)

    save_figure(fig, os.path.join(args.out, "worked_example.pdf"))


if __name__ == "__main__":
    main()
