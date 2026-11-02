#!/usr/bin/env python3
"""Skew-reversal design: the Figure 1 design with skew-normal errors whose skew changes sign
at x = 0.5."""

from __future__ import annotations

import os
from pathlib import Path

# Default to single-threaded BLAS before numpy and torch are loaded.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from run_synthetic import interval_metrics, location_scale, parse_args, run_all

SIZES = "500,1000,3000"
METHODS = "CHR,Norm-SC,MCR,G2-Residual,RLR-CHR"


def sample_skew_reversal(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Figure 1 design with standardized skew-normal eps, of shape 4 on X < 0.5 and -4 above."""
    x = np.random.uniform(0.0, 1.0, size=n)
    f_x, sigma_x = location_scale(x)
    # Mean and standard deviation of the skew-normal with shape lam.
    lam = np.where(x < 0.5, 4.0, -4.0)
    d = lam / np.sqrt(1.0 + lam ** 2)
    mean_sn = d * np.sqrt(2.0 / np.pi)
    sd_sn = np.sqrt(np.maximum(1.0 - 2.0 * d ** 2 / np.pi, 1e-12))
    eps = (skewnorm.rvs(a=lam, size=n) - mean_sn) / sd_sn
    return x.reshape(-1, 1), f_x + sigma_x * eps


def interval_metrics_by_region(
    lo: np.ndarray, hi: np.ndarray, y: np.ndarray, x: np.ndarray
) -> dict[str, float]:
    """Coverage, length and WSC, plus coverage and length on each side of x = 0.5 and of x = 0.7."""
    out = interval_metrics(lo, hi, y, x)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    covered = (y >= lo) & (y <= hi)
    length = hi - lo
    for cut, tag in ((0.5, "05"), (0.7, "07")):
        left = x[:, 0] < cut
        right = ~left
        out[f"coverage_xlt{tag}"] = float(np.mean(covered[left])) if left.any() else float("nan")
        out[f"coverage_xge{tag}"] = float(np.mean(covered[right])) if right.any() else float("nan")
        out[f"length_xlt{tag}"] = float(np.mean(length[left])) if left.any() else float("nan")
        out[f"length_xge{tag}"] = float(np.mean(length[right])) if right.any() else float("nan")
    return out


def main() -> None:
    cfg, args = parse_args(__doc__, sample_sizes=SIZES, methods=METHODS)
    rows = run_all(args.sample_sizes, args.reps, cfg, args.n_workers, "skew_reversal",
                   sample_skew_reversal, interval_metrics_by_region)
    out = Path(args.out) / "synthetic" / "skew_long.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
