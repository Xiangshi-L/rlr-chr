#!/usr/bin/env python3
"""Calibration of the worst-slab coverage (WSC) estimator when conditional coverage is exact."""
import argparse
import csv
import os
import time
from pathlib import Path

# Set unset BLAS thread limits to 1 before importing NumPy.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20260723
ALPHA = 0.1  # every point is covered with probability 1 - ALPHA
DELTA = 0.1  # slab mass


def worst_slab(z, covered, delta):
    """Lowest-coverage slab [a, b] of z with more than round(delta n) points; the first wins ties.

    This is the slab rule of CHR's coverage.wsc, through which the synthetic study computes WSC.
    The engine's worst_slab (rlrchr/conformal_core.cpp) admits slabs of at least ceil(delta n)
    points and keeps the shortest of tied slabs. CHR's wsc_unbiased and the engine search for the
    slab on floor(0.25 n) points; wsc_split below takes round(0.25 n), the same number at the n
    used here.
    """
    order = np.argsort(z)
    z = z[order]
    n = len(z)
    # Covered points before each index.
    count = np.concatenate(([0.0], np.cumsum(covered[order].astype(float))))
    w = int(np.round(delta * n))
    best, a_best, b_best = count[n] / n, 0, n - 1  # whole sample unless a slab is strictly lower
    for a in range(min(int(np.round((1.0 - delta) * n)), n - w)):
        cov = (count[a + w + 1:] - count[a]) / np.arange(w + 1, n - a + 1)
        j = int(np.argmin(cov))
        if cov[j] < best:
            best, a_best, b_best = cov[j], a, a + w + j
    return best, z[a_best], z[b_best]


def directions(M, p, rng):
    """M directions drawn uniformly on the unit sphere of R^p."""
    V = rng.standard_normal((M, p))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def wsc_same_sample(X, cover, delta, M, rng):
    """WSC with the slab searched and evaluated on the same points."""
    best = 1.0
    for v in directions(M, X.shape[1], rng):
        c, _, _ = worst_slab(X @ v, cover, delta)
        best = min(best, c)
    return best


def wsc_split(X, cover, delta, M, rng, search_frac=0.25):
    """WSC with the slab searched on a subsample (25% by default) and evaluated on the rest."""
    n = len(cover)
    idx = rng.permutation(n)
    n_search = int(np.round(search_frac * n))
    search, held_out = idx[:n_search], idx[n_search:]
    V = directions(M, X.shape[1], rng)
    best, vstar, astar, bstar = 1.0, V[0], -np.inf, np.inf
    for v in V:
        c, a, b = worst_slab(X[search] @ v, cover[search], delta)
        if c < best:
            best, vstar, astar, bstar = c, v, a, b
    z = X[held_out] @ vstar
    sel = (z >= astar) & (z <= bstar)
    return cover[held_out][sel].mean() if sel.any() else np.nan


def summary(config, values):
    return dict(config=config, mean=np.mean(values), q05=np.quantile(values, 0.05),
                q95=np.quantile(values, 0.95), nsim=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "results"), help="results root")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    # Minimum coverage over 10 disjoint slabs of 100 points.
    mins = [min(rng.binomial(100, 1 - ALPHA, 10)) / 100 for _ in range(20000)]
    rows = [summary("binomial_min_10_slabs", mins)]
    # Same-sample and split WSC, coverage independent of X: p = 1, n = 1000 and p = 10, n = 2000.
    for tag, p, n, M, nsim in [("p1_n1000", 1, 1000, 100, 200),
                               ("p10_n2000", 10, 2000, 100, 200)]:
        same, split = [], []
        t0 = time.time()
        for _ in range(nsim):
            X = rng.standard_normal((n, p))
            cover = rng.random(n) < (1 - ALPHA)
            same.append(wsc_same_sample(X, cover, DELTA, M, rng))
            split.append(wsc_split(X, cover, DELTA, M, rng))
        rows.append(summary(f"{tag}_same_sample", same))
        rows.append(summary(f"{tag}_split", split))
        print(f"{tag}: same sample {np.mean(same):.4f}, split {np.mean(split):.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    out = Path(args.out) / "wsc_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "mean", "q05", "q95", "nsim"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
