#!/usr/bin/env python3
"""Figure 1 study: CHR, Norm-SC, MCR, G2-Residual, RBC-CHR and RLR-CHR on a skewed
heteroskedastic design."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# Default to single-threaded BLAS before importing numpy and torch; keep existing settings.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import crepes
import numpy as np
import pandas as pd
import torch
from crepes.extras import DifficultyEstimator, MondrianCategorizer
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
CHR_DIR = REPO / "third_party" / "chr"
METHODS = "CHR,Norm-SC,MCR,G2-Residual,RBC-CHR,RLR-CHR"
SIZES = "100,200,300,500,1000,2000,5000,8000"
SEED_BASE = 1000  # repetition r uses seed SEED_BASE + r
SPLIT_SEED = 2020  # training/calibration split, cross-fitting folds and WSC split, as in CHR
JITTER_STREAM = 20261010  # the localizer jitter is drawn from default_rng([seed, JITTER_STREAM])
LOCALIZER_JITTER = 1e-9


def chr_dir() -> Path:
    """The CHR clone named by --chr_dir, read at import so that worker processes find it too."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chr_dir", default=str(CHR_DIR))
    path = Path(parser.parse_known_args()[0].chr_dir).resolve()
    if not (path / "chr").is_dir():
        sys.exit(f"CHR not found in {path}: clone and patch it as described in README.md")
    return path


sys.path.insert(0, str(chr_dir()))
from chr import coverage as chr_coverage  # noqa: E402
from chr.black_boxes import QNet  # noqa: E402
from chr.grey_boxes import HistogramAccumulator  # noqa: E402
from chr.histogram import Histogram  # noqa: E402
from chr.methods import CHR  # noqa: E402


@dataclass(frozen=True)
class Config:
    n_test: int
    alpha: float
    y_steps: int
    delta_alpha: float
    qnet_epochs: int
    qnet_batch_size: int
    qnet_calibrate: int
    nuisance_epochs: int
    nuisance_calibrate: int
    n_folds: int
    methods: tuple[str, ...]


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def rep_range(s: str) -> range:
    """Repetitions A:B with A <= B, or the single repetition A."""
    m = re.fullmatch(r"(-?\d+)(?::(-?\d+))?", s)
    if m is None or (m[2] is not None and int(m[1]) > int(m[2])):
        raise argparse.ArgumentTypeError(f"expected A:B with A <= B, or A, got {s!r}")
    a = int(m[1])
    b = a if m[2] is None else int(m[2])
    return range(a, b + 1)


def method_list(s: str) -> tuple[str, ...]:
    """Comma list of method labels, each one of METHODS."""
    methods = tuple(s.split(","))
    unknown = [m for m in methods if m not in METHODS.split(",")]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown method {','.join(unknown)!r}; methods: {METHODS}"
        )
    return methods


def parse_args(
    description: str = __doc__, sample_sizes: str = SIZES, methods: str = METHODS
) -> tuple[Config, argparse.Namespace]:
    """Flags of the form --name=value, with the paper's settings as defaults."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sample_sizes", type=int_list, default=sample_sizes,
                   help="sizes n of the data, split evenly into training and calibration")
    p.add_argument("--reps", type=rep_range, default="1:100",
                   help="repetitions A:B (or A); repetition r uses seed 1000 + r")
    p.add_argument("--n_workers", type=int, default=3, help="worker processes")
    p.add_argument("--n_test", type=int, default=1000, help="size of the independent test set")
    p.add_argument("--alpha", type=float, default=0.10, help="miscoverage level")
    p.add_argument("--y_steps", type=int, default=1000, help="size of the CHR response grid")
    p.add_argument("--delta_alpha", type=float, default=0.001, help="step of the CHR alpha grid")
    p.add_argument("--qnet_epochs", type=int, default=2000,
                   help="epochs of the quantile networks of CHR, RBC-CHR and RLR-CHR; a cap "
                        "when --qnet_calibrate > 0, below which CHR's QNet chooses the count "
                        "by validation")
    p.add_argument("--qnet_batch_size", type=int, default=1000,
                   help="batch size of every QNet, the median networks included; 500 or more "
                        "trains on the full batch")
    p.add_argument("--qnet_calibrate", type=int, default=10,
                   help="number of seeded 80/20 splits whose median validation loss chooses "
                        "the epoch count, in steps of 10 up to --qnet_epochs, for the final fit "
                        "on all points; 0 trains exactly --qnet_epochs")
    p.add_argument("--nuisance_epochs", type=int, default=500,
                   help="epochs of the median networks (mu_hat, sigma_hat and the MCR "
                        "difficulty estimate); a cap when --nuisance_calibrate > 0")
    p.add_argument("--nuisance_calibrate", type=int, default=0,
                   help="as --qnet_calibrate, for the median networks and up to --nuisance_epochs")
    p.add_argument("--n_folds", type=int, default=5,
                   help="cross-fitting folds for mu_hat and sigma_hat")
    p.add_argument("--methods", type=method_list, default=methods,
                   help=f"comma list from {METHODS}; leaving methods out changes the random "
                        "draws of the methods that run after them; the paper's numbers use the "
                        "default list")
    p.add_argument("--out", default=str(REPO / "results"), help="results root")
    p.add_argument("--chr_dir", default=str(CHR_DIR), help="clone of msesia/chr")
    args = p.parse_args()
    cfg = Config(
        n_test=args.n_test,
        alpha=args.alpha,
        y_steps=args.y_steps,
        delta_alpha=args.delta_alpha,
        qnet_epochs=args.qnet_epochs,
        qnet_batch_size=args.qnet_batch_size,
        qnet_calibrate=args.qnet_calibrate,
        nuisance_epochs=args.nuisance_epochs,
        nuisance_calibrate=args.nuisance_calibrate,
        n_folds=args.n_folds,
        methods=args.methods,
    )
    return cfg, args


def quantile_grid() -> np.ndarray:
    """The 197 quantile levels 0.010, 0.015, ..., 0.990 of the CHR network."""
    return np.arange(0.01, 0.995, 0.005)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f(x) = 2 sin(2 pi x) + (x - 0.5) / 2 and sigma(x), which is 0.15 on x < 0.7 and 1.5 above."""
    return 2.0 * np.sin(2.0 * np.pi * x) + 0.5 * (x - 0.5), np.where(x < 0.70, 0.15, 1.50)


def sample_hetero_skewed(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Figure 1 design: X ~ U(0, 1), Y = f(X) + sigma(X) eps, eps = (Exp(1) - 1) + 0.1 N(0, 1)."""
    x = np.random.uniform(0.0, 1.0, size=n)
    f_x, sigma_x = location_scale(x)
    eps = (np.random.exponential(scale=1.0, size=n) - 1.0) + 0.10 * np.random.normal(size=n)
    return x.reshape(-1, 1), f_x + sigma_x * eps


def make_split(n_data: int, n_test: int, seed: int, sample: Callable) -> tuple[np.ndarray, ...]:
    """n_data points split evenly into training and calibration, plus an independent test set."""
    set_seed(seed)
    x_data, y_data = sample(n_data)
    x_test, y_test = sample(n_test)
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_data, y_data, test_size=0.5, random_state=SPLIT_SEED
    )
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def fit_qnet(
    x: np.ndarray, y: np.ndarray, quantiles: np.ndarray, epochs: int, calibrate: int, cfg: Config
) -> QNet:
    """CHR's quantile network at the given levels, trained on (x, y)."""
    model = QNet(
        quantiles,
        x.shape[1],
        no_crossing=True,
        batch_size=cfg.qnet_batch_size,
        dropout=0.1,
        num_epochs=epochs,
        learning_rate=0.0005,
        calibrate=calibrate,
        verbose=False,
    )
    model.fit(x.astype(np.float32), y.astype(np.float32))
    return model


def fit_median(x: np.ndarray, y: np.ndarray, cfg: Config) -> QNet:
    """Median network of y given x, for mu_hat, sigma_hat and the residuals behind MCR's bins."""
    return fit_qnet(x, y, np.array([0.5]), cfg.nuisance_epochs, cfg.nuisance_calibrate, cfg)


def predict_median(model: QNet, x: np.ndarray) -> np.ndarray:
    return model.predict(x.astype(np.float32))[:, 0].astype(float)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    rank = int(math.ceil((1.0 - alpha) * (len(scores) + 1)))
    vals = np.sort(np.concatenate([scores, np.array([np.inf])]))
    return float(vals[rank - 1])


def interval_metrics(
    lo: np.ndarray, hi: np.ndarray, y: np.ndarray, x: np.ndarray
) -> dict[str, float]:
    bands = np.column_stack([lo, hi])
    # Endpoints ordered as in CHR's evaluate_predictions and coverage.py (min and max per row).
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    covered = (y >= lo) & (y <= hi)
    # wsc_unbiased draws its M directions from the global numpy stream.
    wsc = float(chr_coverage.wsc_unbiased(x, y, bands, M=100, random_state=SPLIT_SEED))
    return {
        "coverage": float(np.mean(covered)),
        "length": float(np.mean(hi - lo)),
        "wsc": wsc,
    }


def mcr_n_bins(n_cal: int, alpha: float, max_bins: int = 20) -> int:
    """Choose up to max_bins Mondrian bins, targeting ceil(2 / alpha) calibration points per bin."""
    min_per_bin = max(1, int(math.ceil(2 / alpha)))
    return max(1, min(max_bins, n_cal // min_per_bin))


def cross_fit(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, n_folds: int, cfg: Config
) -> np.ndarray:
    """Out-of-fold predictions of the median network of y given x."""
    pred = np.empty(len(y))
    for fold in range(n_folds):
        out = folds == fold
        pred[out] = predict_median(fit_median(x[~out], y[~out], cfg), x[out])
    return pred


def fit_nuisance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, ...]:
    """mu_hat and sigma_hat at the calibration and test points, and the standardized residuals.

    Both networks are refitted on the whole training split, sigma_hat to the out-of-fold absolute
    residuals and scaled by c*. The training residuals are standardized by the out-of-fold mu_hat
    and sigma_hat, the calibration residuals by the refitted ones.
    """
    n = x_train.shape[0]
    n_folds = min(max(2, cfg.n_folds), n)
    folds = np.arange(n) % n_folds
    np.random.default_rng(SPLIT_SEED).shuffle(folds)

    mu_oof = cross_fit(x_train, y_train, folds, n_folds, cfg)
    abs_oof = np.abs(y_train - mu_oof)
    sig_oof_raw = np.maximum(cross_fit(x_train, abs_oof, folds, n_folds, cfg), 1e-6)
    # sigma_hat = c* sigma_raw, with c* the median of the out-of-fold |residual| / sigma_raw.
    c_star = np.median(abs_oof / sig_oof_raw)
    sig_oof = np.maximum(c_star * sig_oof_raw, 1e-6)

    mu_model = fit_median(x_train, y_train, cfg)
    mu_cal = predict_median(mu_model, x_cal)
    mu_test = predict_median(mu_model, x_test)

    # The final scale network is trained on the out-of-fold absolute residuals.
    sig_model = fit_median(x_train, abs_oof, cfg)
    sig_cal = np.maximum(c_star * np.maximum(predict_median(sig_model, x_cal), 1e-6), 1e-6)
    sig_test = np.maximum(c_star * np.maximum(predict_median(sig_model, x_test), 1e-6), 1e-6)

    r_train_oof = (y_train - mu_oof) / sig_oof
    r_cal = (y_cal - mu_cal) / sig_cal
    return mu_cal, mu_test, sig_cal, sig_test, r_train_oof, r_cal


def expand_range(y: np.ndarray) -> tuple[float, float]:
    """The range of y padded by 1% of its length on each side."""
    y = np.asarray(y, dtype=float)
    lo = float(np.nanmin(y))
    hi = float(np.nanmax(y))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi <= lo:
        return lo - 1.0, hi + 1.0
    pad = 0.01 * (hi - lo)
    return lo - pad, hi + pad


def fit_black_box(
    x: np.ndarray, y: np.ndarray, cfg: Config
) -> tuple[QNet, np.ndarray, float, float]:
    """CHR black box: the quantile network of y given x and a response grid on y's padded range."""
    y_min, y_max = expand_range(y)
    bbox = fit_qnet(x, y, quantile_grid(), cfg.qnet_epochs, cfg.qnet_calibrate, cfg)
    y_grid = np.linspace(y_min, y_max, num=cfg.y_steps, endpoint=True)
    return bbox, y_grid, y_min, y_max


def chr_level_grid(alpha: float, delta_alpha: float) -> np.ndarray:
    """Alpha grid of chr.grey_boxes.HistogramAccumulator; the CHR score levels are 1 - a on it."""
    alphas = np.round(np.arange(delta_alpha, 1.0, delta_alpha), 4)
    return np.unique(np.sort(np.concatenate((alphas, [alpha]))))


def chr_predict(chr_model: CHR, x: np.ndarray) -> np.ndarray:
    """CHR.predict, but a calibrated alpha below the alpha grid gives the whole response grid."""
    # There CHR.predict returns the empty band [ymax, ymax]; it still runs for its random draw.
    bands = chr_model.predict(x)
    alpha_grid = chr_level_grid(chr_model.alpha, chr_model.delta_alpha)
    if not np.any(alpha_grid <= chr_model.calibrated_alpha):
        lo = float(chr_model.grid_histogram[0])
        hi = float(chr_model.grid_histogram[-1])
        bands = np.tile(np.array([lo, hi], dtype=float), (x.shape[0], 1))
    return bands


def chr_scores(
    bbox: QNet,
    y_grid: np.ndarray,
    y_min: float,
    y_max: float,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Randomized CHR conformity scores of the points (x, y)."""
    q = bbox.predict(x.astype(np.float32))
    hist = Histogram(bbox.get_quantiles(), y_grid)
    pi = hist.compute_histogram(q, y_min, y_max, cfg.alpha)
    acc = HistogramAccumulator(pi, y_grid, cfg.alpha, delta_alpha=cfg.delta_alpha)
    eps = np.random.uniform(low=0.0, high=1.0, size=len(y))
    return np.asarray(
        acc.calibrate_intervals(y.astype(np.float32), epsilon=eps, verbose=False), dtype=float
    )


def chr_bands(
    bbox: QNet,
    y_grid: np.ndarray,
    y_min: float,
    y_max: float,
    x: np.ndarray,
    thresholds: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Bands of the CHR nested sets at per-point thresholds t: CHR's set at alpha = 1 - t.

    alpha is rounded to four decimals, as CHR rounds its own calibrated level, so that 1 - t lands
    on the grid value of the selected level.
    """
    q = bbox.predict(x.astype(np.float32))
    hist = Histogram(bbox.get_quantiles(), y_grid)
    pi = hist.compute_histogram(q, y_min, y_max, cfg.alpha)
    alpha_values = np.round(1.0 - np.asarray(thresholds, dtype=float), 4)
    alpha_values = np.where(np.isfinite(alpha_values), alpha_values, 0.0)
    alpha_values = np.clip(alpha_values, 0.0, 1.0)

    bands = np.empty((x.shape[0], 2), dtype=float)
    eps = np.random.uniform(low=0.0, high=1.0, size=x.shape[0])
    # Points with the same alpha share one accumulator; alpha <= 0 gives the whole grid.
    for alpha_i in np.unique(alpha_values):
        idx = np.where(alpha_values == alpha_i)[0]
        if alpha_i <= 0.0:
            bands[idx] = [y_grid[0], y_grid[-1]]
            continue
        acc = HistogramAccumulator(pi[idx], y_grid, cfg.alpha, delta_alpha=cfg.delta_alpha)
        _, band = acc.predict_intervals(float(alpha_i), epsilon=eps[idx])
        bands[idx] = band
    return bands


def localizer_coordinates(
    sig_cal: np.ndarray, sig_test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Localizer Z = log(sigma_hat) + 1e-9 U with U ~ U(0, 1), whose jitter breaks distance ties."""
    # U comes from its own stream of the repetition seed, so the global numpy stream is untouched.
    rng = np.random.default_rng([int(seed), JITTER_STREAM])
    sig_cal = np.asarray(sig_cal, dtype=float)
    sig_test = np.asarray(sig_test, dtype=float)
    z_cal = (np.log(np.maximum(sig_cal, 1e-8))
             + rng.uniform(size=sig_cal.shape[0]) * LOCALIZER_JITTER)
    z_test = (np.log(np.maximum(sig_test, 1e-8))
              + rng.uniform(size=sig_test.shape[0]) * LOCALIZER_JITTER)
    return z_cal, z_test


def n_neighbors(m: int) -> int:
    """k = max(20, round(sqrt(m))), at most m - 1."""
    return min(max(20, int(round(math.sqrt(m)))), max(1, m - 1))


def calibration_neighborhoods(z_cal: np.ndarray, k: int) -> tuple[list[np.ndarray], np.ndarray]:
    """The k nearest neighbors (nearest first) of each calibration point and the k-th distance."""
    m = len(z_cal)
    nbrs = []
    kth_dist = np.empty(m)
    for i in range(m):
        d = np.abs(z_cal - z_cal[i])
        d[i] = np.inf
        nbrs_i = np.argsort(d)[:k]
        nbrs.append(nbrs_i)
        kth_dist[i] = d[nbrs_i[-1]]
    return nbrs, kth_dist


def quantile_with(sorted_scores: np.ndarray, value: float, idx: int) -> float:
    """The element at zero-based index idx after adding value and sorting; clip idx to the range."""
    arr = np.sort(np.r_[sorted_scores, value])
    return float(arr[min(max(idx, 0), len(arr) - 1)])


def g2_thresholds(E: np.ndarray, z_cal: np.ndarray, z_test: np.ndarray, alpha: float) -> np.ndarray:
    """Threshold t* of Guan's G2 localized calibration at each test point (G2-Residual, RBC-CHR)."""
    m = len(E)
    k = n_neighbors(m)
    nbrs, kth_dist = calibration_neighborhoods(z_cal, k)
    out = np.empty(len(z_test), dtype=float)
    for j, z0 in enumerate(z_test):
        d_test = np.abs(z_cal - z0)
        E_nbrs_test = np.sort(E[np.argsort(d_test)[:k]])
        # The test point enters N_r when closer than the k-th neighbor of r, which it then replaces.
        in_nbhd = d_test < kth_dist
        base_sorted = []
        for r in range(m):
            nb = nbrs[r]
            if in_nbhd[r]:
                nb = nb[:-1]
            base_sorted.append(np.sort(np.r_[E[nb], E[r]]))

        # On the grid p = g / (m + 1) the quantile index ceil(p (k + 1)) - 1 takes only the values
        # 0, ..., k, so the search runs over the index; index k is the added top level T.
        t_star = np.inf
        for idx in range(k + 1):
            t = np.inf if idx >= k else float(E_nbrs_test[idx])
            if not np.isfinite(t):
                t_star = t
                break
            c1 = 0
            c2 = 0
            for r in range(m):
                if in_nbhd[r]:
                    v1 = quantile_with(base_sorted[r], t, idx)
                    v2 = quantile_with(base_sorted[r], 0.0, idx)
                else:
                    v1 = v2 = float(base_sorted[r][idx])
                c1 += E[r] <= v1
                c2 += E[r] <= v2
            if (c1 / (m + 1) >= 1 - alpha) and ((c2 + 1) / (m + 1) >= 1 - alpha):
                t_star = t
                break
        out[j] = t_star
    return out


def rlr_thresholds(
    E: np.ndarray,
    z_cal: np.ndarray,
    z_test: np.ndarray,
    alpha: float,
    delta_alpha: float = 0.001,
) -> np.ndarray:
    """Threshold e* of the residual local-rank calibration for each test point (Algorithm 1)."""
    m = len(E)
    k = n_neighbors(m)
    nbrs, kth_dist = calibration_neighborhoods(z_cal, k)
    # A_i: local rank of E_i among its calibration neighborhood; E_kth: score of its k-th neighbor.
    A = np.empty(m, dtype=int)
    E_kth = np.empty(m)
    for i in range(m):
        E_kth[i] = E[nbrs[i][-1]]
        A[i] = 1 + int(np.sum(E[nbrs[i]] <= E[i]))

    # Candidate levels e, scanned from the largest: 1 - a for every a on CHR's alpha grid, the
    # calibration scores E and 1.
    levels = 1.0 - chr_level_grid(alpha, delta_alpha)
    levels = np.unique(np.r_[levels, E, 1.0])[::-1]
    out = np.empty(len(z_test), dtype=float)
    for j, z0 in enumerate(z_test):
        d_test = np.abs(z_cal - z0)
        in_nbhd = d_test < kth_dist
        E_nbrs_test = np.sort(E[np.argsort(d_test)[:k]])
        # When the test point enters N_i it replaces the k-th neighbor, so A_i loses that neighbor's
        # term; A_base is A_i(e) without the test point's own term 1{e <= E_i}.
        dropped = in_nbhd.astype(int) * (E_kth <= E)
        A_base = (A - dropped).astype(int)
        # A_i(e) >= a holds when A_base_i >= a, or when i has the test point as a neighbor,
        # A_base_i = a - 1 and e <= E_i.
        n_ge = np.zeros(k + 3, dtype=int)
        E_tied: list[np.ndarray] = [np.empty(0)] * (k + 3)
        for a in range(1, k + 2):
            n_ge[a] = int(np.sum(A_base >= a))
            E_tied[a] = np.sort(E[in_nbhd & (A_base == a - 1)])
        e_star = levels[-1]
        for e in levels:
            A_test = 1 + int(np.searchsorted(E_nbrs_test, e, side="right"))
            n_tied = int(len(E_tied[A_test]) - np.searchsorted(E_tied[A_test], e, side="left"))
            p = (1 + n_ge[A_test] + n_tied) / (m + 1)
            if p > alpha:
                e_star = float(e)
                break
        out[j] = e_star
    return out


def run_one(
    rep: int, n_data: int, cfg: Config, dgp: str, sample: Callable, metrics: Callable
) -> list[dict[str, object]]:
    """One repetition of the methods in cfg.methods on data drawn by sample; one row per method."""
    seed = SEED_BASE + rep
    x_train, y_train, x_cal, y_cal, x_test, y_test = make_split(n_data, cfg.n_test, seed, sample)
    rows: list[dict[str, object]] = []
    methods = set(cfg.methods)

    def add(method: str, lo: np.ndarray, hi: np.ndarray) -> None:
        rows.append({"rep": rep, "dgp": dgp, "n_data": n_data, "method": method,
                     **metrics(lo, hi, y_test, x_test)})

    # CHR runs first, before any nuisance model, so that its random draws come first.
    if "CHR" in methods:
        bbox_y, y_grid, y_min, y_max = fit_black_box(x_train, y_train, cfg)
        chr_y = CHR(bbox_y, ymin=y_min, ymax=y_max, y_steps=cfg.y_steps,
                    delta_alpha=cfg.delta_alpha, randomize=True)
        chr_y.calibrate(x_cal, y_cal, cfg.alpha)
        bands = chr_predict(chr_y, x_test)
        add("CHR", bands[:, 0], bands[:, 1])

    if not methods & {"Norm-SC", "MCR", "G2-Residual", "RBC-CHR", "RLR-CHR"}:
        return rows

    mu_cal, mu_test, sig_cal, sig_test, r_train, r_cal = fit_nuisance(
        x_train, y_train, x_cal, y_cal, x_test, cfg
    )
    # One localizer per repetition, shared by G2-Residual, RBC-CHR and RLR-CHR.
    z_cal, z_test = localizer_coordinates(sig_cal, sig_test, seed)

    if "Norm-SC" in methods:
        q = conformal_quantile(np.abs(y_cal - mu_cal) / np.maximum(sig_cal, 1e-8), cfg.alpha)
        add("Norm-SC", mu_test - q * sig_test, mu_test + q * sig_test)

    if "MCR" in methods:
        # crepes Mondrian conformal regression; the intervals are centered at mu_hat. The Mondrian
        # bins come from a k-nearest-neighbor difficulty estimate (k = 25) of the in-sample absolute
        # residuals of mu_mcr, a second median network with mu_hat's specification, which is fitted
        # so that the random streams match the paper's runs.
        de = DifficultyEstimator()
        mu_mcr = fit_median(x_train, y_train, cfg)
        de.fit(X=x_train.astype(float), residuals=np.abs(y_train - predict_median(mu_mcr, x_train)),
               k=25)
        mc = MondrianCategorizer()
        mc.fit(X=x_cal.astype(float), de=de, no_bins=mcr_n_bins(len(y_cal), cfg.alpha))
        bins_cal = mc.apply(x_cal.astype(float))
        bins_test = mc.apply(x_test.astype(float))
        cr = crepes.ConformalRegressor()
        cr.fit(residuals=(y_cal - mu_cal).astype(float), bins=bins_cal)
        intervals = np.asarray(
            cr.predict_int(y_hat=mu_test.astype(float), bins=bins_test, confidence=1 - cfg.alpha),
            dtype=float,
        )
        add("MCR", intervals[:, 0], intervals[:, 1])

    if "G2-Residual" in methods:
        t = g2_thresholds(np.abs(r_cal), z_cal, z_test, cfg.alpha)
        add("G2-Residual", mu_test - t * sig_test, mu_test + t * sig_test)

    if methods & {"RBC-CHR", "RLR-CHR"}:
        # CHR scores E of the standardized residuals (y - mu_hat) / sigma_hat.
        bbox_r, r_grid, r_min, r_max = fit_black_box(x_train, r_train, cfg)
        E_cal = chr_scores(bbox_r, r_grid, r_min, r_max, x_cal, r_cal, cfg)

        if "RBC-CHR" in methods:
            t = g2_thresholds(E_cal, z_cal, z_test, cfg.alpha)
            bands = chr_bands(bbox_r, r_grid, r_min, r_max, x_test, t, cfg)
            add("RBC-CHR", mu_test + sig_test * bands[:, 0], mu_test + sig_test * bands[:, 1])

        if "RLR-CHR" in methods:
            e = rlr_thresholds(E_cal, z_cal, z_test, cfg.alpha, delta_alpha=cfg.delta_alpha)
            bands = chr_bands(bbox_r, r_grid, r_min, r_max, x_test, e, cfg)
            add("RLR-CHR", mu_test + sig_test * bands[:, 0], mu_test + sig_test * bands[:, 1])

    return rows


def run_all(
    sizes: list[int],
    reps: Iterable[int],
    cfg: Config,
    n_workers: int,
    dgp: str,
    sample: Callable,
    metrics: Callable,
) -> list[dict[str, object]]:
    """run_one for every n and repetition, on n_workers processes; rows in (n, rep) order."""
    tasks = [(rep, n) for n in sizes for rep in reps]
    results = {}
    if n_workers <= 1:
        for rep, n in tasks:
            results[rep, n] = run_one(rep, n, cfg, dgp, sample, metrics)
            print(f"n={n} rep={rep} done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(run_one, rep, n, cfg, dgp, sample, metrics): (rep, n)
                    for rep, n in tasks}
            for done, fut in enumerate(as_completed(futs), 1):
                rep, n = futs[fut]
                results[rep, n] = fut.result()
                print(f"[{done}/{len(tasks)}] n={n} rep={rep} done", flush=True)
    return [row for task in tasks for row in results[task]]


def main() -> None:
    cfg, args = parse_args()
    rows = run_all(args.sample_sizes, args.reps, cfg, args.n_workers, "hetero_skewed",
                   sample_hetero_skewed, interval_metrics)
    df = pd.DataFrame(rows)
    grid_dir = Path(args.out) / "synthetic" / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    # One file per sample size, named by its first and last repetition; aggregate_synthetic.py
    # reads the whole directory.
    chunk = f"reps{args.reps[0]}-{args.reps[-1]}"
    for n in args.sample_sizes:
        out = grid_dir / f"n{n}_{chunk}.csv"
        df[df["n_data"] == n].to_csv(out, index=False)
        print(f"wrote {out}")
    print(df.groupby(["n_data", "method"])[["coverage", "length", "wsc"]].mean().to_string())


if __name__ == "__main__":
    main()
