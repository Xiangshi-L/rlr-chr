#!/usr/bin/env bash
# Main study: the eight methods on the eight datasets of data/main_manifest.csv, 100 repetitions each.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

Rscript rlrchr/rlr_chr.R --reps=1:100 --n_workers=7 --alpha=0.10 --n_test=2000 --bins=1000 --levels=1000
