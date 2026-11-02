#!/usr/bin/env bash
# Coverage by covariate density on the eight main datasets: 5 strata of the distance to the 20th
# nearest training point, 100 repetitions each.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

Rscript experiments/run_density.R --reps=1:100 --n_workers=6 --alpha=0.10 --n_test=2000 --bins=1000 \
  --levels=1000 --k=20 --n_strata=5
