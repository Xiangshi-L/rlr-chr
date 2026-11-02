#!/usr/bin/env bash
# Sensitivity studies, 100 repetitions each: bins on BlogFeedback, levels T on SGEMM, and alpha = 0.05
# (CHR, RLR-CHR) and alpha = 0.20 (all eight methods) on SGEMM and Bike Sharing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

Rscript experiments/run_bins_ablation.R --datasets=blog --reps=1:100 --n_workers=4 --alpha=0.10 \
  --n_test=2000 --bins=100,1000 --levels=1000
Rscript experiments/run_t_ablation.R --datasets=sgemm --reps=1:100 --n_workers=4 --alpha=0.10 \
  --n_test=2000 --bins=1000 --levels=25,50,100,200,1000
Rscript experiments/run_bins_ablation.R --datasets=sgemm,bike --reps=1:100 --n_workers=4 --alpha=0.05 \
  --n_test=2000 --bins=1000 --levels=1000
Rscript experiments/run_additional.R --datasets=sgemm,bike --reps=1:100 --n_workers=4 --alpha=0.20 \
  --n_test=2000 --bins=1000 --levels=1000
