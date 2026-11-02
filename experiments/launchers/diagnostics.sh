#!/usr/bin/env bash
# Diagnostics of RLR-CHR on SGEMM, 100 repetitions in chunks of 12.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

for a in $(seq 1 12 100); do
  b=$(( a + 11 < 100 ? a + 11 : 100 ))
  Rscript experiments/run_sgemm_diagnostics.R --datasets=sgemm --reps="$a:$b" --n_workers=6 --alpha=0.10 \
    --n_test=2000 --bins=1000 --levels=1000 --multipliers=0.5,0.75,1,1.25,1.5,1.75,2
done
