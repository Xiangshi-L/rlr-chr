#!/usr/bin/env bash
# RLCP on the splits of the main study, 100 repetitions each; clones the authors' code when it is missing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

commit=89dd18becb1d5120534cbdc92da18487a65fbf95
if [ ! -d third_party/RLCP ]; then
  git clone https://github.com/rohanhore/RLCP.git third_party/RLCP
  git -C third_party/RLCP checkout "$commit"
fi
if [ "$(git -C third_party/RLCP rev-parse HEAD 2> /dev/null)" != "$commit" ]; then
  echo "third_party/RLCP is not at commit $commit; remove it and run this script again" >&2
  exit 1
fi
Rscript experiments/run_rlcp.R --reps=1:100 --n_workers=5 --alpha=0.10 --n_test=2000 \
  --rlcp_dir=third_party/RLCP
