#!/usr/bin/env bash
# Synthetic study and WSC calibration simulation; clones and patches CHR when it is missing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

commit=4f02607f2c7f35237b4e042132db9a6404294b04
patch="$PWD/experiments/synthetic/chr_imports.patch"
if [ ! -d third_party/chr ]; then
  git clone https://github.com/msesia/chr.git third_party/chr
  git -C third_party/chr checkout "$commit"
  git -C third_party/chr apply "$patch"
fi
if [ "$(git -C third_party/chr rev-parse HEAD 2> /dev/null)" != "$commit" ] ||
   ! git -C third_party/chr apply --check -R "$patch" 2> /dev/null; then
  echo "third_party/chr is not commit $commit with experiments/synthetic/chr_imports.patch applied;" \
       "remove it and run this script again" >&2
  exit 1
fi
# Repetitions 1-100 (seeds 1001-1100) of each sample size in chunks of 8, written to
# results/synthetic/grid/n<n>_reps<a>-<b>.csv.
for n in 100 200 300 500 1000 2000 5000 8000; do
  for a in $(seq 1 8 100); do
    b=$(( a + 7 < 100 ? a + 7 : 100 ))
    python3 experiments/synthetic/run_synthetic.py --sample_sizes="$n" --reps="$a:$b" --n_workers=3
  done
done
python3 experiments/synthetic/aggregate_synthetic.py
python3 experiments/synthetic/run_skew.py --sample_sizes=500,1000,3000 --reps=1:100 --n_workers=3
python3 experiments/wsc_calibration_sim.py
