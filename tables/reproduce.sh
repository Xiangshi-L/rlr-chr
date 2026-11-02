#!/usr/bin/env bash
# Table bodies, figures and the diagnostics summary of the paper, from the result files.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
results="$root/results"
out="$root/generated"
data="$root/data/processed"
for arg in "$@"; do
  case "$arg" in
    --results=*) results="${arg#*=}" ;;
    --out=*) out="${arg#*=}" ;;
    --data=*) data="${arg#*=}" ;;
    *)
      echo "unknown argument $arg (expected --results=DIR, --out=DIR or --data=DIR)" >&2
      exit 2
      ;;
  esac
done

# In the paper's order; only make_additional_tables.py reads the datasets (for its Zeros row).
tables="$root/tables"
python3 "$tables/make_fig1.py" --results="$results" --out="$out"
python3 "$tables/make_main_table.py" --results="$results" --out="$out"
python3 "$tables/make_appendix_tables.py" --results="$results" --out="$out"
python3 "$tables/make_additional_tables.py" --results="$results" --out="$out" --data="$data"
python3 "$tables/make_density_table.py" --results="$results" --out="$out"
python3 "$tables/make_diag_figures.py" --results="$results" --out="$out"
python3 "$tables/make_ablation_tables.py" --results="$results" --out="$out"
python3 "$tables/make_alpha_table.py" --results="$results" --out="$out"
python3 "$tables/make_skew_table.py" --results="$results" --out="$out"
python3 "$tables/make_fig_worked_example.py" --out="$out"
echo "Table bodies, figures and diagnostics_summary.csv: $out"
