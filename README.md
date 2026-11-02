# RLR-CHR

Code for *Beyond Marginal Coverage: Efficient Localized Conformal Prediction via Residual Rank
Calibration*, by Xiangshi Li and Wenqing He (NeurIPS 2026).

RLR-CHR applies conformal histogram regression to standardized residuals and calibrates the
resulting scores through their local ranks.

The repository contains

- `rlrchr/`: the R implementation, C++ core, and main experiment driver.
- `experiments/`: scripts and launchers for the remaining studies.
- `tables/`: scripts to generate the paper's tables and figures from result files.
- `data/`: data download and preprocessing; see [data/README.md](data/README.md).

## Installation

You need Git, R and a C++ compiler. We used R 4.5.3 with the R packages Rcpp 1.1.1, keras3 1.5.1,
tensorflow 2.20.0 and reticulate 1.45.0. keras3 calls the Python TensorFlow backend through
reticulate. For the R experiments, create `r-keras` (Python 3.10-3.12) and select it with
`RETICULATE_PYTHON`:

```bash
Rscript -e 'install.packages(c("Rcpp", "keras3", "tensorflow", "reticulate"), repos = "https://cloud.r-project.org")'
python3.10 -m venv ~/.virtualenvs/r-keras
~/.virtualenvs/r-keras/bin/pip install tensorflow==2.19.0 keras==3.11.1 numpy==1.26.4 crepes==0.9.0 \
  scikit-learn==1.7.2 pillow==11.3.0 pydot==4.0.1 ipython==8.37.0
export RETICULATE_PYTHON=~/.virtualenvs/r-keras/bin/python
```

Data preparation, the synthetic experiments in PyTorch and the table scripts use `.venv`
(Python 3.9-3.12). Keeping it separate preserves their different dependency versions:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

Run the commands below from the repository root with `.venv` active and `RETICULATE_PYTHON` set.

## Running one repetition

```bash
python3 data/prepare_data.py --datasets=naval_propulsion_comp_decay
Rscript rlrchr/rlr_chr.R --datasets=naval_propulsion_comp_decay --reps=1:1 --n_workers=1 \
  --out=results/quickstart
```

This takes about five minutes on one core and writes
`results/quickstart/main/naval_propulsion_comp_decay_long.csv`. Flags use `--name=value`;
`--help` lists the options and their defaults.

## Reproducing the paper

Prepare the data first:

```bash
python3 data/prepare_data.py
```

The script downloads twelve datasets into `data/processed/`.
[data/README.md](data/README.md) gives the steps for Bike Sharing and the three MEPS panels.
Prepare all sixteen datasets before running the full reproduction below.

Each launcher in `experiments/launchers/` runs one study with the paper's settings and 100
repetitions and writes under `results/`; `tables/reproduce.sh` then builds every table body and
figure into `generated/` and prints the numerical summaries quoted in the text:

```bash
for study in main rlcp additional density diagnostics ablations synthetic; do
  bash "experiments/launchers/$study.sh"
done
bash tables/reproduce.sh
```

The launchers use three to seven workers; the paper's appendix on computational cost reports
hardware and runtimes. `rlcp.sh` and `synthetic.sh` fetch RLCP and CHR at pinned commits into
`third_party/`. To reproduce one item, use its launcher and generator:

| Paper item | Launcher (`experiments/launchers/`) | Generator (`tables/`) |
|---|---|---|
| Figure 1 | `synthetic.sh` | `make_fig1.py` |
| Table 1 | `main.sh`, `rlcp.sh` | `make_main_table.py` |
| Tables 3-10 | `main.sh`, `rlcp.sh` | `make_appendix_tables.py` |
| Tables 11-12 | `additional.sh` | `make_additional_tables.py` |
| Table 13 | `density.sh` | `make_density_table.py` |
| Figures 2-3 | `diagnostics.sh` | `make_diag_figures.py` |
| Tables 14-16 | `ablations.sh` | `make_ablation_tables.py` |
| Table 17 | `ablations.sh` | `make_alpha_table.py` |
| Table 18 | `synthetic.sh` | `make_skew_table.py` |
| Figure 4, Table 20 | none | `make_fig_worked_example.py` |
| Table 22 | `synthetic.sh` | none; the launcher writes `results/wsc_calibration.csv` |

All released experiments use fixed seeds. Some real data results may differ from those reported
in the paper because the original runs did not fully seed network initialization.

## License

MIT; see [LICENSE](LICENSE). The loaders for the additional datasets in
`data/prepare_data.py` follow `datasets.py` of [CQR](https://github.com/yromano/cqr) and
[SLCP](https://github.com/aaronhan223/SLCP); their licenses are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
