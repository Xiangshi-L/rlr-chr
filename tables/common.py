"""Shared by the generators: flags, datasets, cells over repetitions, formatting, figure style."""
import argparse
import csv
import logging
import math
import os
import statistics as st
import sys
from collections import defaultdict
from decimal import Decimal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
METHODS = ["Split-CP", "Norm-SC", "CQR", "CHR", "MCR", "G2-Residual", "RBC-CHR", "RLR-CHR"]
DAGGER = "$^\\dagger$"


def read_manifest(name):
    """Keys and display names of data/<name>_manifest.csv, in table order."""
    with open(os.path.join(REPO, "data", f"{name}_manifest.csv"), newline="") as fh:
        return [(r["key"], r["name"]) for r in csv.DictReader(fh)]


# Datasets as (key, display name) in table order; additional datasets are ordered by share of zero responses.
MAIN_DATASETS = read_manifest("main")
ADDITIONAL_DATASETS = read_manifest("additional")
NAMES = dict(MAIN_DATASETS + ADDITIONAL_DATASETS)
# The two datasets of the alpha = 0.05 and alpha = 0.20 studies.
ALPHA_DATASETS = [(key, NAMES[key]) for key in ("sgemm", "bike")]


def parse_args(doc, results=True, data=False):
    """Parse --out and the enabled --results and --data flags."""
    ap = argparse.ArgumentParser(description=doc)
    if results:
        ap.add_argument("--results", default=os.path.join(REPO, "results"), metavar="DIR",
                        help="results root (default: results/ in the repository)")
    ap.add_argument("--out", default=os.path.join(REPO, "generated"), metavar="DIR",
                    help="output directory (default: generated/ in the repository)")
    if data:
        ap.add_argument("--data", default=os.path.join(REPO, "data", "processed"), metavar="DIR",
                        help="processed datasets (default: data/processed/ in the repository)")
    return ap.parse_args()


def num(x):
    """A result-file cell as a float; NA, NaN, empty and missing cells are nan."""
    return float("nan") if x is None or str(x).strip() in ("NA", "NaN", "") else float(x)


# Statistics over repetitions: exact means (rounded once) and sample SDs with divisor n - 1.
mean = st.mean
median = st.median


def sd(v):
    """Sample standard deviation, the square root of the sample variance."""
    return math.sqrt(st.variance(v))


def se(v):
    """Standard error of the mean."""
    v = list(v)
    return sd(v) / math.sqrt(len(v))


# A repetition is degenerate when its mean length is non-finite or more than DIV_RATIO
# times CHR's in the same repetition, or, for CQR, when more than EMPTY_CQR_MAX of its
# test intervals are empty. A cell with degenerate repetitions reports its statistics
# over the other repetitions and carries a dagger.
DIV_RATIO = 5.0
EMPTY_CQR_MAX = 0.01
# Share of test points calibrated at the top level T. A repetition with share 1 and
# coverage 1 returned the whole support at every test point.
TOP_LEVEL_COL = {"RLR-CHR": "frac_top_level_rlr", "RBC-CHR": "frac_top_level_rbc"}
# Result-file columns of WSC at slab mass 0.10 (main runs) and 0.05, 0.10, 0.20 (additional runs).
MAIN_WSC = {"wsc": "wsc"}
ADDITIONAL_WSC = {"wsc005": "wsc_005", "wsc": "wsc", "wsc020": "wsc_020"}
WSC_LABEL = {"wsc": "WSC", "wsc005": "WSC(0.05)", "wsc020": "WSC(0.20)"}


def is_degenerate(method, length, chr_length, frac_empty=float("nan")):
    """Whether one repetition of a method is degenerate (rule above)."""
    if not math.isfinite(length) or length > DIV_RATIO * chr_length:
        return True
    return method == "CQR" and math.isfinite(frac_empty) and frac_empty > EMPTY_CQR_MAX


def load_cells(path, wsc_cols, alpha=None):
    """Cells {method: cell} of one result file and its rows {method: {rep: row}}."""
    rows = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            method, rep = r["method"], int(float(r["rep"]))
            if rep in rows[method]:
                sys.exit(f"{path}: more than one row for ({method}, rep {rep})")
            if alpha is not None and not math.isclose(num(r["alpha"]), alpha):
                sys.exit(f"{path}: row ({method}, rep {rep}) has alpha {r['alpha']}, "
                         f"expected {alpha}")
            rows[method][rep] = r
    if set(rows) != set(METHODS):
        sys.exit(f"{path}: methods {sorted(rows)}, expected {sorted(METHODS)}")
    reps = sorted(rows["CHR"])
    if any(sorted(rows[method]) != reps for method in METHODS):
        sys.exit(f"{path}: the methods do not share the same repetitions")
    chr_len = {r: num(rows["CHR"][r]["length"]) for r in reps}
    if not all(math.isfinite(v) and v > 0 for v in chr_len.values()):
        sys.exit(f"{path}: a CHR length is not finite and positive")
    cells = {method: method_cell(method, rows[method], chr_len, wsc_cols, path)
             for method in METHODS}
    return cells, dict(rows)


def method_cell(method, rows, chr_len, wsc_cols, where):
    """Statistics of one method over its non-degenerate repetitions and over all repetitions."""
    reps = sorted(rows)
    cols = {**wsc_cols, "cov": "coverage", "len": "length"}
    values = {s: {r: num(rows[r][col]) for r in reps} for s, col in cols.items()}
    if any(math.isnan(values[s][r]) for s in [*wsc_cols, "cov"] for r in reps):
        sys.exit(f"{where}: {method} lacks WSC or coverage in a repetition")
    # Length relative to CHR's in the same repetition.
    values["rel"] = {r: values["len"][r] / chr_len[r] for r in reps}
    bad = [r for r in reps if is_degenerate(method, values["len"][r], chr_len[r],
                                            num(rows[r]["frac_empty_cqr"]))]
    keep = [r for r in reps if r not in bad]
    if len(keep) < 2:
        sys.exit(f"{where}: {method} has fewer than two non-degenerate repetitions")
    top = TOP_LEVEL_COL.get(method)
    whole = sum(1 for r in bad if top and num(rows[r][top]) == 1 and values["cov"][r] == 1)
    cell = {"bad": bad, "keep": keep, "whole": whole}
    for s, v in values.items():
        kept = [v[r] for r in keep]
        if not all(math.isfinite(x) for x in kept):
            sys.exit(f"{where}: {method} has a non-finite {s} in a non-degenerate repetition")
        cell[s], cell[s + "_se"] = mean(kept), se(kept)
        cell[s + "_all"] = mean(v[r] for r in reps)
    cell["len_sd"] = sd(values["len"][r] for r in keep)
    cell["len_sd_all"] = sd(values["len"][r] for r in reps)
    defined = [x for x in values["rel"].values() if not math.isnan(x)]
    cell["rel_med"] = median(defined) if defined else float("nan")
    return cell


def load_rlcp(path, chr_rows):
    """RLCP cell of one dataset, its lengths relative to CHR's in the same repetition."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    reps = [int(float(r["rep"])) for r in rows]
    if {r["method"] for r in rows} != {"RLCP"} or sorted(reps) != sorted(chr_rows):
        sys.exit(f"{path}: RLCP rows are not one per repetition of the main run")
    # RLCP intervals can be unbounded, so its length is the authors' width convention
    # (twice the median calibrated threshold).
    cols = {"wsc": "wsc", "cov": "coverage", "len": "median_length",
            "time": "inference_time", "frac_inf": "frac_inf_intervals"}
    v = {s: [num(r[col]) for r in rows] for s, col in cols.items()}
    rel = [width / num(chr_rows[rep]["length"]) for width, rep in zip(v["len"], reps)]
    return {"wsc": mean(v["wsc"]), "wsc_se": se(v["wsc"]), "cov": mean(v["cov"]),
            "cov_se": se(v["cov"]), "rel": mean(rel), "rel_se": se(rel), "len": mean(v["len"]),
            "len_sd": sd(v["len"]), "time": mean(v["time"]), "frac_inf": mean(v["frac_inf"])}


def load_main(results):
    """Cells, RLCP cells and rows of the eight main datasets, each a dict by dataset key."""
    cells, rlcp, rows = {}, {}, {}
    for key, _ in MAIN_DATASETS:
        cells[key], rows[key] = load_cells(os.path.join(results, "main", f"{key}_long.csv"),
                                           MAIN_WSC)
        rlcp[key] = load_rlcp(os.path.join(results, "rlcp", f"{key}_long.csv"), rows[key]["CHR"])
    return cells, rlcp, rows


def print_degenerate(cells, datasets):
    """Each cell with degenerate repetitions, its statistics without them and over all of them."""
    for key, name in datasets:
        for method in METHODS:
            cell = cells[key][method]
            if not cell["bad"]:
                continue
            n = len(cell["bad"]) + len(cell["keep"])
            wsc = [(label, s) for s, label in WSC_LABEL.items() if s in cell]
            print(f"{name}, {method}: {len(cell['bad'])} of {n} repetitions degenerate "
                  f"{cell['bad']} ({cell['whole']} returned the whole support)")
            print("  without them: " + ", ".join(f"{label} {cell[s]:.4f}" for label, s in wsc)
                  + f", coverage {cell['cov']:.4f}, rel. length {cell['rel']:.4f}")
            print(f"  all {n}: " + ", ".join(f"{label} {cell[s + '_all']:.4f}" for label, s in wsc)
                  + f", coverage {cell['cov_all']:.4f}, length {cell['len_all']:.4g} "
                  f"(SD {cell['len_sd_all']:.4g}), rel. length {cell['rel_all']:.4g} "
                  f"(median {cell['rel_med']:.4f})")


def print_share_above_T(files, label):
    """Print the largest share of calibration scores above level T over the files {name: path}."""
    best, lacking = None, []
    for name, path in files.items():
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if "frac_cal_score_gt_T" not in reader.fieldnames:
                lacking.append(name)
                continue
            for r in reader:
                v = num(r["frac_cal_score_gt_T"])
                if not math.isnan(v) and (best is None or v > best[0]):
                    best = (v, name, int(float(r["rep"])))
    if lacking:
        print(f"{label}: no column frac_cal_score_gt_T in " + ", ".join(lacking))
    if best:
        print(f"{label}: largest share of calibration scores above level T, {100 * best[0]:.3g}% "
              f"({best[1]}, repetition {best[2]})")


def fmt_se(v, nd=3):
    """A standard error without its leading zero."""
    t = f"{v:.{nd}f}"
    return t[1:] if t.startswith("0.") else t


def mean_with_se(v):
    """'mean (SE)' of the values v at three decimals."""
    v = list(v)
    return f"{mean(v):.3f} ({fmt_se(se(v))})"


def closest(values, target, nd=3):
    """Keys whose value, printed at nd decimals, is closest to target (ties included)."""
    t = Decimal(str(target))
    dist = {key: abs(Decimal(f"{v:.{nd}f}") - t) for key, v in values.items()}
    best = min(dist.values())
    return {key for key, d in dist.items() if d == best}


def shortest(values, nd=3):
    """Keys with the smallest value at nd decimals (ties included)."""
    best = f"{min(values.values()):.{nd}f}"
    return {key for key, v in values.items() if f"{v:.{nd}f}" == best}


def pair_cell(value, std_err, bold=False, dagger=False):
    """A mean with its gray SE and a fixed-width slot for the dagger (Tables 1 and 11)."""
    value = f"{value:.3f}"
    if bold:
        value = f"\\textbf{{{value}}}"
    mark = "\\makebox[2.5pt][l]{" + DAGGER + "}" if dagger else "\\hspace{2.5pt}"
    return ("{\\footnotesize " + value + "}\\,"
            "{\\scriptsize\\color{black!70}(" + fmt_se(std_err) + ")}" + mark)


def cell_tex(cell, stat, bold=False):
    """'mean (SE)', bold if chosen, with a dagger when the cell has degenerate repetitions."""
    text = f"{cell[stat]:.3f} ({fmt_se(cell[stat + '_se'])})"
    if bold:
        text = f"\\textbf{{{text}}}"
    return text + (DAGGER if cell["bad"] else "")


BLOCK = 4   # datasets per block of Tables 1 and 11


def paired_table(cells, datasets, zeros=None, rlcp=None):
    """WSC and relative length per dataset in blocks, with a Zeros row or an RLCP row."""
    out = ["\\scriptsize", "\\setlength{\\tabcolsep}{1.5pt}",
           "\\begin{tabular*}{\\textwidth}{@{}l@{\\extracolsep{\\fill}}"
           f"*{{{BLOCK}}}{{rr}}@{{}}}}",
           "\\toprule"]
    for start in range(0, len(datasets), BLOCK):
        block = datasets[start:start + BLOCK]
        if start:
            out.append("\\midrule[\\heavyrulewidth]")
        out += ["".join(f" & \\multicolumn{{2}}{{c}}{{\\footnotesize\\textbf{{{name}}}}}"
                        for _, name in block).lstrip() + " \\\\",
                " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(block)))]
        if zeros:
            out.append("{\\footnotesize Zeros (\\%)}"
                       + "".join(" & \\multicolumn{2}{c}{\\footnotesize " + zeros[key] + "}"
                                 for key, _ in block) + " \\\\")
        out += ["{\\footnotesize\\textbf{Method}}"
                + " & \\multicolumn{1}{c}{WSC} & \\multicolumn{1}{c}{Rel.\\ length}" * len(block)
                + " \\\\",
                "\\midrule"]
        # Bold: the WSC closest to 0.9 (RLCP included) and the shortest relative length.
        bold_wsc, bold_len = {}, {}
        for key, _ in block:
            wsc = {method: cells[key][method]["wsc"] for method in METHODS}
            if rlcp:
                wsc["RLCP"] = rlcp[key]["wsc"]
            bold_wsc[key] = closest(wsc, 0.9)
            bold_len[key] = shortest({method: cells[key][method]["rel"] for method in METHODS})
        for method in METHODS:
            row = []
            for key, _ in block:
                cell = cells[key][method]
                dagger = bool(cell["bad"])
                row += [pair_cell(cell["wsc"], cell["wsc_se"], method in bold_wsc[key], dagger),
                        pair_cell(cell["rel"], cell["rel_se"], method in bold_len[key], dagger)]
            out.append("{\\footnotesize " + method + "} & " + " & ".join(row) + " \\\\")
        if rlcp:
            # The dagger marks RLCP's width convention (twice the median calibrated threshold).
            row = []
            for key, _ in block:
                r = rlcp[key]
                row += [pair_cell(r["wsc"], r["wsc_se"], bold="RLCP" in bold_wsc[key]),
                        pair_cell(r["rel"], r["rel_se"], dagger=True)]
            out += ["\\midrule", "{\\footnotesize RLCP} & " + " & ".join(row) + " \\\\"]
    out += ["\\bottomrule", "\\end{tabular*}"]
    return out


# Figure text: serif TrueType (Type 42) fonts, close to the Times text of the paper.
FIGURE_STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
}
# Figures 1 and 4: 8-point text, thin axes and outward ticks, no top or right spine.
SMALL_FIGURE_STYLE = {
    **FIGURE_STYLE,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
}
# Figures 2 and 3, drawn wider: 10- and 11-point text.
WIDE_FIGURE_STYLE = {
    **FIGURE_STYLE,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 0.9,
}


def save_figure(fig, path, **kwargs):
    """Save a PDF figure without CreationDate metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Embedding the fonts makes fontTools warn about their timestamps; the figure is unaffected.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    fig.savefig(path, metadata={"CreationDate": None}, **kwargs)
    print(f"wrote {path}")
