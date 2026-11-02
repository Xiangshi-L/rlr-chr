#!/usr/bin/env python3
"""Download and preprocess the datasets of the paper into data/processed/<key>.csv."""

# The builders for the additional datasets follow datasets.py of github.com/yromano/cqr (MIT license)
# and its copy in github.com/aaronhan223/SLCP (BSD-3-Clause license); THIRD_PARTY_NOTICES
# reproduces both licenses.

import argparse
import hashlib
import http.client
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UCI = "https://archive.ics.uci.edu/static/public"
OPENML = "https://www.openml.org/data/v1/download"
GPML = "https://gaussianprocess.org/gpml/data"

SOURCES = {
    "sgemm.zip": f"{UCI}/440/sgemm+gpu+kernel+performance.zip",
    "tetouan.zip": f"{UCI}/849/power+consumption+of+tetouan+city.zip",
    "parking.zip": f"{UCI}/482/parking+birmingham.zip",
    "video.zip": f"{UCI}/335/online+video+characteristics+and+transcoding+time+dataset.zip",
    "naval.zip": f"{UCI}/316/condition+based+maintenance+of+naval+propulsion+plants.zip",
    "casp.zip": f"{UCI}/265/physicochemical+properties+of+protein+tertiary+structure.zip",
    "facebook.zip": f"{UCI}/363/facebook+comment+volume+dataset.zip",
    "blog.zip": f"{UCI}/304/blogfeedback.zip",
    "mv.arff": f"{OPENML}/52247/mv.arff",
    "french_load_consumption.arff": f"{OPENML}/22120916/french_load_consumption.arff",
    "sarcos_inv.mat": f"{GPML}/sarcos_inv.mat",
    "sarcos_inv_test.mat": f"{GPML}/sarcos_inv_test.mat",
}

# MD5 sums of the processed files used in the paper. The builders keep the column names, column
# order and number formats of those files, so that the sums match; the R scripts read the response
# by its name `target`.
MD5 = {
    "naval_propulsion_comp_decay": "905a90c54fc549535451e875dd303e4a",
    "mv": "555a103ad2efdf28387fc2a4d0364f1e",
    "parking_birmingham": "a92df324098be03caea954460fe28294",
    "sarcos_torque1": "50a43148e37ac2b847d71ea2260e11a0",
    "sgemm": "2793e9a595a6acc5c04c0bdba6119be9",
    "tetouan_power_zone1": "5efe1b0ac08bb8bf09a0a8c414466d8b",
    "video_transcoding_utime_full": "41cd08f17acf50b357aae67dd4fb93b9",
    "french_load_consumption": "0a5ab70f3e82f8170a9f13a2525c747d",
    "bike": "fd5faeb06ee7995e8e98f02fa8c06a28",
    "blog": "45a77a89b56a21daad6ca7618d55d5c4",
    "casp_protein": "6467206f17f3701bba4f22ccafdd5715",
    "facebook_comments": "ea1979ec19457bfe20426af7cd8a17cb",
    "facebook_2": "a181f8e76d893bab287953a7bc2f66cd",
    "meps_19": "4ac4249fc95fe13c6ecc137821ceebb0",
    "meps_20": "51f889ab9888180d358a73616025ff01",
    "meps_21": "522318a2e3ab5d16c2a56610aaae4b76",
}


class DownloadError(Exception):
    """A source file that could not be downloaded."""


def fetch(name, raw):
    """Path of a downloaded source file, downloading it on first use."""
    path = raw / name
    if not path.exists():
        print(f"  downloading {SOURCES[name]}")
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(SOURCES[name], timeout=600) as r:
                # A server under load can answer with an error page and status 200.
                if r.headers.get_content_type() == "text/html":
                    raise DownloadError("the server sent an HTML page instead of the file")
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(r, fh)
                size = r.headers.get("Content-Length")
        except (OSError, http.client.HTTPException) as e:
            raise DownloadError(getattr(e, "reason", e)) from None
        # A connection closed early can end the copy without an error.
        if size is not None and tmp.stat().st_size != int(size):
            raise DownloadError(f"{tmp.stat().st_size} of {size} bytes received")
        tmp.rename(path)
    return path


def zip_member(name, member, raw):
    with zipfile.ZipFile(fetch(name, raw)) as z:
        return io.BytesIO(z.read(member))


def read_arff(path):
    """Read a dense ARFF file into a pandas DataFrame."""
    names, data = [], ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith("@attribute"):
                rest = line.strip()[len("@attribute"):].strip()
                if rest[0] in "\"'":
                    names.append(rest[1:rest.index(rest[0], 1)])
                else:
                    names.append(rest.split()[0])
            elif low.startswith("@data"):
                data = fh.read()
                break
    frame = pd.read_csv(io.StringIO(data), header=None, index_col=False, na_values=["?"],
                        keep_default_na=False, comment="%", quotechar='"',
                        skipinitialspace=True, escapechar="\\")
    frame.columns = names
    return frame


def add_calendar(x, stamp):
    """Hour, day of week, month and day of month of a time stamp."""
    t = pd.to_datetime(stamp, errors="coerce")
    x = x.copy()
    x["dt_hour"], x["dt_dow"] = t.dt.hour, t.dt.dayofweek
    x["dt_month"], x["dt_day"] = t.dt.month, t.dt.day
    return x


def numeric_xy(x, y):
    """Numeric covariates (all-missing columns dropped), the response last, complete rows."""
    x = x.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    out = x.copy()
    out["target"] = pd.to_numeric(y, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


# ---------- The eight main datasets ----------

def naval(raw, _):
    df = pd.read_csv(zip_member("naval.zip", "UCI CBM Dataset/data.txt", raw),
                     sep=r"\s+", header=None)
    # The response is the compressor decay (column 17); the turbine decay (column 18) is removed.
    x = df.iloc[:, :16].set_axis([f"x{j}" for j in range(1, 17)], axis=1)
    return numeric_xy(x, df[16])


def mv(raw, _):
    df = read_arff(fetch("mv.arff", raw))
    # The nominal attributes x3, x7, x8 become all-missing and are dropped.
    return numeric_xy(df.drop(columns="y"), df["y"])


def parking(raw, _):
    df = pd.read_csv(zip_member("parking.zip", "dataset.csv", raw))
    # The car-park identifier SystemCodeNumber is text; numeric_xy drops it as all-missing.
    x = add_calendar(df.drop(columns=["Occupancy", "LastUpdated"]), df["LastUpdated"])
    return numeric_xy(x, df["Occupancy"])


def sarcos(raw, _):
    from scipy.io import loadmat
    # The training and test files are stacked; columns 1-21 are inputs, column 22 the first torque.
    z = np.vstack([loadmat(fetch("sarcos_inv.mat", raw))["sarcos_inv"],
                   loadmat(fetch("sarcos_inv_test.mat", raw))["sarcos_inv_test"]])
    out = pd.DataFrame(z[:, :21], columns=[f"input_{j}" for j in range(1, 22)])
    out["target"] = z[:, 21]
    return out


def sgemm(raw, _):
    df = pd.read_csv(zip_member("sgemm.zip", "sgemm_product.csv", raw))
    # The response is the mean of the four run times (ms); the run columns are removed.
    runs = [c for c in df.columns if c.startswith("Run")]
    out = numeric_xy(df.drop(columns=runs), df[runs].mean(axis=1))
    # A fixed subsample of 50,000 of the 241,600 rows, in the original order.
    return out.sample(n=50000, random_state=1).sort_index()


def tetouan(raw, _):
    df = pd.read_csv(zip_member("tetouan.zip", "Tetuan City power consumption.csv", raw))
    # The response is Zone 1; the simultaneous Zone 2 and Zone 3 consumptions are removed.
    y = df["Zone 1 Power Consumption"]
    x = df.drop(columns=["DateTime", "Zone 1 Power Consumption",
                         "Zone 2  Power Consumption", "Zone 3  Power Consumption"])
    return numeric_xy(add_calendar(x, df["DateTime"]), y)


def video(raw, _):
    df = pd.read_csv(zip_member("video.zip", "transcoding_mesurment.tsv", raw), sep="\t")
    # The response is utime; umem (same run) and the video id are removed.
    x = pd.get_dummies(df.drop(columns=["id", "umem", "utime"]), columns=["codec", "o_codec"],
                       dtype=float)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.loc[:, x.nunique(dropna=False) > 1]  # drops the constant b_size
    x.insert(0, "target", pd.to_numeric(df["utime"], errors="coerce"))
    return x.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


def french(raw, _):
    df = read_arff(fetch("french_load_consumption.arff", raw))
    # The response Load comes first; the time stamp is dropped, the row index "Unnamed: 0" is kept.
    x = df.drop(columns=["Load", "date"])
    x.insert(0, "target", df["Load"])
    return x


# ---------- The eight additional datasets ----------

def casp(raw, _):
    df = pd.read_csv(zip_member("casp.zip", "CASP.csv", raw)).fillna(0)
    # The response is RMSD (first column), the covariates are F1..F9.
    out = df.iloc[:, 1:].copy()
    out["target"] = df.iloc[:, 0]
    return out


def facebook(variant, prefix):
    def build(raw, _):
        member = f"Dataset/Training/Features_Variant_{variant}.csv"
        # The files have no header row. CQR's loader reads the first data row as a header and
        # drops it; this script keeps it, as the paper's runs did.
        df = pd.read_csv(zip_member("facebook.zip", member, raw), header=None)
        # Columns 1-53 become output covariates 0..52 or f0..f52; column 54 is the response.
        out = df.iloc[:, :53].set_axis([f"{prefix}{j}" for j in range(53)], axis=1)
        out["target"] = df[53]
        return out
    return build


def blog(raw, _):
    df = pd.read_csv(zip_member("blog.zip", "blogData_train.csv", raw), header=None)
    # Columns 1-280 are the covariates, the last column is the response.
    out = df.iloc[:, :280].set_axis([f"f{j}" for j in range(1, 281)], axis=1)
    out["target"] = df.iloc[:, -1]
    return out


def bike(_, args):
    df = pd.read_csv(args.bike)
    # One-hot season and weather; hour, day of week, month and year (0, 1) from datetime.
    df = pd.concat([df, pd.get_dummies(df["season"], prefix="season", dtype=int),
                    pd.get_dummies(df["weather"], prefix="weather", dtype=int)], axis=1)
    df = df.drop(columns=["season", "weather"])
    t = pd.DatetimeIndex(df["datetime"])
    df["hour"], df["day"], df["month"] = t.hour, t.dayofweek, t.month
    df["year"] = pd.Series(t.year, index=df.index).map({2011: 0, 2012: 1})
    df = df.drop(columns=["datetime", "casual", "registered"])
    out = df.drop(columns="count")
    out["target"] = df["count"]
    return out


# The 139 covariates of the CQR MEPS loaders; PERWT is the panel's survey weight.
MEPS_COLUMNS = (
    ["AGE", "PCS42", "MCS42", "K6SUM42", "PERWT"]
    + [f"REGION={j}" for j in range(1, 5)] + ["SEX=1", "SEX=2"]
    + [f"MARRY={j}" for j in range(1, 11)] + ["FTSTU=-1", "FTSTU=1", "FTSTU=2", "FTSTU=3"]
    + [f"ACTDTY={j}" for j in range(1, 5)] + [f"HONRDC={j}" for j in range(1, 5)]
    + [f"RTHLTH={j}" for j in (-1, 1, 2, 3, 4, 5)] + [f"MNHLTH={j}" for j in (-1, 1, 2, 3, 4, 5)]
    + [f"{v}={j}" for v in ["HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "EMPHDX",
                            "CHBRON", "CHOLDX", "CANCERDX", "DIABDX", "JTPAIN", "ARTHDX"]
       for j in (-1, 1, 2)]
    + [f"ARTHTYPE={j}" for j in (-1, 1, 2, 3)] + ["ASTHDX=1", "ASTHDX=2"]
    + [f"{v}={j}" for v in ["ADHDADDX", "PREGNT", "WLKLIM", "ACTLIM", "SOCLIM", "COGLIM",
                            "DFHEAR42", "DFSEE42", "ADSMOK42"] for j in (-1, 1, 2)]
    + [f"PHQ242={j}" for j in range(-1, 7)] + [f"EMPST={j}" for j in (-1, 1, 2, 3, 4)]
    + [f"POVCAT={j}" for j in range(1, 6)] + [f"INSCOV={j}" for j in range(1, 4)] + ["RACE"]
)


def meps(panel, weight):
    def build(_, args):
        df = pd.read_csv(Path(args.meps) / f"meps_{panel}_reg.csv")
        cols = [weight if c == "PERWT" else c for c in MEPS_COLUMNS]
        out = df[cols].copy()
        out["target"] = df["UTILIZATION_reg"]
        return out
    return build


BUILDERS = {
    "naval_propulsion_comp_decay": naval,
    "mv": mv,
    "parking_birmingham": parking,
    "sarcos_torque1": sarcos,
    "sgemm": sgemm,
    "tetouan_power_zone1": tetouan,
    "video_transcoding_utime_full": video,
    "french_load_consumption": french,
    "bike": bike,
    "blog": blog,
    "casp_protein": casp,
    "facebook_comments": facebook(1, ""),
    "facebook_2": facebook(2, "f"),
    "meps_19": meps(19, "PERWT15F"),
    "meps_20": meps(20, "PERWT15F"),
    "meps_21": meps(21, "PERWT16F"),
}
MANUAL = {"bike": "bike", "meps_19": "meps", "meps_20": "meps", "meps_21": "meps"}


def r_number(x):
    """A number as R's write.csv prints it (15 significant digits, R's fixed/scientific rule)."""
    if x == 0:
        return "0"
    mantissa, exponent = ("%.14e" % x).split("e")
    e = int(exponent)
    n = max(1, len(mantissa.lstrip("-").replace(".", "").rstrip("0")))
    decimal_places = max(0, n - e - 1)
    width_fixed = (x < 0) + (e + 1 if e >= 0 else 1) + (decimal_places + 1 if decimal_places else 0)
    width_sci = (x < 0) + (n + 1 if n > 1 else 1) + (4 if abs(e) < 100 else 5)
    return "%.*f" % (decimal_places, x) if width_fixed <= width_sci else "%.*e" % (n - 1, x)


# Write options that reproduce the MD5 sums: R's number format for SARCOS, CRLF line ends
# for BlogFeedback.
WRITE = {
    "sarcos_torque1": {"float_format": r_number},
    "blog": {"lineterminator": "\r\n"},
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s [--datasets=KEYS] [--data=DIR] [--raw=DIR] [--bike=FILE] [--meps=DIR]")
    p.add_argument("--datasets", default="all", metavar="KEYS",
                   help="comma-separated keys (default: all available)")
    p.add_argument("--data", default=str(HERE / "processed"), metavar="DIR",
                   help="folder of processed datasets (default: data/processed)")
    p.add_argument("--raw", default=str(HERE / "raw"), metavar="DIR",
                   help="download cache (default: data/raw)")
    p.add_argument("--bike", metavar="FILE", help="Kaggle Bike Sharing Demand train.csv")
    p.add_argument("--meps", metavar="DIR",
                   help="folder with meps_19_reg.csv, meps_20_reg.csv, meps_21_reg.csv")
    args = p.parse_args()

    keys = list(BUILDERS) if args.datasets == "all" else args.datasets.split(",")
    unknown = [k for k in keys if k not in BUILDERS]
    if unknown:
        p.error(f"unknown datasets: {', '.join(unknown)}")
    data_dir, raw = Path(args.data), Path(args.raw)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    failed = []
    for key in keys:
        flag = MANUAL.get(key)
        if flag and getattr(args, flag) is None:
            print(f"{key}: skipped (needs --{flag})")
            if args.datasets != "all":
                failed.append(key)
            continue
        print(f"{key}:")
        try:
            df = BUILDERS[key](raw, args)
        except DownloadError as e:
            print(f"{key}: download failed ({e})")
            failed.append(key)
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError, zipfile.BadZipFile) as e:
            print(f"{key}: a source file in {raw} could not be read ({e}); delete it and run again")
            failed.append(key)
            continue
        path = data_dir / f"{key}.csv"
        df.to_csv(path, index=False, **WRITE.get(key, {}))
        digest = md5(path)
        status = "MD5 ok" if digest == MD5[key] else f"MD5 differs (expected {MD5[key]})"
        print(f"  wrote {path} ({df.shape[0]} rows, {df.shape[1] - 1} covariates), "
              f"{digest} {status}")
        if digest != MD5[key]:
            failed.append(key)
    if failed:
        sys.exit(f"not reproduced: {', '.join(failed)}")


if __name__ == "__main__":
    main()
