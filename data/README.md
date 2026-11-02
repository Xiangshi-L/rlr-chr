# Data

`prepare_data.py` downloads the twelve datasets available without a login, writes
`data/processed/<key>.csv` and checks each file's MD5 sum against the recorded values.
For Bike Sharing (Kaggle login) and the three MEPS panels (AHRQ data use agreement),
use the commands below to obtain the inputs from SLCP and pass them to the script.

```
python3 data/prepare_data.py                                    # the twelve public datasets
git clone https://github.com/aaronhan223/SLCP.git third_party/SLCP
git -C third_party/SLCP checkout b14bfb2542136f7e06a461d11b688d6b5eb58b64
python3 data/prepare_data.py --datasets=bike --bike=third_party/SLCP/SLCP/datasets/bike_train.csv
python3 data/prepare_data.py --datasets=meps_19,meps_20,meps_21 --meps=third_party/SLCP/SLCP/get_meps_data
```

Every processed file has a header row, the response in the column `target` and numeric
covariates only. `main_manifest.csv` and `additional_manifest.csv` list the key and display name
of each dataset. The R scripts read `<key>.csv`, for a key of either manifest, from the
folder given by `--data` (default `data/processed`).

## Main datasets

| key | source | license | rows | response | preprocessing |
|---|---|---|---|---|---|
| `sarcos_torque1` | [GPML SARCOS](https://gaussianprocess.org/gpml/data/) | none stated | 48,933 | `torque_1` | training and test files stacked; `input_1`..`input_21` only |
| `sgemm` | [UCI 440](https://archive.ics.uci.edu/dataset/440) | CC BY 4.0 | 50,000 | mean of `Run1`..`Run4` | 50,000 of 241,600 rows (`sample(n=50000, random_state=1)`, original order) |
| `tetouan_power_zone1` | [UCI 849](https://archive.ics.uci.edu/dataset/849) | CC BY 4.0 | 52,416 | Zone 1 consumption | Zones 2 and 3 removed; `DateTime` to hour, weekday, month, day |
| `parking_birmingham` | [UCI 482](https://archive.ics.uci.edu/dataset/482) | CC BY 4.0 | 35,717 | `Occupancy` | `SystemCodeNumber` removed; `LastUpdated` to hour, weekday, month, day |
| `naval_propulsion_comp_decay` | [UCI 316](https://archive.ics.uci.edu/dataset/316) | CC BY 4.0 | 11,934 | compressor decay | turbine decay removed; 16 covariates `x1`..`x16` |
| `mv` | [OpenML 344](https://www.openml.org/d/344) | OpenML: Public | 40,768 | `y` | nominal `x3`, `x7`, `x8` dropped |
| `video_transcoding_utime_full` | [UCI 335](https://archive.ics.uci.edu/dataset/335) | CC BY 4.0 | 68,784 | `utime` | `id` and `umem` removed; `codec`, `o_codec` one-hot; constant `b_size` dropped |
| `french_load_consumption` | [OpenML 46340](https://www.openml.org/d/46340) | OpenML: Public | 105,168 | `Load` | `date` dropped; the row-index column `Unnamed: 0` kept as a covariate |

## Additional datasets

These follow `datasets.py` of the CQR benchmark code (Romano, Patterson and Candes, 2019;
MIT license) and its copy in SLCP (BSD-3-Clause license); CHR reuses the same loaders.
Responses stay on their original scale.

| key | source | license | rows | response | preprocessing |
|---|---|---|---|---|---|
| `bike` | Kaggle Bike Sharing Demand, `train.csv` | Kaggle competition rules | 10,886 | `count` | `season`, `weather` one-hot; hour, weekday, month, year (0, 1) from `datetime`; `casual`, `registered` removed |
| `casp_protein` | [UCI 265](https://archive.ics.uci.edu/dataset/265), `CASP.csv` | CC BY 4.0 | 45,730 | `RMSD` | `F1`..`F9` |
| `meps_19` | MEPS panel 19, `meps_19_reg.csv` | AHRQ data use agreement | 15,785 | `UTILIZATION_reg` | the 139 covariates of the CQR loader |
| `meps_20` | MEPS panel 20, `meps_20_reg.csv` | AHRQ data use agreement | 17,541 | `UTILIZATION_reg` | as `meps_19` |
| `meps_21` | MEPS panel 21, `meps_21_reg.csv` | AHRQ data use agreement | 15,656 | `UTILIZATION_reg` | as `meps_19` (weight `PERWT16F`) |
| `facebook_comments` | [UCI 363](https://archive.ics.uci.edu/dataset/363), `Features_Variant_1.csv` | CC BY 4.0 | 40,949 | column 54 | columns 1-53; all rows kept (CQR's loader reads the first row as a header and drops it) |
| `facebook_2` | [UCI 363](https://archive.ics.uci.edu/dataset/363), `Features_Variant_2.csv` | CC BY 4.0 | 81,312 | column 54 | as `facebook_comments` |
| `blog` | [UCI 304](https://archive.ics.uci.edu/dataset/304), `blogData_train.csv` | CC BY 4.0 | 52,397 | column 281 | columns 1-280 |

Bike Sharing is `train.csv` of the Kaggle competition "Bike Sharing Demand"; the MEPS files
are the `meps_<panel>_reg.csv` that the MEPS scripts of the CQR code build from the AHRQ
full-year files h181 (2015) and h192 (2016). We used the copies in the SLCP repository, which
the commands above clone; the MEPS files are subject to the AHRQ data use agreement.
