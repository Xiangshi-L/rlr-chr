#!/usr/bin/env python3
"""Table 1: WSC and length relative to CHR on the eight main datasets, with RLCP."""
import os

from common import (MAIN_DATASETS, load_main, paired_table, parse_args, print_degenerate,
                    print_share_above_T)
from table_bodies import write_body


def main():
    args = parse_args(__doc__)
    cells, rlcp, _ = load_main(args.results)
    print_degenerate(cells, MAIN_DATASETS)
    frac_inf = [100 * cell["frac_inf"] for cell in rlcp.values()]
    print(f"RLCP: unbounded intervals on {min(frac_inf):.1f}% to {max(frac_inf):.1f}% "
          "of test points")
    print_share_above_T({name: os.path.join(args.results, "main", f"{key}_long.csv")
                         for key, name in MAIN_DATASETS}, "main datasets")
    write_body(os.path.join(args.out, "main_table_merged_body.tex"),
               paired_table(cells, MAIN_DATASETS, rlcp=rlcp))


if __name__ == "__main__":
    main()
