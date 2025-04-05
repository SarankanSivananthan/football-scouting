"""Reads the .rds snapshots downloaded by scripts/download_data.py.

.rds is R's native serialization format; these files were produced by the
R package `worldfootballR`. Rather than requiring R to be installed, this
project reads them directly in Python via the pure-Python `rdata` package
(no R runtime, no C extension build needed).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import rdata

# Resolved relative to this file, not the caller's working directory —
# so this loads correctly whether it's imported from a script run at the
# project root or from a notebook running with notebooks/ as its cwd.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TABLES = [
    "big5_player_standard", "big5_player_shooting", "big5_player_passing",
    "big5_player_possession", "big5_player_defense", "big5_player_gca",
    "big5_player_misc",
]


def load_table(name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        # rdata warns about a couple of R date/time classes (POSIXlt/POSIXt)
        # that these tables don't actually use in any column we read.
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(f"{RAW_DIR}/{name}.rds")
        return rdata.conversion.convert(parsed)


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in TABLES}
