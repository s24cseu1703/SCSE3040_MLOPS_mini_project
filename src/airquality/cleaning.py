"""Cleaning: apply a fix for every problem the data checks found."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_checks import canonical_map


def clean_train(train: pd.DataFrame, later: pd.DataFrame, issues: dict, params: dict) -> pd.DataFrame:
    """Return a cleaned copy of train.csv. Values we cannot trust become NaN
    (the model pipeline imputes them); typos are corrected."""
    d = params["data"]
    df = train.copy()

    # duplicates
    if issues["duplicate_rows"]["present"]:
        cols = [c for c in df.columns if c != d["id_col"]]
        df = df.drop_duplicates(subset=cols).copy()

    # spelling variants -> true levels (taken from the clean later.csv)
    sv = issues["spelling_variants"]
    if sv["present"]:
        col = sv["column"]
        levels = sorted(later[col].unique())
        counts = df[col].astype(str).str.strip().str.lower().value_counts()
        mapping = canonical_map(df[col].unique(), levels, counts)
        df[col] = df[col].map(mapping)

    # unit mix-up -> divide the affected group's values by the factor
    um = issues["unit_mixup"]
    if um["present"]:
        rows = df[d["group_col"]] == um["group_value"]
        df.loc[rows, um["column"]] = df.loc[rows, um["column"]] / um["factor"]

    # sentinel -> missing
    sn = issues["sentinel_value"]
    if sn["present"]:
        df.loc[df[sn["column"]] == sn["value"], sn["column"]] = np.nan

    # stuck sensor -> missing inside the stuck window
    ss = issues["stuck_sensor"]
    if ss["present"]:
        col = ss["column"]
        stuck = df[col] == ss["value"]
        window = df[d["date_col"]].between(ss["start_date"], ss["end_date"])
        df.loc[stuck & window, col] = np.nan

    # target typos -> divide by 10
    if issues["target_typos"]["present"]:
        t = df[d["target"]]
        typo = t > d["target_typo_ratio"] * t.median()
        df.loc[typo, d["target"]] = t[typo] / 10

    return df.reset_index(drop=True)
