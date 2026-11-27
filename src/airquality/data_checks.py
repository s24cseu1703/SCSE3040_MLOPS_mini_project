"""The seven data-problem checks. Every answer comes from looking at the data.

`run_all(train, holdout, later, params)` returns a dict shaped like the
`data_issues` section of findings.yaml.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _features(params: dict) -> list[str]:
    return params["data"]["numeric_features"]


# 1. leaking column ---------------------------------------------------------
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, params: dict) -> dict:
    """A column present in train but missing from holdout (other than the target),
    or one that agrees almost exactly with the target."""
    target = params["data"]["target"]
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    if missing:
        return {"present": True, "column": missing[0]}
    for col in train.select_dtypes("number").columns:
        if col != target and abs(train[col].corr(train[target])) > 0.99:
            return {"present": True, "column": col}
    return {"present": False, "column": None}


# 2. unit mix-up --------------------------------------------------------------
def check_unit_mixup(train: pd.DataFrame, params: dict) -> dict:
    """One group has a column ~10/12/60/100/1000 times bigger or smaller."""
    group = params["data"]["group_col"]
    factors = params["data"]["unit_factor_candidates"]
    best = None
    for col in _features(params):
        med = train.groupby(group)[col].median()
        for g in med.index:
            others = train.loc[train[group] != g, col].median()
            if med[g] == 0 or others == 0:
                continue
            ratio = abs(med[g] / others)
            for f in factors:
                for r, factor in ((ratio, f), (1 / ratio, 1 / f)):
                    err = abs(np.log(r / f))
                    if err < 0.3 and (best is None or err < best[0]):
                        best = (err, col, g, factor)
    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}
    _, col, g, factor = best
    # factor: wrong value = right value x factor (a big group means factor > 1)
    return {"present": True, "column": col, "group_value": g,
            "factor": int(factor) if factor >= 1 else round(factor, 6)}


# 3. sentinel value ---------------------------------------------------------------
def check_sentinel_value(train: pd.DataFrame, params: dict) -> dict:
    best = None
    for col in _features(params):
        for v in params["data"]["sentinel_candidates"]:
            n = int((train[col] == v).sum())
            if n >= 5 and (best is None or n > best[2]):
                best = (col, v, n)
    if best is None:
        return {"present": False, "column": None, "value": None, "count": None}
    return {"present": True, "column": best[0], "value": best[1], "count": best[2]}


# 4. duplicate rows -------------------------------------------------------------------
def check_duplicate_rows(train: pd.DataFrame, params: dict) -> dict:
    """Extra copies of a record. record_id is ignored so re-typed rows with a
    new id would also be caught."""
    cols = [c for c in train.columns if c != params["data"]["id_col"]]
    n = int(train.duplicated(subset=cols).sum())
    return {"present": n > 0, "count": n if n > 0 else None}


# 5. stuck sensor -------------------------------------------------------------------------
def check_stuck_sensor(train: pd.DataFrame, params: dict) -> dict:
    """A value repeated far more often than any other, over a stretch of dates."""
    d = params["data"]
    sentinels = set(d["sentinel_candidates"])
    for col in _features(params):
        vals = train.loc[~train[col].isin(sentinels), col]
        vc = vals.value_counts()
        if len(vc) < 2:
            continue
        top_val, top_n = vc.index[0], int(vc.iloc[0])
        if top_n >= d["stuck_min_rows"] and top_n >= d["stuck_dominance"] * int(vc.iloc[1]):
            dates = train.loc[train[col] == top_val, d["date_col"]]
            return {"present": True, "column": col, "start_date": str(dates.min()),
                    "end_date": str(dates.max()), "value": float(top_val), "rows": top_n}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


# 6. target typos ------------------------------------------------------------------------------
def check_target_typos(train: pd.DataFrame, params: dict) -> dict:
    t = train[params["data"]["target"]]
    n = int((t > params["data"]["target_typo_ratio"] * t.median()).sum())
    return {"present": n > 0, "count": n if n > 0 else None}


# 7. spelling variants ----------------------------------------------------------------------------
def _normalise_token(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def canonical_map(raw_values, true_levels: list[str], counts: pd.Series | None = None) -> dict:
    """Map every raw spelling to one of the true levels (prefix matching).

    A spelling that fits several levels (e.g. 'wee.' = weekday or weekend) goes
    to the most frequent of them.
    """
    mapping = {}
    for raw in raw_values:
        tok = _normalise_token(raw)
        cands = [lv for lv in true_levels if lv.startswith(tok) or tok.startswith(lv)] if tok else []
        if not cands:
            mapping[raw] = None
        elif len(cands) == 1 or counts is None:
            mapping[raw] = cands[0]
        else:
            mapping[raw] = max(cands, key=lambda lv: counts.get(lv, 0))
    return mapping


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame, params: dict) -> dict:
    """later.csv has the clean spellings: a text column with more distinct
    values in train than in later has variants."""
    skip = {params["data"]["id_col"], params["data"]["date_col"]}
    for col in train.select_dtypes(exclude="number").columns:
        if col in skip:
            continue
        levels = sorted(later[col].unique())
        if train[col].nunique() > len(levels):
            return {"present": True, "column": col, "true_levels": len(levels)}
    return {"present": False, "column": None, "true_levels": None}


# all together ----------------------------------------------------------------------------------------
def run_all(train, holdout, later, params) -> dict:
    return {
        "leaking_column": check_leaking_column(train, holdout, params),
        "unit_mixup": check_unit_mixup(train, params),
        "sentinel_value": check_sentinel_value(train, params),
        "duplicate_rows": check_duplicate_rows(train, params),
        "stuck_sensor": check_stuck_sensor(train, params),
        "target_typos": check_target_typos(train, params),
        "spelling_variants": check_spelling_variants(train, later, params),
    }


def findings_section(issues: dict) -> dict:
    """Keep only the keys findings.yaml asks for."""
    keep = {
        "leaking_column": ["present", "column"],
        "unit_mixup": ["present", "column", "group_value", "factor"],
        "sentinel_value": ["present", "column", "value", "count"],
        "duplicate_rows": ["present", "count"],
        "stuck_sensor": ["present", "column", "start_date", "end_date"],
        "target_typos": ["present", "count"],
        "spelling_variants": ["present", "column", "true_levels"],
    }
    out = {}
    for kind, keys in keep.items():
        out[kind] = {k: issues[kind].get(k) for k in keys}
    return out


if __name__ == "__main__":
    import json

    from .config import load_csv, load_params, update_findings

    p = load_params()
    result = run_all(load_csv("train", p), load_csv("holdout", p), load_csv("later", p), p)
    update_findings(p, "data_issues", findings_section(result))
    print(json.dumps(result, indent=2, default=str))
