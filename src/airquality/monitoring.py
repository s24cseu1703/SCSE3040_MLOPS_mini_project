"""Drift (PSI), per-group error, and the v1 vs v2 promotion rule. Pure pandas/sklearn."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import modeling


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. Bin edges are the deciles of `expected`
    (cleaned train); the outer edges are open so every value lands in a bin."""
    e = pd.Series(expected).dropna().to_numpy(dtype=float)
    a = pd.Series(actual).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    e_pct, a_pct = np.clip(e_pct, eps, None), np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(train_clean: pd.DataFrame, later: pd.DataFrame, params: dict) -> dict:
    bins = params["monitoring"]["psi_bins"]
    return {c: psi(train_clean[c], later[c], bins) for c in params["data"]["numeric_features"]}


def mae_by_group(model, df: pd.DataFrame, params: dict) -> dict:
    g = params["data"]["group_col"]
    return {str(k): modeling.mae(model, sub, params) for k, sub in df.groupby(g)}


def promotion_decision(mae_v1: float, mae_v2: float, params: dict) -> str:
    """Promote v2 only if its MAE is at least 8% lower than v1's."""
    need = params["monitoring"]["promote_min_improvement"]
    return "promote" if mae_v2 <= (1 - need) * mae_v1 else "keep"


def analyse(v1, best_cfg: dict, train_clean: pd.DataFrame, later: pd.DataFrame, params: dict) -> dict:
    """Everything section 7 of PROBLEM.md asks for. Returns numbers + the fitted v2."""
    date_col = params["data"]["date_col"]
    psi_vals = psi_table(train_clean, later, params)
    by_group = mae_by_group(v1, later, params)

    later_first, later_last = modeling.time_split(
        later, params["split"]["later_train_fraction"], date_col)
    v2_train = pd.concat([train_clean, later_first], ignore_index=True)
    v2 = modeling.fit(best_cfg["kind"], best_cfg["params"], v2_train, params)

    m1, m2 = modeling.mae(v1, later_last, params), modeling.mae(v2, later_last, params)
    return {
        "psi": psi_vals,
        "most_drifted_feature": max(psi_vals, key=psi_vals.get),
        "mae_by_group": by_group,
        "worst_served_group": max(by_group, key=by_group.get),
        "mae_v1_on_check": m1,
        "mae_v2_on_check": m2,
        "decision": promotion_decision(m1, m2, params),
        "n_v2_train": len(v2_train),
        "n_check": len(later_last),
        "v2": v2,
    }
