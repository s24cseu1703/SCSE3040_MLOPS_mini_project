"""Model pipelines and small evaluation helpers (no MLflow in here)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_columns(params: dict) -> list[str]:
    d = params["data"]
    return d["numeric_features"] + d["categorical_features"]


def make_pipeline(kind: str, hyper: dict, params: dict) -> Pipeline:
    d = params["data"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), d["numeric_features"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), d["categorical_features"]),
    ])
    if kind == "ridge":
        model = Ridge(**hyper)
    elif kind == "gbr":
        model = GradientBoostingRegressor(**hyper)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("pre", pre), ("model", model)])


def fit(kind: str, hyper: dict, df: pd.DataFrame, params: dict) -> Pipeline:
    pipe = make_pipeline(kind, hyper, params)
    return pipe.fit(df[feature_columns(params)], df[params["data"]["target"]])


def mae(model, df: pd.DataFrame, params: dict) -> float:
    pred = model.predict(df[feature_columns(params)])
    return float(mean_absolute_error(df[params["data"]["target"]], pred))


def time_split(df: pd.DataFrame, fraction: float, date_col: str):
    """First `fraction` of the rows by date, and the rest (stable sort)."""
    df = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    cut = int(len(df) * fraction)
    return df.iloc[:cut], df.iloc[cut:]
