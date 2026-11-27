"""Small helpers shared by every module: params, hashing, findings file."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def sha256_of(path: str | Path) -> str:
    """SHA-256 of the file's raw bytes (logged to MLflow as data_sha256)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(name: str, params: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(params["paths"][name]))


def update_findings(params: dict, section: str, values: dict) -> None:
    """Merge `values` into one top-level section of reports/findings.yaml."""
    path = resolve(params["paths"]["findings"])
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc.setdefault(section, {})
    for key, val in values.items():
        if isinstance(val, dict) and isinstance(doc[section].get(key), dict):
            doc[section][key].update(val)
        else:
            doc[section][key] = val
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
