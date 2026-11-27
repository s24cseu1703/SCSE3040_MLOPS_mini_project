"""Stage 4: drift, per-group error, v2 retrain, promotion rule, holdout prediction.

Run (after `python -m airquality.train`):  python -m airquality.monitor
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import modeling, monitoring
from .config import load_params, resolve, sha256_of, update_findings
from .train import prepare


def write_predictions(model, holdout: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = pd.DataFrame({
        "record_id": holdout[params["data"]["id_col"]],
        "prediction": model.predict(holdout[modeling.feature_columns(params)]),
    })
    path = resolve(params["paths"]["predictions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    params = load_params()
    m = params["mlflow"]
    train_clean, later, holdout, _ = prepare(params)

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])
    client = MlflowClient()
    name, alias = m["registered_model_name"], m["champion_alias"]

    # v1 = the current champion, and the settings that won
    v1 = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    v1_version = client.get_model_version_by_alias(name, alias)
    v1_run = client.get_run(v1_version.run_id)
    best_cfg = next(c for c in params["candidates"]
                    if c["name"] == v1_run.info.run_name)

    res = monitoring.analyse(v1, best_cfg, train_clean, later, params)

    # track and register v2 (version 2)
    with mlflow.start_run(run_name=f"v2_{best_cfg['name']}") as run:
        mlflow.log_param("data_sha256", sha256_of(resolve(params["paths"]["train"])))
        mlflow.log_param("model_kind", best_cfg["kind"])
        mlflow.log_params(best_cfg["params"])
        mlflow.log_param("n_train_rows", res["n_v2_train"])
        mlflow.log_metric("mae_check_v1", res["mae_v1_on_check"])
        mlflow.log_metric("mae_check_v2", res["mae_v2_on_check"])
        mlflow.set_tag("decision", res["decision"])
        info = mlflow.sklearn.log_model(res["v2"], name="model", serialization_format="cloudpickle")
    v2_version = mlflow.register_model(info.model_uri, name)

    if res["decision"] == "promote":
        client.set_registered_model_alias(name, alias, v2_version.version)

    champion = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    joblib.dump(champion, resolve(params["paths"]["model"]))
    preds = write_predictions(champion, holdout, params)

    champion_version = int(client.get_model_version_by_alias(name, alias).version)
    update_findings(params, "tracking", {"champion_version": champion_version})
    update_findings(params, "monitoring", {
        "psi": {k: round(v, 4) for k, v in res["psi"].items()},
        "most_drifted_feature": res["most_drifted_feature"],
        "mae_by_group": {k: round(v, 3) for k, v in res["mae_by_group"].items()},
        "worst_served_group": res["worst_served_group"],
        "mae_v1_on_check": round(res["mae_v1_on_check"], 3),
        "mae_v2_on_check": round(res["mae_v2_on_check"], 3),
        "decision": res["decision"],
    })
    print(json.dumps({k: v for k, v in res.items() if k != "v2"}, indent=2, default=str))
    print(f"champion = version {champion_version}; wrote {len(preds)} predictions")


if __name__ == "__main__":
    main()
