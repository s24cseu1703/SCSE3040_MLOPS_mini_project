"""Stage 2: clean train.csv, train v1 candidates, track in MLflow, register v1.

Run:  python -m airquality.train
"""
from __future__ import annotations

import joblib

from . import cleaning, data_checks, modeling
from .config import load_csv, load_params, resolve, sha256_of, update_findings


def prepare(params: dict):
    """Load the three files, run the data checks, return everything needed."""
    train, later, holdout = (load_csv(n, params) for n in ("train", "later", "holdout"))
    issues = data_checks.run_all(train, holdout, later, params)
    train_clean = cleaning.clean_train(train, later, issues, params)
    return train_clean, later, holdout, issues


def evaluate_candidates(train_clean, later, params: dict) -> list[dict]:
    """For each candidate: validation MAE (fit on the first 80% by date, score on
    the last 20%) used for choosing, plus the final model fitted on all of
    cleaned train and its MAE on later.csv (information only)."""
    date_col = params["data"]["date_col"]
    fit_part, val_part = modeling.time_split(train_clean, 1 - params["split"]["val_fraction"], date_col)
    results = []
    for cfg in params["candidates"]:
        probe = modeling.fit(cfg["kind"], cfg["params"], fit_part, params)
        final = modeling.fit(cfg["kind"], cfg["params"], train_clean, params)
        results.append({
            "cfg": cfg,
            "mae_val": modeling.mae(probe, val_part, params),
            "mae_train": modeling.mae(final, train_clean, params),
            "mae_later": modeling.mae(final, later, params),
            "model": final,
        })
    return results


def main() -> None:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    params = load_params()
    m = params["mlflow"]
    train_clean, later, _, issues = prepare(params)
    results = evaluate_candidates(train_clean, later, params)
    best = min(results, key=lambda r: r["mae_val"])

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])
    data_hash = sha256_of(resolve(params["paths"]["train"]))

    for r in results:
        with mlflow.start_run(run_name=r["cfg"]["name"]) as run:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("model_kind", r["cfg"]["kind"])
            mlflow.log_params(r["cfg"]["params"])
            mlflow.log_param("n_train_rows", len(train_clean))
            mlflow.log_metric("mae_val", r["mae_val"])
            mlflow.log_metric("mae_train", r["mae_train"])
            mlflow.log_metric("mae_later", r["mae_later"])
            info = mlflow.sklearn.log_model(
                r["model"], name="model", serialization_format="cloudpickle")
            r["run_id"], r["model_uri"] = run.info.run_id, info.model_uri
            print(f"{r['cfg']['name']:15s} mae_val={r['mae_val']:.2f} "
                  f"mae_later={r['mae_later']:.2f} run={run.info.run_id}")

    # register the best run as version 1 and make it the champion
    client = MlflowClient()
    mv = mlflow.register_model(best["model_uri"], m["registered_model_name"])
    client.set_registered_model_alias(m["registered_model_name"], m["champion_alias"], mv.version)
    joblib.dump(best["model"], resolve(params["paths"]["model"]))

    update_findings(params, "data_issues", data_checks.findings_section(issues))
    update_findings(params, "tracking", {
        "experiment_name": m["experiment_name"],
        "best_run_id": best["run_id"],
        "registered_model_name": m["registered_model_name"],
        "champion_version": int(mv.version),
    })
    print("v1 =", best["cfg"]["name"], "-> registered version", mv.version, "as @champion")


if __name__ == "__main__":
    main()
