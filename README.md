# MP097 - Tomorrow's Air Quality (Bhopal)

**Group:** G142  **Problem:** MP097  **Course:** SCSE3040 MLOps (Faculty: Ravi Pratap, Batch 39)

| Enrolment no. | Name |
|---|---|
| S24CSEU1705 | Jainendra Verma |
| S24CSEU1703 | Aayush Vashistha |
| S24CSEU1708 | Naitik Tiwari |
| S24CSEU1692 | Saksham Agrawal |

Client: Bhopal Clean Air Cell. Task: predict `pm25_next_day` (regression, MAE) per station,
so schools and hospitals get a one-day warning.

## Layout
```
src/airquality/   data_checks.py  the seven data-problem checks
                  cleaning.py     fixes for what the checks find
                  modeling.py     sklearn pipelines
                  train.py        v1: candidates, MLflow tracking, registry, @champion
                  monitoring.py   PSI, MAE per group, promotion rule
                  monitor.py      v2 retrain, decision, holdout predictions
app/main.py       FastAPI service (POST /predict, GET /health, JSON logs)
tests/            pytest (no CSVs needed)
k8s/              Deployment mp097-api (2 replicas), Service mp097-svc (80 -> 8000)
params.yaml       every setting; read by the code
reports/findings.yaml, predictions/holdout_predictions.csv, mlflow.db, models/model.joblib
```

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python -m airquality.data_checks   # stage 1: the seven checks -> reports/findings.yaml
python -m airquality.train         # stage 2: v1, MLflow runs, registry, @champion
python -m airquality.monitor       # stage 4: PSI, per-group MAE, v2, decision, holdout predictions
pytest                             # tests

docker build -t mp097-api:1.0 .
docker run -p 8000:8000 mp097-api:1.0
curl localhost:8000/health
kubectl apply -f k8s/
```

## Data problems found in train.csv
| Problem | Result |
|---|---|
| leaking column | none (train has no column missing from the holdout file) |
| unit mix-up | `pm25_today` for Industrial is x1000 |
| sentinel value | `wind_kmph` = -999 in 46 rows |
| duplicate rows | none |
| stuck sensor | `stubble_fires` = 43, 2026-04-12 to 2026-05-15 |
| target typos | 14 targets 10x too big |
| spelling variants | `day_type` has 17 spellings of 3 real levels |

## Monitoring and retraining
`temp_c` drifts most (about +20 C from January-June to July-September; PSI > 4).
Linear models extrapolate this shift better than tree models, so v1 is a Ridge model.
v2 (train + first 70% of later.csv) must be at least 8% better than v1 on the last 30%
to be promoted; the numbers are in `reports/findings.yaml`.
