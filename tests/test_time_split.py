import pandas as pd
from airquality.modeling import time_split


def test_chronological_split():
    df = pd.DataFrame({
        "date": [
            "2026-01-03",
            "2026-01-01",
            "2026-01-04",
            "2026-01-02"
        ],
        "value": [30, 10, 40, 20]
    })

    train, validation = time_split(df, 0.5, "date")

    assert train["value"].tolist() == [10, 20]
    assert validation["value"].tolist() == [30, 40]
    assert len(train) + len(validation) == len(df)