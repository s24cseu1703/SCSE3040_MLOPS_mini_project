import numpy as np
import pandas as pd

from airquality import modeling, monitoring


def test_psi_is_zero_for_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    assert monitoring.psi(a, a) < 1e-6
    assert monitoring.psi(a, a + 3) > 1.0


def test_promotion_rule(params):
    assert monitoring.promotion_decision(100, 91, params) == "promote"
    assert monitoring.promotion_decision(100, 93, params) == "keep"


def test_model_trains_and_scores_per_group(small_train, params):
    cfg = params["candidates"][0]
    model = modeling.fit(cfg["kind"], cfg["params"], small_train, params)
    by_group = monitoring.mae_by_group(model, small_train, params)
    assert set(by_group) == {"Central", "Residential", "Industrial"}
    assert all(v < 20 for v in by_group.values())
