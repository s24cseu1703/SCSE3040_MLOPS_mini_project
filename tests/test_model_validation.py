import pytest
from airquality.modeling import make_pipeline


def test_invalid_model_type():
    params = {
        "data": {
            "numeric_features": ["temperature", "humidity"],
            "categorical_features": ["station_zone"]
        }
    }

    with pytest.raises(ValueError, match="unknown model kind"):
        make_pipeline("invalid_model", {}, params)