from airquality.modeling import feature_columns


def test_feature_column_order():
    params = {
        "data": {
            "numeric_features": [
                "temperature",
                "humidity",
                "wind_speed"
            ],
            "categorical_features": [
                "station_zone",
                "day_type"
            ]
        }
    }

    expected = [
        "temperature",
        "humidity",
        "wind_speed",
        "station_zone",
        "day_type"
    ]

    assert feature_columns(params) == expected