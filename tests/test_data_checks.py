import numpy as np

from airquality import cleaning, data_checks


def _broken(df):
    df = df.copy()
    # unit mix-up: Industrial pm25_today recorded x1000
    df.loc[df.station_zone == "Industrial", "pm25_today"] *= 1000
    # sentinel -999 in wind
    df.loc[df.index[:12], "wind_kmph"] = -999
    # typos in the target
    df.loc[df.index[20:24], "pm25_next_day"] *= 10
    # messy spellings
    df.loc[df.index[30:40], "day_type"] = " WEEKEND"
    # stuck sensor
    df.loc[df.index[100:150], "stubble_fires"] = 43
    return df


def test_each_problem_is_found(small_train, params):
    df = _broken(small_train)
    holdout = df.drop(columns="pm25_next_day")
    later = small_train.copy()
    issues = data_checks.run_all(df, holdout, later, params)

    assert issues["unit_mixup"]["present"]
    assert issues["unit_mixup"]["column"] == "pm25_today"
    assert issues["unit_mixup"]["group_value"] == "Industrial"
    assert issues["unit_mixup"]["factor"] == 1000
    assert issues["sentinel_value"]["value"] == -999
    assert issues["sentinel_value"]["count"] == 12
    assert issues["target_typos"]["count"] == 4
    assert issues["spelling_variants"]["column"] == "day_type"
    assert issues["spelling_variants"]["true_levels"] == 3
    assert issues["stuck_sensor"]["column"] == "stubble_fires"
    assert issues["stuck_sensor"]["start_date"] < issues["stuck_sensor"]["end_date"]
    assert not issues["leaking_column"]["present"]
    assert not issues["duplicate_rows"]["present"]


def test_duplicates_and_leak_are_found(small_train, params):
    df = small_train.copy()
    df["pm25_final_report"] = df["pm25_next_day"]          # leaks the answer
    df = __import__("pandas").concat([df, df.iloc[:5]], ignore_index=True)
    holdout = small_train.drop(columns="pm25_next_day")
    assert data_checks.check_duplicate_rows(df, params)["count"] == 5
    leak = data_checks.check_leaking_column(df, holdout, params)
    assert leak == {"present": True, "column": "pm25_final_report"}


def test_cleaning_repairs_the_data(small_train, params):
    df = _broken(small_train)
    later = small_train.copy()
    issues = data_checks.run_all(df, df.drop(columns="pm25_next_day"), later, params)
    clean = cleaning.clean_train(df, later, issues, params)
    assert clean.pm25_today.max() < 1000
    assert (clean.wind_kmph.dropna() > 0).all()
    assert clean.wind_kmph.isna().sum() == 12
    assert clean.pm25_next_day.max() < 500
    assert set(clean.day_type) == {"weekday", "weekend", "festival"}
    assert clean.stubble_fires.isna().sum() >= 40
