import pandas as pd
import pytest


@pytest.fixture
def synthetic_weekly_data() -> pd.DataFrame:
    groups = ["A", "B"]
    rows = []
    for group in groups:
        for week in range(52):
            rows.append({"id": group, "dt": f"2024-01-{week + 1:02d}", "val": float(week)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def canonical_weekly_data() -> pd.DataFrame:
    groups = ["A", "B"]
    rows = []
    for group in groups:
        for week in range(52):
            rows.append({"unique_id": group, "ds": f"2024-01-{week + 1:02d}", "y": float(week)})
    return pd.DataFrame(rows)
