import pandas as pd
import pytest

from forecast_forge.loaders.schema import SchemaAdapter


def test_renames_columns_to_canonical():
    df = pd.DataFrame({"store": ["A", "B"], "dt": ["2024-01-01", "2024-01-08"], "sales": [100.0, 200.0]})
    adapter = SchemaAdapter(column_map={"unique_id": "store", "ds": "dt", "y": "sales"})
    result = adapter.map(df)
    assert list(result.columns) == ["unique_id", "ds", "y"]
    assert result["unique_id"].tolist() == ["A", "B"]
    assert result["y"].tolist() == [100.0, 200.0]


def test_extra_columns_pass_through():
    df = pd.DataFrame({"a": [1, 2], "ds": ["2024-01-01", "2024-01-08"], "y": [10.0, 20.0], "extra": [3, 4]})
    adapter = SchemaAdapter(column_map={"unique_id": "a"})
    result = adapter.map(df)
    assert list(result.columns) == ["unique_id", "ds", "y", "extra"]


def test_missing_required_column_raises_key_error():
    df = pd.DataFrame({"unique_id": ["A"], "ds": ["2024-01-01"]})
    adapter = SchemaAdapter(column_map={})
    with pytest.raises(KeyError, match="y"):
        adapter.map(df)


def test_exogenous_columns_preserved():
    df = pd.DataFrame({
        "id": ["A", "B"], "dt": ["2024-01-01", "2024-01-08"], "y": [100.0, 200.0],
        "promo": [1, 0], "holiday": [0, 1],
    })
    adapter = SchemaAdapter(column_map={"unique_id": "id", "ds": "dt"}, exogenous=["promo", "holiday"])
    result = adapter.map(df)
    assert "promo" in result.columns
    assert "holiday" in result.columns


def test_empty_map_identity():
    df = pd.DataFrame({"unique_id": ["A"], "ds": ["2024-01-01"], "y": [10.0]})
    adapter = SchemaAdapter(column_map={})
    result = adapter.map(df)
    assert list(result.columns) == ["unique_id", "ds", "y"]


def test_null_unique_id_raises_value_error():
    df = pd.DataFrame({"unique_id": ["A", None], "ds": ["2024-01-01", "2024-01-08"], "y": [10.0, 20.0]})
    adapter = SchemaAdapter(column_map={})
    with pytest.raises(ValueError, match="unique_id"):
        adapter.map(df)


def test_map_spark_renames_columns(spark):
    pdf = pd.DataFrame({"store": ["A"], "dt": ["2024-01-01"], "sales": [100.0]})
    sdf = spark.createDataFrame(pdf)
    adapter = SchemaAdapter(column_map={"unique_id": "store", "ds": "dt", "y": "sales"})
    result = adapter.map_spark(sdf)
    assert result.columns == ["unique_id", "ds", "y"]


def test_map_spark_missing_column_raises_key_error(spark):
    pdf = pd.DataFrame({"unique_id": ["A"], "ds": ["2024-01-01"]})
    sdf = spark.createDataFrame(pdf)
    adapter = SchemaAdapter(column_map={})
    with pytest.raises(KeyError, match="y"):
        adapter.map_spark(sdf)


def test_map_spark_null_unique_id_raises_value_error(spark):
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    schema = StructType([
        StructField("unique_id", StringType(), True),
        StructField("ds", StringType(), True),
        StructField("y", DoubleType(), True),
    ])
    sdf = spark.createDataFrame([(None, "2024-01-01", 10.0)], schema=schema)
    adapter = SchemaAdapter(column_map={})
    with pytest.raises(ValueError, match="unique_id"):
        adapter.map_spark(sdf)
