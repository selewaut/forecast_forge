import pandas as pd
import pytest
from pyspark.sql import DataFrame

from forecast_forge.loaders.base import BaseDataLoader


def test_cannot_instantiate_base_loader():
    with pytest.raises(TypeError):
        BaseDataLoader()


def test_missing_load_method_raises_type_error():
    with pytest.raises(TypeError):

        class _Invalid(BaseDataLoader):
            pass

        _Invalid()


def test_dict_loader_returns_dataframe():
    class _DictLoader(BaseDataLoader):
        def load(self) -> pd.DataFrame:
            return pd.DataFrame({"unique_id": ["A"], "ds": ["2024-01-01"], "y": [10.0]})

    loader = _DictLoader()
    result = loader.load()
    assert isinstance(result, pd.DataFrame)


def test_dict_loader_applies_schema():
    from forecast_forge.loaders.schema import SchemaAdapter

    class _DictLoader(BaseDataLoader):
        def __init__(self):
            super().__init__()
            self._adapter = SchemaAdapter(column_map={"unique_id": "id", "ds": "dt", "y": "val"},
                                          exogenous=["promo"])

        def load(self) -> pd.DataFrame:
            df = pd.DataFrame({"id": ["A"], "dt": ["2024-01-01"], "val": [10.0], "promo": [1]})
            return self._adapter.map(df)

    loader = _DictLoader()
    result = loader.load()
    assert list(result.columns) == ["unique_id", "ds", "y", "promo"]


def test_load_spark_raises_not_implemented():
    class _Loader(BaseDataLoader):
        def load(self) -> pd.DataFrame:
            return pd.DataFrame()

    loader = _Loader()
    with pytest.raises(NotImplementedError, match="Spark"):
        loader.load_spark(None)


def test_load_spark_can_be_overridden(spark):
    class _SparkLoader(BaseDataLoader):
        def load(self) -> pd.DataFrame:
            return pd.DataFrame()

        def load_spark(self, spark):
            return spark.createDataFrame(pd.DataFrame({"unique_id": ["A"], "ds": ["2024-01-01"], "y": [10.0]}))

    result = _SparkLoader().load_spark(spark)
    assert isinstance(result, DataFrame)


def test_csv_loader_reads_from_disk(tmp_path):
    from forecast_forge.loaders.schema import SchemaAdapter

    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"id": ["A", "B"], "dt": ["2024-01-01", "2024-01-08"], "y": [10.0, 20.0]}).to_csv(csv_path, index=False)

    class _CsvLoader(BaseDataLoader):
        def __init__(self, path):
            super().__init__()
            self._path = path
            self._adapter = SchemaAdapter(column_map={"unique_id": "id", "ds": "dt"})

        def load(self) -> pd.DataFrame:
            df = pd.read_csv(self._path)
            return self._adapter.map(df)

    loader = _CsvLoader(csv_path)
    result = loader.load()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["unique_id", "ds", "y"]
    assert len(result) == 2
