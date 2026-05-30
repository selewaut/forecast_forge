from dataclasses import dataclass

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col


@dataclass
class SchemaAdapter:
    column_map: dict
    exogenous: list[str] | None = None

    _REQUIRED = ["unique_id", "ds", "y"]

    def __post_init__(self):
        if self.exogenous is None and "exogenous" in self.column_map:
            self.exogenous = self.column_map["exogenous"]

    def map(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {orig: canon for canon, orig in self.column_map.items() if isinstance(orig, str)}
        df = df.rename(columns=rename)

        for required in self._REQUIRED:
            if required not in df.columns:
                raise KeyError(f"Required column '{required}' not found in DataFrame")

        if df["unique_id"].isnull().any():
            raise ValueError("Column 'unique_id' contains null values")

        return df

    def map_spark(self, sdf: SparkDataFrame) -> SparkDataFrame:
        for canon, orig in self.column_map.items():
            if isinstance(orig, str):
                sdf = sdf.withColumnRenamed(orig, canon)

        for required in self._REQUIRED:
            if required not in sdf.columns:
                raise KeyError(f"Required column '{required}' not found in DataFrame")

        if sdf.filter(col("unique_id").isNull()).count() > 0:
            raise ValueError("Column 'unique_id' contains null values")

        return sdf
