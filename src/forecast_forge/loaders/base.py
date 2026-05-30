from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataLoadingConfig:
    loader_module: str = ""
    loader_class: str = ""
    files: dict | None = None
    schema: dict | None = None
    download: dict | None = None


class BaseDataLoader(ABC):
    def __init__(self, config: DataLoadingConfig | None = None):
        self.config = config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    def load_spark(self, spark):
        raise NotImplementedError(
            f"{type(self).__name__} does not support Spark-native loading. "
            "Use load() with spark.createDataFrame() instead."
        )
