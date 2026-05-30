import dataclasses

import yaml

from forecast_forge.loaders.base import DataLoadingConfig


def test_yaml_round_trip():
    config = DataLoadingConfig(
        loader_module="forecast_forge.loaders.csv",
        loader_class="CsvLoader",
        files={"train_path": "data/train.csv"},
        schema={"unique_id": "id", "ds": "date", "y": "target", "exogenous": ["promo"]},
    )
    data = yaml.safe_load(yaml.dump(dataclasses.asdict(config)))
    recreated = DataLoadingConfig(**data)
    assert recreated == config


def test_schema_from_yaml_dict():
    from forecast_forge.loaders.schema import SchemaAdapter

    mapping = {"unique_id": "store", "ds": "dt", "y": "sales", "exogenous": ["promo", "holiday"]}
    adapter = SchemaAdapter(column_map=mapping)
    assert adapter.column_map["unique_id"] == "store"
    assert adapter.column_map["ds"] == "dt"
    assert adapter.exogenous == ["promo", "holiday"]
