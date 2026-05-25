# Task 0001: Migrate Packaging to UV

## Status

Done

## Epic

[Epic 0001: Local UV, Python, and Spark Modernization](../epics/0001-uv-python-spark-modernization.md)

## Objective

Add `uv` project metadata and lockfile support while preserving the existing package layout.

## Scope

- Add `pyproject.toml`.
- Add `uv.lock`.
- Track `.python-version` with Python 3.13.
- Move runtime dependency declarations into `pyproject.toml`.
- Add `pytest` as a development dependency.
- Keep legacy packaging files for now to reduce migration risk.

## Implementation Notes

- `mlflow` was moved from `2.16.2` to `>=3.12,<4` because the old pin resolved to `pyarrow==17.0.0`, which failed to install under Python 3.13 in this environment.
- `pyarrow>=18.0.0` is declared directly to avoid older Python 3.13-incompatible resolution.

## Acceptance Criteria

- `uv lock` succeeds.
- `uv run python -c "import forecast_forge; print('ok')"` succeeds.
- `.python-version` is visible to git.

## Validation

```sh
uv lock
uv run python -c "import forecast_forge; print('ok')"
uv run pytest
```

## Result

- `uv lock` succeeded.
- Package import succeeded through `uv run`.
- `uv run pytest` ran under Python 3.13.13 and collected zero tests.
