# Task 0002: Modernize Local Spark Setup

## Status

Done

## Epic

[Epic 0001: Local UV, Python, and Spark Modernization](../epics/0001-uv-python-spark-modernization.md)

## Objective

Define and validate a local Spark workflow that works with `uv` and Python 3.13.

## Scope

- Add explicit PySpark dependency for local Spark execution without Docker.
- Document Java 17 setup for macOS, Linux, and Windows/WSL.
- Validate `SparkSession.builder.master("local[*]")` execution.
- Decide whether local development should use PySpark from PyPI, downloaded Spark, or Docker only.

## Decision

Local development uses PySpark from PyPI through `uv`. Docker remains the preferred path for testing cluster-like Spark behavior.

## Implementation Notes

- Added `pyspark>=4.1,<4.2` to `pyproject.toml`.
- `uv lock` resolved `pyspark==4.1.1` and `py4j==0.10.9.9`.
- Installed OpenJDK 17 with Homebrew on macOS.
- Homebrew's `openjdk@17` is keg-only, so local commands need `JAVA_HOME=/opt/homebrew/opt/openjdk@17` unless the user's shell or macOS Java wrapper is configured.
- Added `.envrc` for project-local Java 17 configuration with `direnv`.

## Acceptance Criteria

- Local `uv` environment can start a Spark session.
- README documents required Java version and setup commands.
- Validation command succeeds:

```sh
uv run python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[*]').getOrCreate(); spark.range(1).show(); spark.stop()"
```

## Validation

```sh
uv lock
uv run python -c "import pyspark; print(pyspark.__version__)"
JAVA_HOME=/opt/homebrew/opt/openjdk@17 PATH=/opt/homebrew/opt/openjdk@17/bin:$PATH uv run python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[*]').appName('forecast-forge-smoke').getOrCreate(); spark.range(1).show(); spark.stop()"
```

## Result

- `uv lock` succeeded.
- `uv run python -c "import pyspark; print(pyspark.__version__)"` printed `4.1.1`.
- Local Spark smoke test succeeded with Java 17 configured through `JAVA_HOME`.
- `direnv` was installed with Homebrew and `.envrc` was allowed for this repository.
- `direnv exec . sh -c 'echo JAVA_HOME=$JAVA_HOME; java -version'` confirmed Java 17 loads from `.envrc`.
