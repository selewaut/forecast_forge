# Task 0005: Add Root-Level Makefile for Project Automation

## Status

Done

## Epic

[Epic 0001: Local UV, Python, and Spark Modernization](../epics/0001-uv-python-spark-modernization.md)

## Objective

A new developer can clone the repo, run `make setup && make data && make check`, and be ready to run forecasts — without reading multi-page docs first.

## Scope

- Add a root `Makefile` with targets for the common local workflow.
- Include prerequisite checks (Java 17, Kaggle API key, data files).
- Add a data download + verification target.
- Keep targets composable and documented with `##` comments (so `make help` works).
- Match the existing pattern used in `spark-setup/Makefile`.

## Non-Goals

- No CI/CD integration — this is purely local developer experience.
- No changes to the Docker Spark workflow.
- No changes to Python packaging or configuration.

## Acceptance Criteria

- `make setup` runs `uv sync` and confirms the venv is ready.
- `make data` downloads the Kaggle dataset (if missing) and verifies all 4 CSV files exist.
- `make check` validates Java 17, Kaggle API credentials, and data presence with clear error messages.
- `make run` runs the forecast pipeline via `spark-submit --master local[*]`.
- `make clean` removes generated output directories and cached data.
- `make help` lists all targets with descriptions.
- Each target is idempotent — running it twice is safe.
- `spark-setup/Makefile` targets remain unchanged and continue to work.

## Validation

```sh
make check          # all green
make setup          # uv sync succeeds
make data           # downloads data/ directory
make clean          # removes generated output
make run            # forecast completes without error
make help           # prints documented targets
```

## Documentation Updates

- README updated with quick-start section using `make` commands.
- `make help` output serves as built-in documentation.

## Notes

- `config.py` currently hardcodes `data/walmart_sales_forecasting/` — the Makefile should match this path.
- Kaggle download supports the current `~/.kaggle/access_token` flow and legacy `~/.kaggle/kaggle.json` credentials.
- The Makefile should be thin — delegate logic to short shell commands and scripts rather than inlining complex logic.
