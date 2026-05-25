# Epic 0001: Local UV, Python, and Spark Modernization

## Status

Done

## Objective

Modernize Python dependency management, local development, and Spark runtime setup so the project can run consistently on Python 3.13 for local development.

## Scope

- Migrate packaging and dependency management to `uv`.
- Target Python 3.13 for local development.
- Rework Spark setup around a modern Spark line compatible with Python 3.13.
- Refresh README setup and validation instructions.

## Tasks

- [x] [Task 0001: Migrate packaging to UV](../tasks/0001-migrate-packaging-to-uv.md)
- [x] [Task 0002: Modernize local Spark setup](../tasks/0002-modernize-local-spark-setup.md)
- [x] [Task 0004: Update developer documentation](../tasks/0004-update-developer-documentation.md)
- [x] [Task 0005: Add root-level Makefile for project automation](../tasks/0005-add-root-level-makefile.md)

## Decisions

- [ADR 0001: Use UV for Python management](../ADRs/0001-use-uv-for-python-management.md)
- [ADR 0002: Use Direnv for project-local runtime environment](../ADRs/0002-use-direnv-for-project-local-runtime-environment.md)

## Notes

Current repository state before this epic used legacy `setup.py`/`setup.cfg` metadata, `requirements.txt`, Python 3.11 metadata, and Docker Spark assets based on Python 3.10, Spark 3.3.3, and Java 11.

Docker Spark modernization was moved to [Epic 0002](0002-docker-spark-modernization.md) and is intentionally paused while local pipeline development takes priority.
