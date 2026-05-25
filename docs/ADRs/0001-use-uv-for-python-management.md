# ADR 0001: Use UV for Python Management

## Status

Accepted

## Context

The project previously used `setup.py`, `setup.cfg`, and `requirements.txt` for packaging and dependencies. The modernization effort targets Python 3.13 and needs reproducible dependency resolution for local development, tests, and Docker builds.

## Decision

Use `uv` as the primary Python project manager.

Project metadata and dependencies will live in `pyproject.toml`. Locked dependencies will live in `uv.lock`. The project will track `.python-version` so contributors and automation resolve the same Python version by default.

## Consequences

- Local setup becomes:

```sh
uv sync
uv run pytest
```

- Docker can either run `uv sync` directly or install from an exported requirements file generated from `uv.lock`.
- Legacy packaging files may remain temporarily during migration, but `pyproject.toml` is the source of truth.
