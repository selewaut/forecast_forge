# Task 0004: Update Developer Documentation

## Status

Planned

## Epic

[Epic 0001: UV, Python, and Spark Modernization](../epics/0001-uv-python-spark-modernization.md)

## Objective

Refresh developer-facing setup and run instructions after the UV and Spark modernization work is complete.

## Scope

- Update README installation instructions to use `uv`.
- Document local Python/PySpark workflow.
- Document Docker Spark cluster workflow.
- Replace Java 8 instructions with Java 17 instructions.
- Clarify macOS, Linux, and Windows/WSL setup paths.
- Document validation commands.
- Ensure docs reflect the project-local `.envrc` and `direnv` setup.
- Ensure task docs record implementation results for Tasks 0001 through 0003.

## Acceptance Criteria

- README has current setup instructions.
- README has local and Docker Spark execution examples.
- Old `pip install -r requirements.txt`-first workflow is removed or clearly marked legacy.
- `docs/README.md` explains when implementation tasks must update setup docs.
