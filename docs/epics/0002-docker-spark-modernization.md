# Epic 0002: Docker Spark Modernization

## Status

On hold

## Objective

Modernize and validate the Docker-based Spark cluster so it can run the project with Python 3.13, Spark 4.x, Java 17, and the `uv` project environment.

## Scope

- Update Docker-based Spark cluster setup.
- Validate the Spark master, worker, and history server lifecycle.
- Validate Docker smoke-test submission.
- Validate project forecast submission through the Docker Spark master.
- Keep Docker documentation aligned with the final runtime behavior.

## Tasks

- [ ] [Task 0003: Modernize Docker Spark setup](../tasks/0003-modernize-docker-spark-setup.md)

## Decisions

- [ADR 0001: Use UV for Python management](../ADRs/0001-use-uv-for-python-management.md)

## Notes

This epic is intentionally paused while local development and forecasting pipeline improvements take priority. Task 0003 has static implementation work in place, but it still needs runtime validation with Docker before this epic can move forward.
