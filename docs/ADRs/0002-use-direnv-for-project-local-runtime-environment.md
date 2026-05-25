# ADR 0002: Use Direnv for Project-Local Runtime Environment

## Status

Accepted

## Context

Local PySpark execution needs Java 17. On macOS with Homebrew, `openjdk@17` is keg-only, so it is installed but not exposed as the default system `java`. Setting `JAVA_HOME` globally would affect unrelated projects.

## Decision

Track a repository `.envrc` that sets `JAVA_HOME` and prepends the Java 17 binary directory to `PATH`.

Use `direnv` to load those values only when the shell is inside this repository. The shell hook is installed once globally, while `.envrc` remains project-specific and must be approved per checkout with `direnv allow`.

## Consequences

- Local Spark commands can run with the intended Java version without changing the machine-wide default Java.
- New contributors need to install `direnv`, enable the shell hook, and run `direnv allow`.
- Non-interactive scripts should use `direnv exec . <command>` or set `JAVA_HOME` and `PATH` explicitly.
