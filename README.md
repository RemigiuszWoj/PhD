# PhD JSSP Workspace

Minimal skeleton for Job Shop Scheduling Problem experiments (Taillard instances initially).

## Environment setup

Recommended: local virtual environment pinned to project directory.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\\Scripts\\activate       # Windows PowerShell
python -m pip install --upgrade pip
```

Install development dependencies (pytest etc.) if not already present:

```bash
python -m pip install pytest
```

If you prefer an explicit requirements file later, create `requirements-dev.txt` and freeze:

```bash
python -m pip freeze > requirements-dev.txt
```

Deactivate environment when done:

```bash
deactivate
```

## Running tests

Pytest is configured via `pyproject.toml` with options:

```
-ra  # show summary of skipped/xfailed/xpassed/failed
-vv  # verbose test names
--durations=5 --durations-min=0.05  # slow test report
--color=yes
```

Basic run:

```bash
pytest
```

Quiet run:

```bash
pytest -q
```

Module invocation (equivalent):

```bash
python -m pytest
```

Run only tests matching substring:

```bash
pytest -k parser
```

Stop after first failure:

```bash
pytest -x
```

## Custom test summary

A custom `pytest_terminal_summary` hook (see `tests/conftest.py`) prints a compact counts line at the end, plus the slowest tests.

## Parser

Implemented Taillard parser (`src/parser.py`) producing `DataInstance` (`src/models.py`) with automatic normalization of 1-based machine indices.

