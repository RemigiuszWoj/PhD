#!/usr/bin/env bash
set -euo pipefail

# Run full local quality gate sequentially.
# Usage: ./scripts/check_all.sh [--no-mypy] [--no-flake8] [--no-ruff] [--no-tests] [--no-format]
# Returns non‑zero exit on first failure (default Bash -e behavior).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv311/bin/python" ]]; then
    PYTHON=".venv311/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="$(command -v python || command -v python3)"
  fi
fi

run() { echo -e "\n[STEP] $*"; eval "$*"; }

DO_BLACK=1; DO_RUFF=1; DO_FLAKE8=1; DO_MYPY=1; DO_TESTS=1
for arg in "$@"; do
  case "$arg" in
    --no-format) DO_BLACK=0 ;;
    --no-ruff) DO_RUFF=0 ;;
    --no-flake8) DO_FLAKE8=0 ;;
    --no-mypy) DO_MYPY=0 ;;
    --no-tests) DO_TESTS=0 ;;
  esac
  shift || true
done

if [[ $DO_BLACK -eq 1 ]]; then
  run "$PYTHON -m black --check --diff ."
fi
if [[ $DO_RUFF -eq 1 ]]; then
  run "$PYTHON -m ruff check ."
fi
if [[ $DO_FLAKE8 -eq 1 ]]; then
  run "$PYTHON -m flake8 src tests"
fi
if [[ $DO_MYPY -eq 1 ]]; then
  run "$PYTHON -m mypy src tests"
fi
if [[ $DO_TESTS -eq 1 ]]; then
  run "$PYTHON -m pytest"
fi

echo -e "\nAll selected checks passed."
