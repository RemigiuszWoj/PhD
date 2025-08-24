"""Pytest configuration & custom summary hook.

Also ensures 'src' directory (src layout) is on sys.path for imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so 'import src.*' works
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:  # noqa: D401
    """Append a compact custom summary at the end of test session."""
    stats = terminalreporter.stats
    collected = terminalreporter._numcollected  # type: ignore[attr-defined]
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    errors = len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    xfailed = len(stats.get("xfailed", []))
    xpassed = len(stats.get("xpassed", []))

    terminalreporter.section("Custom summary", sep="=")
    terminalreporter.write_line(
        "Collected: "
        f"{collected} | Passed: {passed} | Failed: {failed} | "
        f"Errors: {errors} | Skipped: {skipped} | "
        f"xfailed: {xfailed} | xpassed: {xpassed}"
    )
    if failed:
        terminalreporter.write_line("Failed tests:")
        for rep in stats["failed"]:
            terminalreporter.write_line(f"  - {rep.nodeid}")
