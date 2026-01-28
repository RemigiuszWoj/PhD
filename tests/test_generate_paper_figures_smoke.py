from __future__ import annotations

from pathlib import Path

from scripts.generate_paper_figures import read_summary_csv


def test_read_summary_csv_smoke() -> None:
    summary = Path("results/experiments/20251206_181012/summary.csv")
    assert summary.exists(), "expected test data to exist in repo"

    rows = read_summary_csv(summary)
    assert rows, "should parse at least one row"

    # Basic schema sanity
    r0 = rows[0]
    assert r0.algorithm in {"tabu", "sa"}
    assert r0.time_limit_ms > 0
    assert r0.cmax_best > 0
