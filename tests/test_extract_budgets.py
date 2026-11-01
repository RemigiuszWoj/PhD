"""Budget extraction from the move log (scripts/extract_budgets.py)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("extract_budgets", ROOT / "scripts" / "extract_budgets.py")
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)


def _run(tl, move_log, initial=1000, lb=900, seed=0):
    best_hist, t_hist, best = [initial], [0], initial
    for end, b in move_log:
        if b < best:
            best = b
            best_hist.append(b)
            t_hist.append(end)
    return {
        "config": {"algorithm": "ils", "neighborhood": "kbonacci_k3", "instance_file": "data/tai20_5.txt",
                   "instance_number": 0, "seed": seed, "time_limit_ms": tl},
        "cmax_best": best, "lower_bound": lb, "upper_bound": 950, "move_log": move_log,
        "cmax_history": best_hist, "time_history_ms": t_hist,
    }


def test_move_started_before_budget_counts():
    r = _run(10000, [[40, 990], [90, 980], [130, 970], [400, 960]])
    assert eb.value_at(r, 100)["cmax_best"] == 970      # 3rd move started at 90 < 100, ended at 130
    assert eb.value_at(r, 100)["iterations"] == 3
    assert eb.value_at(r, 90)["cmax_best"] == 980       # 3rd move starts at 90: not before 90
    assert eb.value_at(r, 30)["cmax_best"] == 990       # first move always starts at 0
    assert eb.value_at(r, 10000)["cmax_best"] == 960


def _write(d: Path, name: str, r: dict):
    (d / name).mkdir(parents=True)
    (d / name / "result.json").write_text(json.dumps(r))


def test_short_independent_runs_are_not_used_as_sources(tmp_path):
    src, out = tmp_path / "validate", tmp_path / "out"
    base = "algo=ils__neigh=kbonacci_k3__file=tai20_5__inst=0__n20__m5"
    _write(src, f"{base}__tl=10000ms__seed=0", _run(10000, [[50, 990], [150, 950], [9000, 940]]))
    # independent short runs with DIFFERENT values: they must not overwrite the extraction
    _write(src, f"{base}__tl=100ms__seed=0", _run(100, [[60, 999]]))
    _write(src, f"{base}__tl=500ms__seed=0", _run(500, [[60, 998], [300, 997]]))
    eb.extract(src, out, [100, 500, 10000])
    got = {int(p.parent.name.split("__tl=")[1].split("ms")[0]): json.loads(p.read_text())["cmax_best"]
           for p in out.rglob("result.json")}
    # 500 ms: the 3rd move starts at 150 ms < 500 ms, so it counts although it ends at 9000 ms
    assert got == {100: 950, 500: 940, 10000: 940}


def test_compare_skips_only_the_source_run(tmp_path, capsys):
    base = "algo=ils__neigh=kbonacci_k3__file=tai20_5__inst=0__n20__m5"
    src, out, old = tmp_path / "src", tmp_path / "out", tmp_path / "old"
    for seed in range(6):
        _write(src, f"{base}__tl=10000ms__seed={seed}", _run(10000, [[50, 990], [150, 950]], seed=seed))
        # an "old campaign" independent 10 s run with the same key but another value
        _write(old, f"{base}__tl=10000ms__seed={seed}", _run(10000, [[50, 990], [150, 940]], seed=seed))
    eb.extract(src, out, [10000])
    eb.compare(out, src)            # validation-style: only the source runs -> no pairs
    assert "no comparable pairs" in capsys.readouterr().out
    eb.compare(out, old)            # replication-style: independent old runs are compared
    printed = capsys.readouterr().out
    assert "10000" in printed and "no comparable pairs" not in printed
