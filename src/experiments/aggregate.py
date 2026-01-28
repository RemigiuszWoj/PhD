from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_results_dir(timestamp_dir: Path) -> List[Dict[str, Any]]:
    """Recursively load all JSON result files under a timestamp directory.

    Supports new per-run subdirectory layout where each run is stored in its own folder
    containing result.json (but we also accept legacy flat *.json files).
    """
    results: List[Dict[str, Any]] = []
    # First, legacy flat files
    for file in timestamp_dir.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"[Aggregate] Failed to load {file}: {e}")
    # Recursive search for per-run result.json
    for file in timestamp_dir.rglob("result.json"):
        # Avoid double-loading if flat style also named result.json at root (unlikely)
        if file.parent == timestamp_dir:
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"[Aggregate] Failed to load {file}: {e}")
    return results


def write_summary_csv(timestamp_dir: Path) -> Path:
    rows = load_results_dir(timestamp_dir)
    if not rows:
        print("[Aggregate] No result files found to summarize.")
        return timestamp_dir / "summary.csv"
    out_path = timestamp_dir / "summary.csv"
    # Determine columns
    columns = [
        "algorithm",
        "neighborhood",
        "instance_file",
        "instance_number",
        "jobs",
        "machines",
        "seed",
        "time_limit_ms",
        "cmax_best",
        "lower_bound",
        "gap_percent",
        "time_to_best_ms",
        "total_time_ms",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in rows:
            cfg = r["config"]
            writer.writerow(
                [
                    cfg.get("algorithm"),
                    cfg.get("neighborhood"),
                    cfg.get("instance_file"),
                    cfg.get("instance_number"),
                    r.get("instance_jobs"),
                    r.get("instance_machines"),
                    cfg.get("seed"),
                    cfg.get("time_limit_ms"),
                    r.get("cmax_best"),
                    r.get("lower_bound"),
                    r.get("gap_percent"),
                    r.get("time_to_best_ms"),
                    r.get("total_time_ms"),
                    # ensure alignment (time_limit_ms already earlier)
                ]
            )
    print(f"[Aggregate] Summary written: {out_path}")
    return out_path


def write_wide_gap_table(timestamp_dir: Path, summary_path: Path | None = None) -> Path:
    """Create a wide table with one row per (instance_file, instance_number, time_limit_ms)
    and columns:
        instance_file, instance_number, time_limit_ms,
        tabu_adjacent, tabu_quantum_adjacent, tabu_quantum_fibonahi, tabu_fibonahi,
        tabu_dynasearch, tabu_motzkin,
        sa_adjacent, sa_quantum_adjacent, sa_quantum_fibonahi, sa_fibonahi,
        sa_dynasearch, sa_motzkin

    Values = best (minimal) gap_percent over seeds for that
    (algorithm, neighborhood, instance, tl_ms).
    If gap unavailable -> blank.
    """
    # Load summary (create if absent)
    if summary_path is None:
        summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        summary_path = write_summary_csv(timestamp_dir)

    # Read rows
    import csv as _csv

    records: list[dict[str, str]] = []
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            records.append(row)
    if not records:
        print("[Aggregate] No records for wide table.")
        out = timestamp_dir / "wide_summary.csv"
        with open(out, "w", encoding="utf-8", newline="") as fw:
            fw.write(
                "instance_number,tabu_adjacent,tabu_quantum_adjacent,tabu_quantum_fibonahi,"
                "tabu_fibonahi,tabu_dynasearch,tabu_motzkin,sa_adjacent,sa_quantum_adjacent,"
                "sa_quantum_fibonahi,sa_fibonahi,sa_dynasearch,sa_motzkin\n"
            )
        return out

    # Group best gap per (alg, neigh, inst, tl)
    from math import inf

    best: dict[tuple[str, str, str, int, int], float] = {}
    for r in records:
        inst_file = r.get("instance_file") or ""
        try:
            inst = int(r["instance_number"])
        except Exception:
            continue
        try:
            tl = int(r.get("time_limit_ms")) if r.get("time_limit_ms") not in (None, "") else None
        except Exception:
            tl = None
        if tl is None:
            continue
        alg = r["algorithm"].lower()
        neigh = r["neighborhood"]
        gap_raw = r.get("gap_percent")
        try:
            gap = float(gap_raw) if gap_raw not in ("", None) else inf
        except ValueError:
            gap = inf
        key = (alg, neigh, inst_file, inst, tl)
        if key not in best or gap < best[key]:
            best[key] = gap

    # Collect rows keys (instance_number, time_limit_ms)
    inst_tl_pairs = sorted(
        {
            (r.get("instance_file"), int(r["instance_number"]), int(r["time_limit_ms"]))
            for r in records
            if r.get("instance_number") not in (None, "")
            and r.get("time_limit_ms") not in (None, "")
        }
    )

    columns = [
        "instance_file",
        "instance_number",
        "time_limit_ms",
        "tabu_adjacent",
        "tabu_quantum_adjacent",
        "tabu_quantum_fibonahi",
        "tabu_fibonahi",
        "tabu_dynasearch",
        "tabu_motzkin",
        "sa_adjacent",
        "sa_quantum_adjacent",
        "sa_quantum_fibonahi",
        "sa_fibonahi",
        "sa_dynasearch",
        "sa_motzkin",
    ]
    out_path = timestamp_dir / "wide_summary.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for inst_file, inst, tl in inst_tl_pairs:

            def fmt(alg: str, neigh: str) -> str:
                val = best.get((alg, neigh, inst_file, inst, tl))
                if val is None or val == float("inf"):
                    return ""
                return f"{val:.4f}"

            row = [
                inst_file,
                inst,
                tl,
                fmt("tabu", "adjacent"),
                fmt("tabu", "quantum_adjacent"),
                fmt("tabu", "quantum_fibonahi"),
                fmt("tabu", "fibonahi"),
                fmt("tabu", "dynasearch"),
                fmt("tabu", "motzkin"),
                fmt("sa", "adjacent"),
                fmt("sa", "quantum_adjacent"),
                fmt("sa", "quantum_fibonahi"),
                fmt("sa", "fibonahi"),
                fmt("sa", "dynasearch"),
                fmt("sa", "motzkin"),
            ]
            w.writerow(row)
    print(f"[Aggregate] Wide summary written: {out_path}")
    return out_path


def write_matrix_gap_table(timestamp_dir: Path, summary_path: Path | None = None) -> Path:
    """Create a matrix-style CSV showing GAP dla KAŻDEGO odpalenia (seed) zamiast minimum.

    Nowy format (per-run):
        - Każdy wiersz: (instance_number, seed, TABU:3 kolumny, SA:3 kolumny,
          best_cmax_seed, lower_bound, gap_best_seed)
        - Kolumny TABU/SA zawierają gap_percent dla dokładnego (alg, neigh, seed) runu.
        - best_cmax_seed = minimum cmax_best w tym (instance, seed) po wszystkich alg+neigh.
        - gap_best_seed = (best_cmax_seed - lower_bound)/lower_bound*100 jeśli lower_bound dostępny.

    Różnica vs poprzednia wersja: brak agregacji po seed; zamiast best per inst mamy pełny przekrój.
    """
    if summary_path is None:
        summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        summary_path = write_summary_csv(timestamp_dir)

    import csv as _csv

    records: list[dict[str, str]] = []
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"[Aggregate] Failed reading summary for matrix table: {e}")
    if not records:
        out = timestamp_dir / "matrix_summary.csv"
        with open(out, "w", encoding="utf-8", newline="") as fw:
            fw.write(
                "instance_number,tabu,tabu,tabu,tabu,sa,sa,sa,sa,"
                + "best_cmax,lower_bound,best_gap_percent\n"
            )
            fw.write(
                ",adjacent,fibonahi_neighborhood,dynasearch_neighborhood,"
                + "motzkin_neighborhood,adjacent,fibonahi_neighborhood,"
                + "dynasearch_neighborhood,motzkin_neighborhood,,,,\n"
            )
        return out

    # Zgrupuj dane per (instance_number, seed)
    from collections import defaultdict

    # Structure: gaps[(inst, seed, tl)][(alg, neigh)] = gap
    gaps: dict[tuple[str, int, int, int], dict[tuple[str, str], float]] = defaultdict(dict)
    cmax_map: dict[tuple[int, int, str, str], int] = {}
    lower_bounds: dict[int, int] = {}

    for r in records:
        try:
            inst_file = r.get("instance_file") or ""
            inst = int(r.get("instance_number"))
        except Exception:
            continue
        try:
            seed = int(r.get("seed"))
        except Exception:
            # jeśli brak seed w summary, traktuj jako 0
            seed = 0
        try:
            tl = int(r.get("time_limit_ms")) if r.get("time_limit_ms") not in (None, "") else None
        except Exception:
            tl = None
        if tl is None:
            continue
        alg = (r.get("algorithm") or "").lower()
        neigh = r.get("neighborhood") or ""
        gap_raw = r.get("gap_percent")
        try:
            gap_val = float(gap_raw) if gap_raw not in ("", None) else None
        except ValueError:
            gap_val = None
        if gap_val is not None:
            gaps[(inst_file, inst, seed, tl)][(alg, neigh)] = gap_val
        # cmax dla ustalenia best_cmax_seed
        cmax_raw = r.get("cmax_best")
        try:
            cmax_val = int(cmax_raw) if cmax_raw not in ("", None) else None
        except ValueError:
            cmax_val = None
        if cmax_val is not None:
            cmax_map[(inst, seed, alg, neigh)] = cmax_val
        # lower bound (stały per inst)
        lb_raw = r.get("lower_bound")
        try:
            lb_val = int(lb_raw) if lb_raw not in ("", None) else None
        except ValueError:
            lb_val = None
        if lb_val is not None and inst not in lower_bounds:
            lower_bounds[inst] = lb_val

    # Przygotuj wiersze
    keys = sorted(gaps.keys())  # (file, inst, seed, tl)
    out_path = timestamp_dir / "matrix_summary.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance_file",
                "instance_number",
                "time_limit_ms",
                "seed",
                "tabu_adjacent",
                "tabu_fibonahi_neighborhood",
                "tabu_dynasearch_neighborhood",
                "tabu_motzkin_neighborhood",
                "sa_adjacent",
                "sa_fibonahi_neighborhood",
                "sa_dynasearch_neighborhood",
                "sa_motzkin_neighborhood",
            ]
        )

        def get_gap(d, alg, neigh):
            val = d.get((alg, neigh))
            if val is None:
                return ""
            return f"{val:.4f}"

        for inst_file, inst, seed, tl in keys:
            per = gaps[(inst_file, inst, seed, tl)]
            w.writerow(
                [
                    inst_file,
                    inst,
                    tl,
                    seed,
                    get_gap(per, "tabu", "adjacent"),
                    get_gap(per, "tabu", "fibonahi_neighborhood"),
                    get_gap(per, "tabu", "dynasearch_neighborhood"),
                    get_gap(per, "tabu", "motzkin_neighborhood"),
                    get_gap(per, "sa", "adjacent"),
                    get_gap(per, "sa", "fibonahi_neighborhood"),
                    get_gap(per, "sa", "dynasearch_neighborhood"),
                    get_gap(per, "sa", "motzkin_neighborhood"),
                ]
            )
    print(f"[Aggregate] Matrix per-run summary written: {out_path}")
    return out_path


def write_matrix_per_seed_table(timestamp_dir: Path, summary_path: Path | None = None) -> Path:
    """Create a detailed matrix with relative error (gap_percent) for each
    algorithm / neighborhood / seed.

    Format columns:
        instance_number, seed, tabu_adjacent, tabu_fibonahi, tabu_dynasearch,
        sa_adjacent, sa_fibonahi, sa_dynasearch, best_cmax_seed,
        lower_bound_seed, gap_best_seed

    Where:
        - Each row corresponds to one (instance_number, seed).
        - Columns with algorithm/neighborhood hold the gap_percent for that exact run.
        - best_cmax_seed = minimal cmax across all alg+neigh for that (instance, seed).
        - gap_best_seed = (best_cmax_seed - lower_bound)/lower_bound*100 if LB present.
    """
    if summary_path is None:
        summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        summary_path = write_summary_csv(timestamp_dir)

    import csv as _csv

    records: list[dict[str, str]] = []
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"[Aggregate] Failed reading summary for per-seed matrix: {e}")
    if not records:
        out = timestamp_dir / "matrix_per_seed.csv"
        with open(out, "w", encoding="utf-8", newline="") as fw:
            fw.write(
                "instance_number,seed,tabu_adjacent,tabu_fibonahi,tabu_dynasearch,"
                "sa_adjacent,sa_fibonahi,sa_dynasearch,best_cmax_seed,lower_bound_seed,"
                "gap_best_seed\n"
            )
        return out

    # Organize per (inst, seed)
    from math import inf

    inst_seed_gaps: dict[tuple[str, int, int, int, str, str], float] = {}
    inst_seed_cmax: dict[tuple[str, int, int, int, str, str], int] = {}
    lower_bounds: dict[int, int] = {}

    for r in records:
        try:
            inst_file = r.get("instance_file") or ""
            inst = int(r.get("instance_number"))
        except Exception:
            continue
        try:
            seed = int(r.get("seed", 0))
        except Exception:
            seed = 0
        try:
            tl = int(r.get("time_limit_ms")) if r.get("time_limit_ms") not in (None, "") else None
        except Exception:
            tl = None
        if tl is None:
            continue
        alg = (r.get("algorithm") or "").lower()
        neigh = r.get("neighborhood") or ""
        gap_raw = r.get("gap_percent")
        try:
            gap_val = float(gap_raw) if gap_raw not in ("", None) else inf
        except ValueError:
            gap_val = inf
        cmax_raw = r.get("cmax_best")
        try:
            cmax_val = int(cmax_raw) if cmax_raw not in ("", None) else None
        except ValueError:
            cmax_val = None
        lb_raw = r.get("lower_bound")
        try:
            lb_val = int(lb_raw) if lb_raw not in ("", None) else None
        except ValueError:
            lb_val = None
        if lb_val is not None and inst not in lower_bounds:
            lower_bounds[inst] = lb_val
        key = (inst_file, inst, seed, tl, alg, neigh)
        inst_seed_gaps[key] = gap_val
        if cmax_val is not None:
            inst_seed_cmax[(inst_file, inst, seed, tl, alg, neigh)] = cmax_val
    inst_seed_tls = sorted({(k[0], k[1], k[2], k[3]) for k in inst_seed_gaps.keys()})

    out_path = timestamp_dir / "matrix_per_seed.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "instance_file",
                "instance_number",
                "time_limit_ms",
                "seed",
                "tabu_adjacent",
                "tabu_fibonahi",
                "tabu_dynasearch",
                "tabu_motzkin",
                "sa_adjacent",
                "sa_fibonahi",
                "sa_dynasearch",
                "sa_motzkin",
                "best_cmax_seed",
                "lower_bound_seed",
                "gap_best_seed",
            ]
        )

        def gap_fmt(inst_file: str, inst: int, seed: int, tl: int, alg: str, neigh: str) -> str:
            val = inst_seed_gaps.get((inst_file, inst, seed, tl, alg, neigh))
            if val is None or val == float("inf"):
                return ""
            return f"{val:.4f}"

        for inst_file, inst, seed, tl in inst_seed_tls:
            # best cmax across all alg/neigh for this inst+seed
            best_c = None
            for alg in ("tabu", "sa"):
                for neigh in (
                    "adjacent",
                    "fibonahi_neighborhood",
                    "dynasearch_neighborhood",
                    "motzkin_neighborhood",
                ):
                    c = inst_seed_cmax.get((inst_file, inst, seed, tl, alg, neigh))
                    if c is not None and (best_c is None or c < best_c):
                        best_c = c
            lb = lower_bounds.get(inst)
            if best_c is not None and lb is not None and lb > 0:
                gap_best_seed = (best_c - lb) / lb * 100.0
                gap_best_seed_str = f"{gap_best_seed:.4f}"
            else:
                gap_best_seed_str = ""
            w.writerow(
                [
                    inst_file,
                    inst,
                    tl,
                    seed,
                    gap_fmt(inst_file, inst, seed, tl, "tabu", "adjacent"),
                    gap_fmt(inst_file, inst, seed, tl, "tabu", "fibonahi_neighborhood"),
                    gap_fmt(inst_file, inst, seed, tl, "tabu", "dynasearch_neighborhood"),
                    gap_fmt(inst_file, inst, seed, tl, "tabu", "motzkin_neighborhood"),
                    gap_fmt(inst_file, inst, seed, tl, "sa", "adjacent"),
                    gap_fmt(inst_file, inst, seed, tl, "sa", "fibonahi_neighborhood"),
                    gap_fmt(inst_file, inst, seed, tl, "sa", "dynasearch_neighborhood"),
                    gap_fmt(inst_file, inst, seed, tl, "sa", "motzkin_neighborhood"),
                    best_c if best_c is not None else "",
                    lb if lb is not None else "",
                    gap_best_seed_str,
                ]
            )
    print(f"[Aggregate] Per-seed matrix written: {out_path}")
    return out_path
