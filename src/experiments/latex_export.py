from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _escape_latex(text: str) -> str:
    """Simple escaping of LaTeX special characters in short fields."""
    repl = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "$": "\\$",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "\\": "\\textbackslash{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def generate_latex_tables(timestamp_dir: Path, out_root: Path | None = None) -> Path:
    """Wygeneruj tabelki LaTeX na podstawie summary.csv.

    Struktura wyjścia:
        <repo>/latex_tables/<timestamp_basename>/table_<file>_<algorithm>.tex

        Each table: rows sorted by (time_limit_ms, neighborhood, seed).
        Columns:
            time_limit_ms, neighborhood, seed, cmax_best, lower_bound,
            gap_percent, time_to_best_ms, total_time_ms
    """
    timestamp_dir = Path(timestamp_dir)
    if out_root is None:
        # folder w repo ignorowany przez git (.gitignore)
        out_root = Path("latex_tables")
    out_dir = out_root / timestamp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Brak pliku summary.csv w {timestamp_dir}")

    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return out_dir

    # Grupowanie: (instance_file_stem, algorithm)
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        inst_file = r.get("instance_file", "unknown")
        stem = Path(inst_file).stem
        alg = (r.get("algorithm") or "").lower()
        groups.setdefault((stem, alg), []).append(r)

    for (stem, alg), rs in groups.items():
        # Sort
        def key_fn(x: dict):
            try:
                tl = int(x.get("time_limit_ms") or 0)
            except Exception:
                tl = 0
            try:
                seed = int(x.get("seed") or 0)
            except Exception:
                seed = 0
            neigh = x.get("neighborhood") or ""
            return tl, neigh, seed

        rs.sort(key=key_fn)

        table_filename = out_dir / f"table_{stem}_{alg}.tex"
        with open(table_filename, "w", encoding="utf-8") as fw:
            fw.write(
                "% Auto-generated LaTeX table for instance file='{}' algorithm='{}'\n".format(
                    stem, alg
                )
            )
            fw.write("% Source directory: {}\n".format(timestamp_dir))
            fw.write("\\begin{table}[ht]\\centering\n")
            caption = f"Results for {stem} ({alg})"
            fw.write(f"\\caption{{{_escape_latex(caption)}}}\n")
            fw.write("\\small\n")
            fw.write("\\begin{tabular}{rrrrrrrr}\\hline\n")
            fw.write(
                "tl(ms) & neigh & seed & c$_{max}$ & LB & gap(\\%) & "
                "t$_{best}$(ms) & t$_{tot}$(ms) \\\\ \\hline\n"
            )
            for r in rs:
                tl = r.get("time_limit_ms", "")
                neigh = _escape_latex(r.get("neighborhood", ""))
                seed = r.get("seed", "")
                cmax = r.get("cmax_best", "")
                lb = r.get("lower_bound", "")
                gap = r.get("gap_percent", "")
                if gap not in ("", None):
                    try:
                        gap = f"{float(gap):.2f}"
                    except Exception:
                        pass
                t_best = r.get("time_to_best_ms", "")
                t_tot = r.get("total_time_ms", "")
                # Każdy wiersz musi kończyć się podwójnym \\\\ (poprawka z pojedynczego \\)
                fw.write(
                    f"{tl} & {neigh} & {seed} & {cmax} & {lb} & {gap} & {t_best} & {t_tot} \\\\\n"
                )
            fw.write("\\hline\n\\end{tabular}\n")
            fw.write("\\end{table}\n")
        print(f"[LaTeX] Written {table_filename}")

    return out_dir


def generate_master_table(timestamp_dir: Path, out_root: Path | None = None) -> Path:
    """Generate a single longtable with all rows from summary.csv.

    Columns:
      file, inst, alg, neigh, seed, tl(ms), jobs, mach, cmax, LB, gap(%), t_best(ms), t_tot(ms)
    """
    timestamp_dir = Path(timestamp_dir)
    if out_root is None:
        out_root = Path("latex_tables")
    out_dir = out_root / timestamp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {timestamp_dir}")
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return out_dir / "all_results.tex"
    # Sort for stable ordering

    def key_fn(r: dict):
        try:
            inst_file = r.get("instance_file") or ""
            stem = Path(inst_file).stem
            inst = int(r.get("instance_number") or 0)
            alg = (r.get("algorithm") or "").lower()
            tl = int(r.get("time_limit_ms") or 0)
            neigh = r.get("neighborhood") or ""
            seed = int(r.get("seed") or 0)
        except Exception:
            inst_file = r.get("instance_file") or ""
            stem = Path(inst_file).stem
            inst = 0
            alg = (r.get("algorithm") or "").lower()
            tl = 0
            neigh = r.get("neighborhood") or ""
            seed = 0
        return stem, inst, alg, tl, neigh, seed

    rows.sort(key=key_fn)
    master_path = out_dir / "all_results.tex"
    with open(master_path, "w", encoding="utf-8") as fw:
        fw.write("% Auto-generated master table from summary.csv\n")
        fw.write("% Directory: {}\n".format(timestamp_dir))
        fw.write("\\begin{longtable}{lllllllllllll}\\toprule\n")
        fw.write(
            "file & inst & alg & neigh & seed & tl(ms) & jobs & mach & c$_{max}$ & LB & "
            "gap(\\%) & t$_{best}$(ms) & t$_{tot}$(ms) \\\\ \\midrule\n"
        )
        fw.write("\\endfirsthead\n")
        fw.write(
            "file & inst & alg & neigh & seed & tl(ms) & jobs & mach & c$_{max}$ & LB & "
            "gap(\\%) & t$_{best}$(ms) & t$_{tot}$(ms) \\\\ \\midrule\n"
        )
        fw.write("\\endhead\n")
        for r in rows:
            inst_file = r.get("instance_file") or ""
            stem = _escape_latex(Path(inst_file).stem)
            inst = r.get("instance_number", "")
            alg = _escape_latex((r.get("algorithm") or "").lower())
            neigh = _escape_latex(r.get("neighborhood") or "")
            seed = r.get("seed", "")
            tl = r.get("time_limit_ms", "")
            jobs = r.get("jobs", r.get("instance_jobs", ""))
            mach = r.get("machines", r.get("instance_machines", ""))
            cmax = r.get("cmax_best", "")
            lb = r.get("lower_bound", "")
            gap = r.get("gap_percent", "")
            if gap not in ("", None):
                try:
                    gap = f"{float(gap):.2f}"
                except Exception:
                    pass
            t_best = r.get("time_to_best_ms", "")
            t_tot = r.get("total_time_ms", "")
            # Analogicznie: poprawka na podwójny backslash końca wiersza
            fw.write(
                f"{stem} & {inst} & {alg} & {neigh} & {seed} & {tl} & {jobs} & {mach} & "
                f"{cmax} & {lb} & {gap} & {t_best} & {t_tot} \\\\\n"
            )
        fw.write("\\bottomrule\n\\end{longtable}\n")
    print(f"[LaTeX] Written master table {master_path}")
    return master_path


__all__ = ["generate_latex_tables", "generate_master_table"]


def generate_latex_avg_tables(timestamp_dir: Path, out_root: Path | None = None) -> Path:
    """Generate averaged LaTeX tables per (time_limit_ms, neighborhood, seed) for each
    (instance_file_stem, algorithm).

    Dla każdego pliku instancji (stem) i algorytmu agregujemy wiele instancji (różne
    instance_number) do jednego wiersza przez uśrednienie pól liczbowych:
        cmax_best, lower_bound, gap_percent, time_to_best_ms, total_time_ms.

    Zachowujemy ten sam układ kolumn co w oryginalnych tabelach, żeby łatwo
    porównywać: tl(ms), neigh, seed, c_max, LB, gap(%), t_best(ms), t_tot(ms)

    NOTA: gap(%) jest średnią arytmetyczną pojedynczych gapów, *nie* liczymy jej
    z uśrednionych c_max i LB (świadomy wybór – minimalna ingerencja). Jeśli
    potrzebna byłaby wersja ważona lub rekalkulowana, można dodać parametr.
    """
    timestamp_dir = Path(timestamp_dir)
    if out_root is None:
        out_root = Path("latex_tables")
    out_dir = out_root / timestamp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Brak pliku summary.csv w {timestamp_dir}")

    import csv as _csv

    with open(summary_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return out_dir

    # Grupowanie najpierw po (stem, alg)
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        inst_file = r.get("instance_file", "")
        stem = Path(inst_file).stem
        alg = (r.get("algorithm") or "").lower()
        groups.setdefault((stem, alg), []).append(r)

    for (stem, alg), rs in groups.items():
        # Drugie grupowanie: (time_limit_ms, neighborhood, seed)
        agg: Dict[Tuple[int, str, int], List[dict]] = {}
        for r in rs:
            try:
                tl = int(r.get("time_limit_ms") or 0)
            except Exception:
                tl = 0
            neigh = r.get("neighborhood") or ""
            try:
                seed = int(r.get("seed") or 0)
            except Exception:
                seed = 0
            agg.setdefault((tl, neigh, seed), []).append(r)

        # Posortowana lista kluczy dla deterministycznego wyjścia
        keys = sorted(agg.keys(), key=lambda k: (k[0], k[1], k[2]))

        table_filename = out_dir / f"table_{stem}_{alg}_avg.tex"
        with open(table_filename, "w", encoding="utf-8") as fw:
            fw.write("% Auto-generated AVG LaTeX table (grouped by tl, neighborhood, seed)\n")
            fw.write("% Instance file='{}' algorithm='{}'\n".format(stem, alg))
            fw.write("% Source directory: {}\n".format(timestamp_dir))
            fw.write("% Each row aggregates multiple raw instance rows via arithmetic mean.\n")
            fw.write("\\begin{table}[ht]\\centering\n")
            caption = f"Averaged results for {stem} ({alg})"
            fw.write(f"\\caption{{{_escape_latex(caption)}}}\n")
            fw.write("\\small\n")
            fw.write("\\begin{tabular}{rrrrrrrr}\\hline\n")
            fw.write(
                "tl(ms) & neigh & seed & c$_{max}$ & LB & gap(\\%) & t$_{best}$(ms) & "
                "t$_{tot}$(ms) \\\\ \\hline\n"
            )
            for tl, neigh, seed in keys:
                lst = agg[(tl, neigh, seed)]
                # Akumulacja

                def _nums(field: str):
                    vals = []
                    for item in lst:
                        v = item.get(field)
                        if v in (None, ""):
                            continue
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass
                    return vals

                c_vals = _nums("cmax_best")
                lb_vals = _nums("lower_bound")
                gap_vals = _nums("gap_percent")
                tb_vals = _nums("time_to_best_ms")
                tt_vals = _nums("total_time_ms")

                def avg(vals, default=0.0):
                    return sum(vals) / len(vals) if vals else default

                c_avg = avg(c_vals)
                lb_avg = avg(lb_vals)
                gap_avg = avg(gap_vals)
                tb_avg = avg(tb_vals)
                tt_avg = avg(tt_vals)

                # Formatowanie: c_max i LB jako int (zaokr.), gap 2 miejsca, t_best int, t_tot int
                c_fmt = str(int(round(c_avg)))
                lb_fmt = str(int(round(lb_avg)))
                gap_fmt = f"{gap_avg:.2f}" if gap_vals else ""
                tb_fmt = str(int(round(tb_avg))) if tb_vals else ""
                tt_fmt = str(int(round(tt_avg))) if tt_vals else ""

                fw.write(
                    f"{tl} & {_escape_latex(neigh)} & {seed} & {c_fmt} & {lb_fmt} & {gap_fmt} & "
                    f"{tb_fmt} & {tt_fmt} \\\\\n"
                )
            fw.write("\\hline\n\\end{tabular}\n")
            fw.write("\\end{table}\n")
        print(f"[LaTeX][AVG] Written {table_filename}")

    return out_dir


def generate_master_avg_table(timestamp_dir: Path, out_root: Path | None = None) -> Path:
    """Generate a longtable with averages over instance_number for each
    (file stem, algorithm, neighborhood, time_limit_ms, seed).

    Kolumny:
      file, alg, neigh, seed, tl(ms), jobs, mach, c$_{max}$, LB, gap(%), t_best(ms), t_tot(ms), n
    gdzie n = liczba rekordów (instancji) uśrednianych.

    Gap średni liczony jako średnia arytmetyczna pojedynczych gapów (spójnie
    z per-file avg). c_max, LB, t_best, t_tot także średnie z zaokrągleniem.
    """
    timestamp_dir = Path(timestamp_dir)
    if out_root is None:
        out_root = Path("latex_tables")
    out_dir = out_root / timestamp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = timestamp_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {timestamp_dir}")
    import csv as _csv

    with open(summary_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return out_dir / "all_results_avg.tex"

    from collections import defaultdict

    # Group by (stem, alg, neigh, tl, seed)
    grouped = defaultdict(list)
    for r in rows:
        inst_file = r.get("instance_file") or ""
        stem = Path(inst_file).stem
        alg = (r.get("algorithm") or "").lower()
        neigh = r.get("neighborhood") or ""
        try:
            tl = int(r.get("time_limit_ms") or 0)
        except Exception:
            tl = 0
        try:
            seed = int(r.get("seed") or 0)
        except Exception:
            seed = 0
        grouped[(stem, alg, neigh, tl, seed)].append(r)

    def _num(x):
        try:
            return float(x)
        except Exception:
            return None

    aggregates = []
    for key, lst in grouped.items():
        stem, alg, neigh, tl, seed = key
        c_vals = [_num(r.get("cmax_best")) for r in lst if _num(r.get("cmax_best")) is not None]
        lb_vals = [
            _num(r.get("lower_bound")) for r in lst if _num(r.get("lower_bound")) is not None
        ]
        gap_vals = [
            _num(r.get("gap_percent")) for r in lst if _num(r.get("gap_percent")) is not None
        ]
        tb_vals = [
            _num(r.get("time_to_best_ms"))
            for r in lst
            if _num(r.get("time_to_best_ms")) is not None
        ]
        tt_vals = [
            _num(r.get("total_time_ms")) for r in lst if _num(r.get("total_time_ms")) is not None
        ]
        jobs_vals = [
            _num(r.get("jobs") or r.get("instance_jobs"))
            for r in lst
            if _num(r.get("jobs") or r.get("instance_jobs")) is not None
        ]
        mach_vals = [
            _num(r.get("machines") or r.get("instance_machines"))
            for r in lst
            if _num(r.get("machines") or r.get("instance_machines")) is not None
        ]

        def avg(v):
            return sum(v) / len(v) if v else None

        c_avg = avg(c_vals)
        lb_avg = avg(lb_vals)
        gap_avg = avg(gap_vals)
        tb_avg = avg(tb_vals)
        tt_avg = avg(tt_vals)
        jobs_avg = avg(jobs_vals)
        mach_avg = avg(mach_vals)
        n = len(lst)
        aggregates.append(
            {
                "stem": stem,
                "alg": alg,
                "neigh": neigh,
                "tl": tl,
                "seed": seed,
                "c": c_avg,
                "lb": lb_avg,
                "gap": gap_avg,
                "tb": tb_avg,
                "tt": tt_avg,
                "jobs": jobs_avg,
                "mach": mach_avg,
                "n": n,
            }
        )

    # Sort for stable output
    aggregates.sort(key=lambda x: (x["stem"], x["alg"], x["tl"], x["neigh"], x["seed"]))

    master_avg_path = out_dir / "all_results_avg.tex"
    with open(master_avg_path, "w", encoding="utf-8") as fw:
        fw.write("% Auto-generated averaged master table from summary.csv\n")
        fw.write("% Directory: {}\n".format(timestamp_dir))
        fw.write("\\begin{longtable}{lllllllllllll}\\toprule\n")
        fw.write(
            "file & alg & neigh & seed & tl(ms) & jobs & mach & c$_{max}$ & LB & gap(\\%) & "
            "t$_{best}$(ms) & t$_{tot}$(ms) & n \\\\ \\midrule\n"
        )
        fw.write("\\endfirsthead\n")
        fw.write(
            "file & alg & neigh & seed & tl(ms) & jobs & mach & c$_{max}$ & LB & gap(\\%) & "
            "t$_{best}$(ms) & t$_{tot}$(ms) & n \\\\ \\midrule\n"
        )
        fw.write("\\endhead\n")
        for a in aggregates:

            def fmt(v, is_int=False, digits=2):
                if v is None:
                    return ""
                if is_int:
                    return str(int(round(v)))
                return f"{v:.{digits}f}"

            fw.write(
                f"{_escape_latex(a['stem'])} & {_escape_latex(a['alg'])} & "
                f"{_escape_latex(a['neigh'])} & {a['seed']} & {a['tl']} & "
                f"{fmt(a['jobs'], is_int=True)} & {fmt(a['mach'], is_int=True)} & "
                f"{fmt(a['c'], is_int=True)} & {fmt(a['lb'], is_int=True)} & {fmt(a['gap'])} & "
                f"{fmt(a['tb'], is_int=True)} & {fmt(a['tt'], is_int=True)} & {a['n']} \\\n"
            )
        fw.write("\\bottomrule\n\\end{longtable}\n")
    print(f"[LaTeX][AVG] Written master average table {master_avg_path}")
    return master_avg_path
