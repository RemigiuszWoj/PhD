from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_RESULTS = Path("results/experiments")
LATEX_DIR = Path("latex_tables")


def find_latest_results_dir() -> Optional[Path]:
    if not BASE_RESULTS.exists():
        return None
    dirs = [d for d in BASE_RESULTS.iterdir() if d.is_dir()]
    if not dirs:
        return None
    # sort by name (timestamps like 20251206_181012)
    return sorted(dirs)[-1]


def load_summary(timestamp_dir: Path) -> List[dict]:
    summary = timestamp_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Brak pliku summary.csv w {timestamp_dir}")
    with summary.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


def build_motzkin_pivot(rows: List[dict]) -> Dict[int, List[List[str]]]:
    """Zbuduj dane pivot dla Motzkin: klucze to tl_ms; wartości to wiersze CSV.

    Każdy wiersz: file, n, m, tl_ms, TABU_gap_avg, SA_gap_avg (wszystko sformatowane).
    """
    # Filter only motzkin rows
    mot_rows = [r for r in rows if (r.get("neighborhood") or "") == "motzkin_neighborhood"]
    if not mot_rows:
        return {}

    # Group by (stem, n, m, tl_ms, alg)
    from collections import defaultdict

    grouped: Dict[Tuple[str, int, int, int, str], List[float]] = defaultdict(list)
    for r in mot_rows:
        inst_file = r.get("instance_file") or ""
        stem = Path(inst_file).stem
        try:
            n = int(r.get("jobs") or r.get("instance_jobs") or 0)
        except Exception:
            n = 0
        try:
            m = int(r.get("machines") or r.get("instance_machines") or 0)
        except Exception:
            m = 0
        try:
            tl = int(r.get("time_limit_ms") or 0)
        except Exception:
            tl = 0
        alg = (r.get("algorithm") or "").lower()
        gap = _num(r.get("gap_percent"))
        if gap is not None:
            grouped[(stem, n, m, tl, alg)].append(gap)

    # Aggregate averages per (stem, n, m, tl)
    pivot: Dict[int, List[List[str]]] = {}
    keys = sorted(
        {(k[0], k[1], k[2], k[3]) for k in grouped.keys()},
        key=lambda x: (x[1], x[2], x[0]),
    )
    for stem, n, m, tl in keys:
        # compute avg for ils and sa
        def avg_for(alg: str) -> Optional[float]:
            vals = grouped.get((stem, n, m, tl, alg), [])
            return sum(vals) / len(vals) if vals else None

        row = [stem, str(n), str(m), str(tl)]
        ils_avg = avg_for("ils")
        sa_avg = avg_for("sa")
        row += [
            f"{ils_avg:.2f}" if ils_avg is not None else "",
            f"{sa_avg:.2f}" if sa_avg is not None else "",
        ]
        pivot.setdefault(tl, []).append(row)

    # Append AVG row for each tl
    for tl, rows_list in pivot.items():

        def _parse(v: str) -> Optional[float]:
            try:
                return float(v)
            except Exception:
                return None

        ils_vals = [_parse(r[4]) for r in rows_list if _parse(r[4]) is not None]
        sa_vals = [_parse(r[5]) for r in rows_list if _parse(r[5]) is not None]
        ils_avg = sum(ils_vals) / len(ils_vals) if ils_vals else None
        sa_avg = sum(sa_vals) / len(sa_vals) if sa_vals else None
        rows_list.append(
            [
                "AVG",
                "",
                "",
                str(tl),
                f"{ils_avg:.4f}" if ils_avg is not None else "",
                f"{sa_avg:.4f}" if sa_avg is not None else "",
            ]
        )

    return pivot


def write_motzkin_pivot_csv_and_tex(timestamp_dir: Path) -> List[Path]:
    rows = load_summary(timestamp_dir)
    pivot = build_motzkin_pivot(rows)
    if not pivot:
        print("[Motzkin] Brak danych motzkin_neighborhood w summary.csv")
        return []

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # For each tl, write CSV with two header rows compatible with dynamic writer
    for tl, rows_list in sorted(pivot.items()):
        csv_path = LATEX_DIR / f"motzkin_pivot_tl{tl}.csv"
        tex_path = LATEX_DIR / f"motzkin_pivot_tl{tl}.tex"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            # header rows
            w.writerow(["file", "n", "m", "tl_ms", "motzkin_neighborhood", ""])
            w.writerow(["", "", "", "", "ils", "sa"])
            # data
            for r in rows_list:
                w.writerow(r)
        # generate LaTeX dynamically from the CSV (self-contained)
        write_pivot_latex_local(str(csv_path), str(tex_path))
        print(f"[Motzkin] Written {csv_path} and {tex_path}")
        written.append(tex_path)
    return written


# --- LaTeX writer (local copy, dynamic neighborhoods) ---
def write_pivot_latex_local(csv_path: str, latex_path: str) -> None:
    """Write LaTeX table with siunitx S columns from a pivot CSV with two header rows.

    Supports dynamic set of neighborhoods. After base columns file,n,m,tl_ms, expects
    pairs of columns per neighborhood (ils, sa)."""
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if len(reader) < 3:
        raise ValueError(f"Pivot CSV too short: {csv_path}")
    header1 = reader[0]
    header2 = reader[1]
    data = reader[2:]

    # Build neighborhood column indices
    neigh_cols = []
    i = 4
    while i < len(header1):
        neigh = (header1[i] or "").strip()
        if not neigh:
            break
        idx_ils = i
        idx_sa = i + 1 if i + 1 < len(header2) else None
        neigh_cols.append((neigh, idx_ils, idx_sa))
        i += 2

    def short(name: str) -> str:
        m = {
            "fibonahi_neighborhood": "fib",
            "dynasearch_neighborhood": "dyn",
            "adjacent": "adj",
            "motzkin_neighborhood": "mot",
        }
        return m.get(name, name)

    def fmt_num(val: str) -> str:
        try:
            x = float(val)
            return f"{x:.2f}"
        except Exception:
            return val

    s_cols = "".join(["S[table-format=2.2]S[table-format=2.2]" for _ in neigh_cols])
    tab_fmt = "lccS[table-format=3.0]" + s_cols

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% --- Auto-generated table (siunitx S columns) ---\n")
        input_path = latex_path.replace(".tex", "").replace(
            "/Users/remigiuszwojewodzki/Desktop/Doktorat/PhD/", ""
        )
        f.write(f"% \\input{{{input_path}}}\n")
        f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{tab_fmt}}}\n")

        tl_val = data[0][3] if data and len(data[0]) > 3 else ""
        total_cols = 4 + 2 * len(neigh_cols)
        f.write(f"\\multicolumn{{{total_cols}}}{{c}}{{")
        f.write(f"\\textbf{{Time limit:}} $tl_{{\\mathrm{{ms}}}}={tl_val}$")
        f.write("} \\ \\hline\n")

        left = "\\textbf{file} & {$n$} & {$m$} & {tl$_{\\mathrm{ms}}$}"
        parts = [
            f"{{{short(name)}$_{{\\mathrm{{ILS}}}}$}} & {{"
            + f"{short(name)}$_{{\\mathrm{{SA}}}}$}}"
            for name, _, _ in neigh_cols
        ]
        header_tex = left + " & " + " & ".join(parts) + " \\ \\hline\n"
        f.write(header_tex)

        for row in data:
            file_name = (row[0] if len(row) > 0 else "").replace("_", "\\_")
            n = row[1] if len(row) > 1 else ""
            m = row[2] if len(row) > 2 else ""
            tl = row[3] if len(row) > 3 else ""
            try:
                tl_fmt = str(int(float(tl))) if tl not in ("", None) else tl
            except Exception:
                tl_fmt = tl
            vals = [file_name, n, m, tl_fmt]
            for _, idx_ils, idx_sa in neigh_cols:
                ils_v = row[idx_ils] if idx_ils is not None and idx_ils < len(row) else ""
                sa_v = row[idx_sa] if idx_sa is not None and idx_sa < len(row) else ""
                vals.append(fmt_num(ils_v))
                vals.append(fmt_num(sa_v))
            f.write(" & ".join(vals) + " \\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


if __name__ == "__main__":
    ts = find_latest_results_dir()
    if ts is None:
        print("[Motzkin] Nie znaleziono katalogu z wynikami: results/experiments")
    else:
        write_motzkin_pivot_csv_and_tex(ts)
