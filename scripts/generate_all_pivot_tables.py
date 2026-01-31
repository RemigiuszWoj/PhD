from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_RESULTS = Path("results/experiments")
OUT_DIR_LATEX = Path("latex_tables")
OUT_DIR_TEX = Path("tables")


def find_latest_results_dir() -> Optional[Path]:
    if not BASE_RESULTS.exists():
        return None
    dirs = [d for d in BASE_RESULTS.iterdir() if d.is_dir()]
    if not dirs:
        return None
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


def detect_neighborhoods(rows: List[dict]) -> List[str]:
    neighs = sorted({(r.get("neighborhood") or "") for r in rows if r.get("neighborhood")})
    # prefer known order
    order = [
        "adjacent",
        "fibonahi_neighborhood",
        "dynasearch_neighborhood",
        "motzkin_neighborhood",
    ]
    # keep order; append any unknowns
    ordered = [n for n in order if n in neighs]
    for n in neighs:
        if n not in ordered:
            ordered.append(n)
    return ordered


def build_avg_by_file_tl(rows: List[dict]) -> Dict[Tuple[str, int], Dict[str, Dict[str, float]]]:
    """Return dict keyed by (stem, tl_ms) with avg gaps per (neigh, alg).

    Structure: keys (stem, tl_ms) map to { '<neigh>_ils': avg_gap, '<neigh>_sa': avg_gap }.
    The average is computed across instance_number for a given
    stem/algorithm/neighborhood/time limit.
    """
    from collections import defaultdict

    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    acc: Dict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    dims: Dict[str, Tuple[int, int]] = {}
    for r in rows:
        stem = Path(r.get("instance_file") or "").stem
        if not stem:
            continue
        try:
            tl = int(r.get("time_limit_ms") or 0)
        except Exception:
            tl = 0
        alg = (r.get("algorithm") or "").lower()
        neigh = r.get("neighborhood") or ""
        gap = _num(r.get("gap_percent"))
        if gap is not None:
            acc[(stem, tl, neigh, alg)].append(gap)
        # capture n,m dims
        try:
            n = int(r.get("jobs") or r.get("instance_jobs") or 0)
        except Exception:
            n = 0
        try:
            m = int(r.get("machines") or r.get("instance_machines") or 0)
        except Exception:
            m = 0
        if stem not in dims and n and m:
            dims[stem] = (n, m)
    # aggregate
    for (stem, tl, neigh, alg), vals in acc.items():
        key = (stem, tl)
        out.setdefault(key, {})
        out[key][f"{neigh}_{alg}"] = sum(vals) / len(vals)
    # attach dims via separate map in caller
    return out, dims


def write_pivot_for_tl(
    tl: int,
    stems_for_tl: List[Tuple[str, int, int]],
    neighs: List[str],
    values: Dict[Tuple[str, int], Dict[str, float]],
    out_csv: Path,
    out_tex: Path,
) -> None:
    """Write a pivot CSV with two header rows, then call LaTeX writer to produce .tex."""
    OUT_DIR_LATEX.mkdir(parents=True, exist_ok=True)
    OUT_DIR_TEX.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        # header rows: file,n,m,tl_ms plus 2 cols per neigh
        header1 = ["file", "n", "m", "tl_ms"]
        for neigh in neighs:
            header1 += [neigh, ""]
        header2 = ["", "", "", ""]
        for _ in neighs:
            header2 += ["ils", "sa"]
        w.writerow(header1)
        w.writerow(header2)
        for stem, n, m in stems_for_tl:
            row = [stem, n, m, tl]
            vals = values.get((stem, tl), {})
            for neigh in neighs:
                ils = vals.get(f"{neigh}_ils")
                sa = vals.get(f"{neigh}_sa")
                row.append(f"{ils:.2f}" if ils is not None else "")
                row.append(f"{sa:.2f}" if sa is not None else "")
            w.writerow(row)
        # AVG row

        def parse_float(v: str) -> Optional[float]:
            try:
                return float(v)
            except Exception:
                return None

        avg_row = ["AVG", "", "", tl]
        # compute per column averages (skip first 4)
        # read back rows to calculate averages easily
    # read back for avg
    with out_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    data = reader[2:]
    # number of numeric columns
    num_cols = len(reader[0]) - 4
    col_avgs: List[Optional[float]] = []
    for i in range(num_cols):
        vals = []
        for row in data:
            v = row[4 + i]
            try:
                x = float(v)
                vals.append(x)
            except Exception:
                pass
        col_avgs.append(sum(vals) / len(vals) if vals else None)
    avg_row = ["AVG", "", "", tl]
    for avg in col_avgs:
        avg_row.append(f"{avg:.4f}" if avg is not None else "")
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(avg_row)

    # LaTeX: use local dynamic writer (siunitx S columns)
    write_pivot_latex_local(str(out_csv), str(out_tex))
    # also copy to tables/ with same basename (without extension in \input)
    try:
        OUT_DIR_TEX.mkdir(parents=True, exist_ok=True)
        # ensure copy
        with (
            out_tex.open("r", encoding="utf-8") as src,
            (OUT_DIR_TEX / out_tex.name).open("w", encoding="utf-8") as dst,
        ):
            dst.write(src.read())
    except Exception:
        pass


def main():
    ts = find_latest_results_dir()
    if ts is None:
        print("[AllPivot] Nie znaleziono katalogu z wynikami: results/experiments")
        return
    rows = load_summary(ts)
    if not rows:
        print("[AllPivot] Brak rekordów w summary.csv")
        return
    neighs = detect_neighborhoods(rows)
    values, dims = build_avg_by_file_tl(rows)
    # collect stems per tl with n,m
    from collections import defaultdict

    stems_per_tl: Dict[int, List[Tuple[str, int, int]]] = defaultdict(list)
    for stem, tl in values.keys():
        n, m = dims.get(stem, (0, 0))
        stems_per_tl[tl].append((stem, n, m))
    for tl, lst in sorted(stems_per_tl.items()):
        out_csv = OUT_DIR_LATEX / f"all_results_avg_pivot_tl{tl}.csv"
        out_tex = OUT_DIR_LATEX / f"all_results_avg_pivot_tl{tl}.tex"
        write_pivot_for_tl(tl, sorted(lst), neighs, values, out_csv, out_tex)
        print(f"[AllPivot] Written {out_csv} and {out_tex}")


def write_pivot_latex_local(csv_path: str, latex_path: str) -> None:
    """Write LaTeX table (siunitx) from a pivot CSV with two header rows.

    After base columns file,n,m,tl_ms, data contains pairs (neigh ILS, neigh SA).
    Neighborhood set is dynamic and reflected in headers.
    """
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
    # tl column removed from table
    tab_fmt = "lcc" + s_cols

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{tab_fmt}}}\n")

        tl_val = data[0][3] if data and len(data[0]) > 3 else ""
        total_cols = 3 + 2 * len(neigh_cols)
        f.write(f"\\multicolumn{{{total_cols}}}{{c}}{{")
        f.write(f"\\textbf{{Time limit:}} $tl_{{\\mathrm{{ms}}}}={tl_val}$")
        f.write("} \\\\ \n")
        f.write("\\hline\n")

        left = "\\textbf{file} & {$n$} & {$m$}"
        parts = [
            f"{{{short(name)}$_{{\\mathrm{{ILS}}}}$}} & {{"
            + f"{short(name)}$_{{\\mathrm{{SA}}}}$}}"
            for name, _, _ in neigh_cols
        ]
        header_tex = left + " & " + " & ".join(parts) + " \\\\ \n"
        f.write(header_tex)
        f.write("\\hline\n")

        for row in data:
            file_name_raw = row[0] if len(row) > 0 else ""
            file_name = file_name_raw.replace("_", "\\_")
            n = row[1] if len(row) > 1 else ""
            m = row[2] if len(row) > 2 else ""
            # Build cell list (omit tl column; it's in header)
            cells: List[str] = [file_name, n, m]
            numeric_values: List[Tuple[int, float]] = []  # (cell_index, rounded_value)

            def to_float(v: str) -> Optional[float]:
                try:
                    return float(v)
                except Exception:
                    return None

            for j, (neigh_name, idx_ils, idx_sa) in enumerate(neigh_cols):
                ils_v = row[idx_ils] if idx_ils is not None and idx_ils < len(row) else ""
                sa_v = row[idx_sa] if idx_sa is not None and idx_sa < len(row) else ""

                # format displayed values
                ils_fmt = fmt_num(ils_v)
                sa_fmt = fmt_num(sa_v)
                cells.append(ils_fmt)
                cells.append(sa_fmt)

                # collect numeric values (rounded to 2 decimals to match display)
                # omit dynasearch from bolding
                if neigh_name != "dynasearch_neighborhood":
                    tf = to_float(ils_v)
                    sf = to_float(sa_v)
                    base_offset = 3  # file, n, m
                    if tf is not None:
                        numeric_values.append((base_offset + 2 * j, round(tf, 2)))
                    if sf is not None:
                        numeric_values.append((base_offset + 2 * j + 1, round(sf, 2)))

            # Determine min among neighborhood values (skip AVG row)
            is_avg_row = (file_name_raw or "").strip().upper() == "AVG"
            bold_indices: set[int] = set()
            if numeric_values and not is_avg_row:
                min_val = min(v for _, v in numeric_values)
                # bold all ties equal to min
                bold_indices = {idx for idx, v in numeric_values if v == min_val}

            # Apply bolding in LaTeX for S columns using \bfseries
            for idx in sorted(bold_indices):
                cells[idx] = "{\\bfseries " + cells[idx] + "}"

            f.write(" & ".join(cells) + " \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


if __name__ == "__main__":
    main()
