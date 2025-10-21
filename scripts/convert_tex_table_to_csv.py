#!/usr/bin/env python3
"""Convert a LaTeX longtable (auto-generated master table) to CSV.

By default operates on: latex_tables/20251011_215445/all_results.tex
and writes to: latex_tables/20251011_215445/csv/all_results.csv

The parser looks for the literal sequence '\\endhead' and treats lines after it as
data rows. It performs light sanitization of LaTeX escapes (for example '\\_' ->
'_', '\\%' -> '%') and removes math delimiters like '$' and braces.
"""
import argparse
import csv
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import numpy as np
except Exception:
    np = None


def unlatex(s: str) -> str:
    s = s.strip()
    s = s.replace("\\_", "_")
    s = s.replace("\\%", "%")
    # remove math delimiters and braces
    s = s.replace("$", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("\\", "")
    # normalize spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_header(name: str) -> str:
    # produce a conservative, file-friendly header name
    h = unlatex(name)
    h = h.strip()
    h = h.replace("(ms)", "_ms")
    h = h.replace("(%)", "_pct")
    h = h.replace(" ", "_")
    h = h.replace("%", "pct")
    h = h.replace("__", "_")
    h = h.strip("_")
    return h


def parse_table(tex_path: Path):
    txt = tex_path.read_text(encoding="utf-8")
    lines = txt.splitlines()

    # find \endhead; data follows after that
    endhead_idx = None
    for i, L in enumerate(lines):
        if "\\endhead" in L:
            endhead_idx = i
            break

    # find header: scan backwards from endhead to find a line with '&'
    header_line = None
    if endhead_idx is not None:
        for j in range(endhead_idx - 1, -1, -1):
            if "&" in lines[j]:
                header_line = lines[j]
                break
    else:
        # fallback: first line after \toprule
        top_idx = None
        for i, L in enumerate(lines):
            if "\\toprule" in L:
                top_idx = i
                break
        if top_idx is not None:
            for j in range(top_idx + 1, len(lines)):
                if "&" in lines[j]:
                    header_line = lines[j]
                    break

    if header_line is None:
        raise RuntimeError(f"Could not find header line in {tex_path}")

    # sanitize header: remove \midrule and trailing \\\\ if present
    header_line = header_line.replace("\\midrule", "")
    header_line = re.sub(r"\\\\\s*$", "", header_line)
    header_cells = [c.strip() for c in header_line.split("&")]
    header = [normalize_header(c) for c in header_cells]

    # data lines: start after endhead_idx
    data_lines = []
    start_idx = (endhead_idx + 1) if endhead_idx is not None else 0
    for L in lines[start_idx:]:
        if not L.strip():
            continue
        if L.strip().startswith("%"):
            continue
        if "\\end{longtable}" in L:
            break
        # consider only lines that contain & and end with LaTeX linebreak '\\'
        if "&" in L and L.rstrip().endswith("\\\\"):
            # skip the repeated header line (starts with 'file & inst')
            low = L.lower()
            if low.strip().startswith("file & inst") or low.strip().startswith("file & inst & alg"):
                continue
            data_lines.append(L)

    rows = []
    for L in data_lines:
        # remove trailing LaTeX line ending and midrule if any
        L = L.replace("\\midrule", "")
        L = re.sub(r"\\\\\s*$", "", L)
        cells = [unlatex(c).strip() for c in L.split("&")]
        rows.append(cells)

    return header, rows


def write_csv(out_dir: Path, header, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_results.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            # ensure row length equals header length
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            elif len(r) > len(header):
                r = r[: len(header)]
            writer.writerow(r)
    return out


def try_parse_num(s: str):
    """Try to parse a string as int or float; return None if not numeric."""
    s = s.strip()
    if s == "":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except Exception:
            return None


def group_and_average(rows, header):
    """Group rows by (file, tl_ms, alg, neigh) and average numeric columns.

    Rules:
    - Group key columns: 'file', 'tl_ms', 'alg', 'neigh'
    - For grouped rows, create one output row with 'inst' set to 1.
    - For numeric columns (ints/floats) compute the arithmetic mean and round
      to nearest int if all inputs were ints; otherwise keep float.
    - For non-numeric columns pick the first non-empty value.
    - Preserve header order.
    """
    # map header name -> index
    idx = {h: i for i, h in enumerate(header)}
    key_cols = [
        "file",
        "tl_ms",
        "alg",
        "neigh",
    ]
    # ensure key columns exist
    for k in key_cols:
        if k not in idx:
            raise RuntimeError(f"Required key column missing: {k}")

    groups = {}
    for r in rows:
        # pad/truncate to header length
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[: len(header)]
        key = tuple(r[idx[k]].strip() for k in key_cols)
        groups.setdefault(key, []).append(r)

    out_rows = []
    for key, group in groups.items():
        # aggregate columns
        agg = [""] * len(header)
        # set key columns from key
        for i, k in enumerate(key_cols):
            agg[idx[k]] = key[i]
        # set inst to '1' if present
        if "inst" in idx:
            agg[idx["inst"]] = "1"

        for col_i, col_name in enumerate(header):
            if col_name in key_cols or col_name == "inst":
                continue
            values = []
            for r in group:
                parsed = try_parse_num(r[col_i])
                if parsed is not None:
                    values.append(parsed)
                else:
                    # keep the raw stripped string for non-numeric
                    values.append(r[col_i].strip())
            # separate numeric and non-numeric
            nums = [v for v in values if isinstance(v, (int, float))]
            nonnums = [v for v in values if not isinstance(v, (int, float)) and v != ""]
            if nums:
                # if all ints, average and round to int
                if all(isinstance(v, int) for v in nums):
                    avg = int(round(sum(nums) / len(nums)))
                else:
                    avg = sum(nums) / len(nums)
                    # keep 6 significant digits-ish
                    avg = round(avg, 6)
                agg[col_i] = str(avg)
            elif nonnums:
                agg[col_i] = str(nonnums[0])
            else:
                agg[col_i] = ""

        out_rows.append(agg)

    # sort results for determinism (by key)
    out_rows.sort(key=lambda r: tuple(r[idx[k]] for k in key_cols))
    return out_rows


def make_plot(
    header,
    rows,
    out_dir: Path,
    out_name: str = "gap_vs_time.png",
    aggregate: bool = False,
):
    """Create a scatter plot: x=tl_ms, y=gap_pct, colored by (alg,neigh) combo.

    Expects header to contain 'tl_ms' and 'gap_pct'. Saves PNG to out_dir/gap_vs_time.png
    """
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    hidx = {h: i for i, h in enumerate(header)}
    if "tl_ms" not in hidx or "gap_pct" not in hidx or "alg" not in hidx or "neigh" not in hidx:
        raise RuntimeError("Required columns missing for plotting: tl_ms,gap_pct,alg,neigh")

    tl_i = hidx["tl_ms"]
    gap_i = hidx["gap_pct"]
    alg_i = hidx["alg"]
    neigh_i = hidx["neigh"]

    combos = []
    for r in rows:
        combos.append((r[alg_i], r[neigh_i]))
    combos = sorted(set(combos))

    # map combos to 6 colors (if more combos exist we cycle)
    base_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    combo_to_color = {c: base_colors[i % len(base_colors)] for i, c in enumerate(combos)}
    # provide distinct linestyles to help distinguish overlapping/close lines
    base_linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    combo_to_ls = {c: base_linestyles[i % len(base_linestyles)] for i, c in enumerate(combos)}

    # quantize x: map tl_ms values to discrete positions (one per unique tl)
    tl_values = sorted({int(r[tl_i]) for r in rows if r[tl_i] != "" and r[tl_i] is not None})
    tl_to_pos = {v: i for i, v in enumerate(tl_values)}

    fig, ax = plt.subplots(figsize=(8, 5))
    # collect points per combo so we can draw connecting lines per color
    combo_points = {}
    if aggregate:
        # Build a lookup of values per (alg,neigh,tl)
        agg = {}
        for r in rows:
            try:
                tl_raw = r[tl_i]
                if tl_raw == "" or tl_raw is None:
                    continue
                tl = int(float(tl_raw))
                gap = float(r[gap_i])
            except Exception:
                continue
            key = (r[alg_i], r[neigh_i], tl)
            agg.setdefault(key, []).append(gap)

        # For each combo (alg,neigh) and each tl value, compute mean gap so that
        # there are up to len(combos) points per time bucket (typically 6)
        for combo in combos:
            alg, neigh = combo
            for tl in tl_values:
                vals = agg.get((alg, neigh, tl), [])
                if not vals:
                    continue
                mean_gap = sum(vals) / len(vals)
                x = tl_to_pos.get(tl)
                marker = "o"
                size = 30
                combo_points.setdefault((alg, neigh), []).append(
                    {"x": x, "y": mean_gap, "marker": marker, "size": size}
                )
    else:
        for r in rows:
            try:
                tl_raw = r[tl_i]
                if tl_raw == "" or tl_raw is None:
                    continue
                tl = int(float(tl_raw))
                x = tl_to_pos.get(tl)
                y = float(r[gap_i])
            except Exception:
                continue
            key = (r[alg_i], r[neigh_i])
            # always use point marker (circle); keep size consistent
            marker = "o"
            size = 20
            combo_points.setdefault(key, []).append(
                {"x": x, "y": y, "marker": marker, "size": size}
            )
    # draw connecting lines and points per combo
    handles = []
    labels = []
    for combo, pts in combo_points.items():
        col = combo_to_color.get(combo, "C6")
        # sort by x-position (time buckets)
        pts_sorted = sorted([p for p in pts if p["x"] is not None], key=lambda p: p["x"])
        xs = [p["x"] for p in pts_sorted]
        ys = [p["y"] for p in pts_sorted]
        # draw connecting lines for all plots (including aggregated)
        if len(xs) > 1:
            ls = combo_to_ls.get(combo, "-")
            ax.plot(xs, ys, color=col, linestyle=ls, linewidth=1.8, alpha=1.0)
        # draw individual points (to respect different markers/sizes)
        for p in pts_sorted:
            ax.scatter(
                p["x"],
                p["y"],
                color=col,
                edgecolor="k",
                alpha=1.0,
                s=p["size"],
                marker=p["marker"],
            )

        # legend entry for this combo
        # legend entry shows the line style and color for the combo
        handles.append(
            plt.Line2D([0], [0], color=col, linestyle=combo_to_ls.get(combo, "-"), linewidth=1.8)
        )
        labels.append(f"{combo[0]} | {combo[1]}")

    # final legend: combo entries + short marker explanation (Polish)
    marker_handles = [plt.Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=7)]
    marker_labels = ["o = punkt"]
    all_handles = handles + marker_handles
    all_labels = labels + marker_labels
    ax.legend(
        all_handles,
        all_labels,
        title="alg | neigh",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_xlabel("time limit (ms)")
    ax.set_ylabel("gap (%)")
    # set discrete x ticks to the actual tl values (quantized)
    ax.set_xticks(list(range(len(tl_values))))
    ax.set_xticklabels([str(v) for v in tl_values])
    ax.set_title("Gap (%) vs time limit")
    plt.tight_layout()
    outp = out_dir / out_name
    plt.savefig(outp, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {outp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="latex_tables/20251011_215445/all_results.tex")
    p.add_argument("--outdir", "-o", default="latex_tables/20251011_215445/csv")
    p.add_argument("--plot", action="store_true", help="Generate gap vs time scatter plot")
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate gap_pct across files and plot one averaged point per alg/neigh/tl",
    )
    args = p.parse_args()
    tex_path = Path(args.input)
    out_dir = Path(args.outdir)
    if not tex_path.exists():
        print(f"Input file not found: {tex_path}")
        raise SystemExit(1)
    header, rows = parse_table(tex_path)
    # group and average rows that share the same (file, tl_ms, alg, neigh)
    try:
        rows = group_and_average(rows, header)
    except Exception as e:
        print(f"Warning: grouping failed ({e}), writing original rows")

    # remove unwanted columns (inst, seed) and sort rows by file order
    def drop_and_sort(header, rows):
        # columns to drop
        drop = {"inst", "seed"}
        # compute indices to keep
        keep_indices = [i for i, h in enumerate(header) if h not in drop]
        new_header = [header[i] for i in keep_indices]

        # desired ordering for jobs and machines
        jobs_order = [20, 50, 100, 200, 500]
        mach_order = [5, 10, 20]

        def parse_file(fname: str):
            # expect patterns like 'tai100_10' or 'tai100_10_extra'
            m = re.search(r"(?P<jobs>\d+)_(?P<mach>\d+)", fname)
            if not m:
                return None, None
            return int(m.group("jobs")), int(m.group("mach"))

        def sort_key(row):
            # row contains original columns; look up file column
            try:
                file_col = header.index("file")
            except ValueError:
                file_col = None
            fname = row[file_col] if file_col is not None else ""
            jobs, mach = parse_file(fname)
            try:
                jpos = jobs_order.index(jobs) if jobs in jobs_order else len(jobs_order)
            except Exception:
                jpos = len(jobs_order)
            try:
                mpos = mach_order.index(mach) if mach in mach_order else len(mach_order)
            except Exception:
                mpos = len(mach_order)
            # keep deterministic tiebreakers
            return (jpos, mpos, fname, row)

        # build new rows with kept columns
        new_rows = [[r[i] for i in keep_indices] for r in rows]
        # sort by desired order
        new_rows.sort(key=sort_key)
        return new_header, new_rows

    try:
        header, rows = drop_and_sort(header, rows)
    except Exception as e:
        print(f"Warning: drop_and_sort failed ({e}), writing unmodified data")
    # add 'over_limit' column: True if t_tot_ms > tl_ms else False
    if "over_limit" in header:
        # avoid duplicate
        pass
    else:
        header.append("over_limit")
        # find indices for tl_ms and t_tot_ms
        try:
            tl_idx = header.index("tl_ms")
        except ValueError:
            tl_idx = None
        try:
            tot_idx = header.index("t_tot_ms")
        except ValueError:
            tot_idx = None

        new_rows = []
        for r in rows:
            # ensure row length
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            # compute over_limit with 10% margin: mark True if total time > 110% of tl_ms
            over = "False"
            try:
                if tl_idx is not None and tot_idx is not None:
                    tl = float(r[tl_idx])
                    tot = float(r[tot_idx])
                    margin = 0.10
                    over = "True" if tot > tl * (1 + margin) else "False"
            except Exception:
                over = "False"
            r[-1] = over
            new_rows.append(r)
        rows = new_rows

    out = write_csv(out_dir, header, rows)
    print(f"Wrote CSV: {out}")

    # optional plotting
    if args.plot:
        if plt is None:
            print("matplotlib not available; cannot plot")
            return
        try:
            # combined plot
            make_plot(header, rows, out_dir, out_name="gap_vs_time.png", aggregate=args.aggregate)
            # per-file plots
            file_idx = None
            try:
                file_idx = header.index("file")
            except ValueError:
                file_idx = None
            if file_idx is not None:
                files = sorted({r[file_idx] for r in rows if r[file_idx]})
                for fname in files:
                    sel = [r for r in rows if r[file_idx] == fname]
                    safe_name = fname.replace("/", "_").replace(" ", "_")
                    out_name = f"gap_vs_time_{safe_name}.png"
                    make_plot(header, sel, out_dir, out_name=out_name, aggregate=args.aggregate)
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
