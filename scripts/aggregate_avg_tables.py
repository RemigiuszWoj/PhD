import csv
from pathlib import Path


def write_pivot_latex(csv_path: str, latex_path: str) -> None:
    """Zbuduj tabelę LaTeX (siunitx) z pliku pivot CSV.

    Obsługuje dynamicznie zestaw sąsiedztw: odczytuje pierwsze dwa wiersze nagłówków,
    gdzie po kolumnach file,n,m,tl_ms następują pary (neigh, '') i w drugim wierszu
    odpowiadające im (tabu, sa). Dzięki temu, jeśli pojawi się np. motzkin_neighborhood,
    tabela zostanie rozszerzona o kolejne dwie kolumny S.
    """

    # Wczytaj pivot CSV: pierwsze 2 wiersze to nagłówki
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if len(reader) < 3:
        raise ValueError(f"Pivot CSV too short: {csv_path}")
    header1 = reader[0]
    header2 = reader[1]
    data = reader[2:]

    # Zbuduj listę (neigh_name, idx_tabu, idx_sa) po 4 bazowych kolumnach
    neigh_cols = []
    i = 4
    while i < len(header1):
        neigh = (header1[i] or "").strip()
        if not neigh:
            break
        # Indeksy dla TABU/SA (wg drugiego wiersza nagłówka)
        idx_tabu = i
        idx_sa = i + 1 if i + 1 < len(header2) else None
        # w praktyce header2[idx_tabu] == 'tabu', header2[idx_sa] == 'sa'
        neigh_cols.append((neigh, idx_tabu, idx_sa))
        i += 2

    def neigh_short(name: str) -> str:
        m = {
            "fibonahi_neighborhood": "fib",
            "dynasearch_neighborhood": "dyn",
            "adjacent": "adj",
            "motzkin_neighborhood": "mot",
        }
        return m.get(name, name)

    def fmt_num(val: str) -> str:
        """Formatuj liczby do 2 miejsc po przecinku dla kolumn S; inne wartości zostaw."""
        try:
            x = float(val)
            return f"{x:.2f}"
        except Exception:
            return val

    # Zbuduj format kolumn: l c c dla file,n,m,tl_ms i po dwie S na sąsiedztwo
    s_cols = "".join(["S[table-format=2.2]S[table-format=2.2]" for _ in neigh_cols])
    tab_fmt = "lccS[table-format=3.0]" + s_cols

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% --- Auto-generated table (siunitx S columns) ---\n")
        # Opcjonalny komentarz z \input (relative path)
        input_path = latex_path.replace(".tex", "").replace(
            "/Users/remigiuszwojewodzki/Desktop/Doktorat/PhD/", ""
        )
        f.write(f"% \\input{{{input_path}}}\n")
        f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{tab_fmt}}}\n")

        # Wartość tl_ms do nagłówka (jeśli dostępna, z pierwszego wiersza danych)
        tl_val = ""
        if data and len(data[0]) > 3:
            tl_val = data[0][3]

        # Linia z informacją o limicie czasu
        total_cols = 4 + 2 * len(neigh_cols)
        f.write(f"\\multicolumn{{{total_cols}}}{{c}}{{")
        f.write(f"\\textbf{{Time limit:}} $tl_{{\\mathrm{{ms}}}}={tl_val}$")
        f.write("} \\ \\hline\n")

        # Nagłówek kolumn
        header_tex_left = "\\textbf{file} & {$n$} & {$m$} & {tl$_{\\mathrm{ms}}$}"
        parts = []
        for name, _, _ in neigh_cols:
            short = neigh_short(name)
            parts.append(
                f"{{{short}$_{{\\mathrm{{TABU}}}}$}} & " f"{{{short}$_{{\\mathrm{{SA}}}}$}}"
            )
        header_tex = header_tex_left + " & " + " & ".join(parts) + " \\ \\hline\n"
        f.write(header_tex)

        # Wiersze danych
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
            for _, idx_tabu, idx_sa in neigh_cols:
                tabu_v = row[idx_tabu] if idx_tabu is not None and idx_tabu < len(row) else ""
                sa_v = row[idx_sa] if idx_sa is not None and idx_sa < len(row) else ""
                vals.append(fmt_num(tabu_v))
                vals.append(fmt_num(sa_v))

            f.write(" & ".join(vals) + " \\ \n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def _plot_gap_vs_time_from_pivot(csv_path: str, out_dir: str | Path = "latex_tables/plots"):
    """Narysuj wykres gap[%] vs time[ms] na podstawie pivot CSV (heurystyka kolumn)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if len(reader) < 3:
        print(f"[plot_gap] Not enough rows in {csv_path}")
        return
    header = reader[0]
    data = reader[2:]

    time_idx = None
    gap_indices = []  # list[tuple[str,int]]
    for i, h in enumerate(header):
        key = h.strip().lower()
        if "time" in key or "tl" in key:
            time_idx = i
        # heurystyka: numeric columns after first 4 columns
        if i >= 4:
            gap_indices.append((h.strip(), i))

    times = []
    gap_series = {name: [] for name, _ in gap_indices}
    for row in data:
        if time_idx is not None and time_idx < len(row):
            try:
                t = float(row[time_idx])
            except Exception:
                t = None
        else:
            t = None
        for name, idx in gap_indices:
            val = None
            if idx < len(row):
                try:
                    val = float(row[idx])
                except Exception:
                    val = None
            gap_series[name].append(val)
        times.append(t)

    if all(t is None for t in times):
        times = list(range(len(data)))
    else:
        last = 0.0
        for i, t in enumerate(times):
            if t is None:
                times[i] = last
            else:
                last = times[i]

    # Plot
    plt.figure(figsize=(10, 6))
    for name, vals in gap_series.items():
        y = [v if v is not None else float("nan") for v in vals]
        plt.plot(times, y, marker="o", label=name)
    plt.xlabel("Time [ms]")
    plt.ylabel("Gap [%]")
    plt.title(f"Gap vs Time - {csv_path.stem}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    out_file = out_dir / (csv_path.stem + "_gap.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"[plot_gap] Saved {out_file}")


if __name__ == "__main__":
    base = Path("latex_tables")
    csv_files = list(base.glob("**/*_pivot_tl*.csv"))
    for csv_path in csv_files:
        latex_path = str(csv_path).replace(".csv", ".tex")
        write_pivot_latex(str(csv_path), latex_path)
