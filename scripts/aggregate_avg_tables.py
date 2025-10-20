def write_pivot_latex(csv_path, latex_path):
    """Generuje tabelę LaTeX na podstawie pliku pivot CSV."""
    import csv

    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.reader(f))
    header1 = reader[0]
    header2 = reader[1]
    data = reader[2:]
    ncols = len(header1)

    def mathify(val):
        # Jeśli liczba lub AVG, otocz $...$
        try:
            float(val)
            return f"${val}$"
        except Exception:
            if val.strip().upper() == "AVG":
                return f"$\\mathrm{{AVG}}$"
            return val

    def mathify_header(h):
        # Zamień n, m, tl_ms na $n$, $m$, $tl_{ms}$, inne tekstowe zostaw
        if h == "n":
            return "$n$"
        if h == "m":
            return "$m$"
        if h == "tl_ms":
            return "$tl_{ms}$"
        if h == "gap_pct":
            return "$gap$"
        if "_" in h:
            # np. dynasearch_neighborhood
            return f"$\\mathrm{{{h}}}$"
        return f"\\textbf{{{h}}}" if h else ""

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% --- Tabela wygenerowana automatycznie ---\n")
        # Dodaj opis czasu na podstawie pierwszego wiersza danych (tl_ms)
        tl_val = None
        if data and len(data[0]) > 3:
            tl_val = data[0][3]
        # Dodaj komentarz z \\input na górze pliku
        input_path = latex_path.replace(".tex", "").replace(
            "/Users/remigiuszwojewodzki/Desktop/Doktorat/PhD/", ""
        )
        f.write(f"% \\input{{{input_path}}}\n")
        f.write("\\small\n")
        tabular_format = "c" * 10
        f.write(f"\\begin{{tabular}}{{{tabular_format}}}\n")
        # Dodaj wiersz nagłówkowy z czasem
        if tl_val is not None:
            f.write(
                f"\\multicolumn{{10}}{{c}}{{\\textbf{{Time limit:}} $tl_{{{{ms}}}}={tl_val}$}} \\\\ \n"
            )
        # Nagłówek: file, n, m, tl_ms, fibonahi_tabu, fibonahi_sa, dynasearch_tabu, dynasearch_sa, adjacent_tabu, adjacent_sa
        header_row = [
            "file",
            "n",
            "m",
            "tl_{ms}",
            "fibonahi_tabu",
            "fibonahi_sa",
            "dynasearch_tabu",
            "dynasearch_sa",
            "adjacent_tabu",
            "adjacent_sa",
        ]
        f.write(
            " & ".join(
                [
                    "\\textbf{" + h + "}" if i == 0 else mathify_header(h)
                    for i, h in enumerate(header_row)
                ]
            )
            + " \\\\ \n"
        )
        # Wiersze danych
        for row in data:
            # Zamień _ na \\_ w pierwszej kolumnie (nazwy instancji/plików)
            first_col = row[0].replace("_", "\\_")
            out_row = [first_col, row[1], row[2], row[3]]
            out_row += [row[8], row[9], row[6], row[7], row[4], row[5]]
            f.write(" & ".join([mathify(val) for val in out_row]) + " \\\\ \n")
        f.write("\\end{tabular}\n")
        # Usunięto nieużywaną zmienną tl_caption, która powodowała błąd NameError


from pathlib import Path

base = Path("latex_tables")
csv_files = list(base.glob("**/*_pivot_tl*.csv"))
for csv_path in csv_files:
    latex_path = str(csv_path).replace(".csv", ".tex")
    write_pivot_latex(str(csv_path), latex_path)


def _plot_gap_vs_time_from_pivot(csv_path: str, out_dir: str | Path = "latex_tables/plots"):
    """Wczytaj pivot CSV i narysuj wykres gap[%] vs time[ms] dla każdej sąsiedztwa.

    Oczekujemy, że w CSV istnieją kolumny zawierające czasy (tl_ms / time) oraz
    wartości gap (gap_pct lub podobne). Jeśli nie, funkcja próbuje dopasować pola
    heurystycznie. Wynikowe pliki zapisywane są jako PNG w `out_dir`.
    """
    import csv

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

    # Try to detect columns: time and gap columns per neighborhood
    # Heuristic: look for 'tl' or 'time' in headers for time; 'gap' or 'gap_pct' for gaps
    time_idx = None
    gap_indices = []  # list of tuples (neigh_name, col_idx)
    for i, h in enumerate(header):
        key = h.strip().lower()
        if "time" in key or "tl" in key:
            time_idx = i
        if "gap" in key:
            neigh = h.strip()
            gap_indices.append((neigh, i))

    # Fallback: if no explicit gap columns, maybe columns contain per-algo values in fixed positions
    if not gap_indices:
        # try to collect numeric columns after 4th column (heuristic from original script)
        for i in range(4, min(len(header), 12)):
            gap_indices.append((header[i], i))

    # Collect time and gap series
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

    # If times are all None, try to synthesize as indices
    if all(t is None for t in times):
        times = list(range(len(data)))
    else:
        # replace None with previous or zero
        last = 0.0
        for i, t in enumerate(times):
            if t is None:
                times[i] = last
            else:
                last = times[i]

    # Plot
    plt.figure(figsize=(10, 6))
    for name, vals in gap_series.items():
        # sanitize values
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
