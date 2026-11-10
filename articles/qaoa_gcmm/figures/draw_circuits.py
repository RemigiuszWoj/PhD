"""Draw the example p=1 QAOA circuits of Figs. 1-2 as TikZ.

The coefficients are those of the submitted figures (small illustrative QUBOs).
The circuits are written as TikZ so that the labels are typeset in the font of
the paper and stay legible at full text width (reviewer 1, comment 5).
Commuting R_ZZ gates whose wire spans do not overlap share a column.

    python3 articles/qaoa_gcmm/figures/draw_circuits.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

CIRCUITS = {
    # name: (linear coefficients h_i, couplings (i, j, J_ij))
    "qaoa_circuit_fibonacci": ([1, -0.6, 0.8, -0.4], [(0, 1, 1.2), (1, 2, 1), (2, 3, 0.8)]),
    "qaoa_adjacent": ([1, -0.6, 0.8, -0.4],
                      [(0, 1, 1), (0, 2, 1), (1, 2, 1), (0, 3, 1), (1, 3, 1), (2, 3, 1)]),
    "qaoa_dynasearch": ([0, 1, 0, 0.1, -0.6],
                        [(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 2, 1), (0, 4, 1),
                         (1, 3, 1), (1, 4, 1), (2, 3, 1), (2, 4, 1), (3, 4, 1)]),
    "qaoa_motzkin": ([0, 1, 0, 0.1, -0.6], [(0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 4, 1)]),
}

DX = 0.82   # R_ZZ column width [cm]
DY = 0.62   # wire spacing [cm]


def coef(c):
    if c == 0:
        return "0"
    if c == 1:
        return r"\gamma"
    return f"{c:g}" + r"\gamma"


def layers(edges):
    """Greedy columns of R_ZZ gates with disjoint wire spans (all R_ZZ commute)."""
    cols = []
    for i, j, c in edges:
        lo, hi = min(i, j), max(i, j)
        for col in cols:
            if all(hi < a or lo > b for a, b, _ in col):
                col.append((lo, hi, c))
                break
        else:
            cols.append([(lo, hi, c)])
    return cols


def tikz(h, J):
    k = len(h)
    zz = layers(J)
    xh, xz, x0 = 0.85, 2.05, 3.25               # H, R_Z, first R_ZZ column [cm]
    xr = x0 + len(zz) * DX + 0.1                # R_X
    xend = xr + 0.7
    y = lambda q: -q * DY
    s = [r"\begin{tikzpicture}[font=\footnotesize,"
         r"g/.style={draw,fill=white,minimum height=0.46cm,minimum width=0.62cm,inner sep=1.5pt},"
         r"dot/.style={circle,fill,inner sep=1.3pt}]"]
    for q in range(k):
        s.append(rf"\draw ({0.25:.2f},{y(q):.2f}) node[left]{{$q_{q}$}} -- ({xend:.2f},{y(q):.2f});")
    for q in range(k):
        s.append(rf"\node[g] at ({xh:.2f},{y(q):.2f}) {{$H$}};")
        s.append(rf"\node[g] at ({xz:.2f},{y(q):.2f}) {{$R_Z({coef(h[q])})$}};")
        s.append(rf"\node[g] at ({xr:.2f},{y(q):.2f}) {{$R_X(2\beta)$}};")
    for ci, col in enumerate(zz):
        x = x0 + ci * DX
        for lo, hi, c in col:
            s.append(rf"\draw ({x:.2f},{y(lo):.2f}) node[dot]{{}} -- ({x:.2f},{y(hi):.2f}) node[dot]{{}};")
            s.append(rf"\node[right,inner sep=1pt,font=\scriptsize] at ({x:.2f},{(y(lo)+y(hi))/2:.2f}) {{${coef(c)}$}};")
    s.append(r"\end{tikzpicture}")
    return "\n".join(s) + "\n"


for name, (h, J) in CIRCUITS.items():
    (OUT / f"{name}.tex").write_text(tikz(h, J))
    print(name, "columns:", 3 + len(layers(J)))
