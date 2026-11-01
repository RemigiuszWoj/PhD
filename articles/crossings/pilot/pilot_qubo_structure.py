"""QUBO structure for Motzkin vs involution vs pair-corrected models on the same improving candidates.
D-Wave: minorminer embedding on Pegasus P16 (Advantage). Gate: QAOA p=1 transpiled on a 4x5 grid (Garnet-like)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
RES = HERE / "results"
import json, random, sys, time, warnings
warnings.filterwarnings("ignore")
import networkx as nx, minorminer, dwave_networkx as dnx
from src.permutation_procesing import c_max
from pilot_anatomy import lcg_instance, all_deltas, apply
from crossing_select import select_c0

def states(n, m, seed, want):
    p = lcg_instance(seed, n, m); pi = list(range(n)); cur = c_max(pi, p); out = []
    for step in range(200):
        base, D = all_deltas(pi, p); C = {ab: w for ab, w in D.items() if w < 0}
        if not C: break
        out.append((step, pi[:], C, base))
        v, mv = select_c0(C); q = apply(pi, mv); qc = c_max(q, p)
        if qc >= cur:
            ab = min(C, key=C.get); q = apply(pi, [ab]); qc = c_max(q, p)
        pi, cur = q, qc
    return p, out

def graphs(pi, p, C, base, with_pq):
    arcs = sorted(C); G_m, G_i, G_q = nx.Graph(), nx.Graph(), nx.Graph()
    for g in (G_m, G_i, G_q): g.add_nodes_from(range(len(arcs)))
    for x in range(len(arcs)):
        a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            shared = len({a, b, c, d}) < 4
            (s1, e1), (s2, e2) = sorted([arcs[x], arcs[y]])
            cross = (not shared) and s1 < s2 < e1 < e2
            if shared or cross: G_m.add_edge(x, y)
            if shared: G_i.add_edge(x, y)
            if with_pq:
                if shared: G_q.add_edge(x, y)
                else:
                    q = (c_max(apply(pi, [arcs[x], arcs[y]]), p) - base) - C[arcs[x]] - C[arcs[y]]
                    if q != 0: G_q.add_edge(x, y)
    return G_m, G_i, G_q

P16 = dnx.pegasus_graph(16)
def embed(G, timeout=60):
    if G.number_of_edges() == 0: return G.number_of_nodes(), 1
    t0 = time.time(); emb = minorminer.find_embedding(G.edges(), P16.edges(), timeout=timeout, random_seed=1)
    if not emb: return None, None
    return sum(len(c) for c in emb.values()), max(len(c) for c in emb.values())

def qaoa_2q(G):
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
    K = G.number_of_nodes()
    qc = QuantumCircuit(K)
    qc.h(range(K))
    for (u, v) in G.edges(): qc.rzz(0.3, u, v)
    for u in range(K): qc.rz(0.2, u)
    qc.rx(0.4, range(K))
    cmap = CouplingMap.from_grid(4, 5)
    tq = transpile(qc, coupling_map=cmap, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3, seed_transpiler=1)
    ops = tq.count_ops()
    return ops.get("cz", 0), tq.depth()

rng = random.Random(2026)
rows = []
for (n, m) in [(20, 5), (20, 10), (20, 20), (50, 10)]:
    for rep in range(2):
        seed = rng.randrange(1, 2**31 - 1)
        p, S = states(n, m, seed, None)
        picks = [S[0]] + ([S[len(S) // 2]] if len(S) > 2 else [])
        for kind, (step, pi, C, base) in zip(("start", "traj"), picks):
            Gm, Gi, Gq = graphs(pi, p, C, base, with_pq=len(C) <= 60)
            r = {"cls": f"{n}x{m}", "kind": kind, "K": len(C), "E_motz": Gm.number_of_edges(), "E_inv": Gi.number_of_edges(),
                 "E_pq": Gq.number_of_edges() if len(C) <= 60 else None}
            if len(C) <= 300:
                r["emb_motz"] = embed(Gm); r["emb_inv"] = embed(Gi)
                if len(C) <= 60: r["emb_pq"] = embed(Gq)
            if len(C) <= 20:
                r["qaoa_motz"] = qaoa_2q(Gm); r["qaoa_inv"] = qaoa_2q(Gi); r["qaoa_pq"] = qaoa_2q(Gq)
            rows.append(r); print(r, flush=True)
json.dump(rows, open(RES / "qubo_structure.json", "w"))
