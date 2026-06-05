# Research ideas — potential future directions

Notes for future papers / experiments. Not part of any current submission.

---

## 1. k-bonacci / Tribonacci neighborhoods

**Premise.** The Fibonacci neighborhood (composite disjoint-adjacent swap) is defined by the constraint
$$
i \in S \;\Longrightarrow\; i+1 \notin S
$$
i.e. no *two* consecutive adjacent-swap positions selected. The number of valid subsets equals the Fibonacci number $F_{n+1}$ — hence the nickname.

**Generalization.** Relax the constraint to "no $k$ consecutive positions selected":
$$
\{i, i+1, \dots, i+k-1\} \not\subseteq S \quad \text{for every } i.
$$
The number of valid subsets of $\{1,\dots,n-1\}$ then satisfies a $k$-step linear recurrence (the **k-bonacci numbers**):
$$
a_n \;=\; a_{n-1} + a_{n-2} + \dots + a_{n-k}.
$$

- $k = 2$ → Fibonacci ($a_n = a_{n-1} + a_{n-2}$), the current paper.
- $k = 3$ → **Tribonacci** ($a_n = a_{n-1} + a_{n-2} + a_{n-3}$). First terms: 1, 1, 2, 4, 7, 13, 24, 44, 81, 149, …
- $k = 4$ → Tetranacci, etc.

Asymptotic growth rate $\lambda_k$ of $a_n$:

| $k$ | $\lambda_k$ | constraint               |
|-----|-------------|--------------------------|
| 2   | 1.6180      | golden ratio (Fibonacci) |
| 3   | 1.8393      | tribonacci constant      |
| 4   | 1.9276      | tetranacci constant      |
| $\infty$ | 2.0    | unconstrained (all $2^{n-1}$ subsets) |

**Why study it?**
- Larger $k$ → richer neighborhood at the same per-iteration cost ($O(mn^2)$ for deltas + $O(nk)$ DP — still essentially linear in $n$).
- Trade-off: bigger neighborhood means more swap interactions per iteration, but bigger risk of two swaps fighting over the same Cmax-bottleneck job.
- Empirical question: does the per-iteration improvement keep growing with $k$, or saturate quickly?

**Open questions.**
- Is there a $k^*$ that gives the best PRD-vs-time trade-off across Taillard sizes?
- How does $k$ interact with $n/m$ ratio (machine-bound vs job-bound instances)?
- DP recurrence: track the last $k-1$ "taken/skipped" flags in the state — straightforward generalization.

**Connection to other neighborhoods.**
- $k \to \infty$ recovers an unconstrained subset, i.e. "apply every improving swap simultaneously" — essentially a single-step Newton-style move. The non-overlap constraint then has to be enforced post hoc (overlapping swaps cancel each other's deltas).

---

## 2. Dynasearch in matrix form — disjoint squares spanned by ones

**Premise.** Dynasearch picks a set of non-overlapping segment swaps $(i,j)$, $i < j$, that minimize $\sum \Delta_{i,j}$. The current implementation enumerates $O(n^2)$ candidate pairs and uses a DP / recursion.

**Matrix encoding.** Map each swap $(i,j)$ to the unit-square cell $M[i,j]$ in an upper-triangular $n \times n$ binary matrix. A valid composite move is a subset of cells (each cell = a 1) such that:

- **Non-overlap of segments**: the intervals $[i,j]$ for selected cells are pairwise disjoint $\Rightarrow$ in matrix terms, selected 1's are *axis-aligned disjoint squares* (each cell at $(i,j)$ "spans" the rectangle of indices $[i \dots j]$ both ways).

Visually:
```
   j: 0 1 2 3 4 5 6 7
i=0   . . . 1 . . . .       swap (0,3)
i=1   . . . . . . . .
i=2   . . . . . . . .
i=3   . . . . . . . .
i=4   . . . . . 1 . .       swap (4,5)
i=5   . . . . . . . 1       swap (5,7)  — overlaps with (4,5)! ✗
```

Two cells $(i_1, j_1)$ and $(i_2, j_2)$ are **compatible** iff $j_1 < i_2$ or $j_2 < i_1$ (one strictly to the right of the other on the diagonal).

**Reformulation as QUBO.** Binary variables $x_{ij} \in \{0,1\}$ for each cell, objective
$$
\min \;\sum_{i<j} \Delta_{i,j} \, x_{ij}
\;+\; P \sum_{\text{overlapping } (i_1,j_1),(i_2,j_2)} x_{i_1 j_1}\, x_{i_2 j_2}.
$$

- Quadratic penalty $P$ enforces compatibility (no two overlapping selected).
- Total variables: $\binom{n}{2}$ — fits a windowed D-Wave decomposition for $n \leq 50$.
- The coupling graph has a clear geometric structure (disjoint upper-triangular cells) that may admit efficient embeddings.

**Why study it?**
- Native QUBO formulation opens dynasearch to D-Wave QPU.
- Geometric "disjoint squares" interpretation gives a clean visual / pedagogical hook.
- Possible polynomial-time relaxation: replace the quadratic overlap penalty with a flow constraint on a DAG, reducing to longest-path / shortest-path on $O(n^2)$ nodes.

**Open questions.**
- Compare classical DP dynasearch vs.\ QUBO-encoded dynasearch on D-Wave for small $n$.
- Does the windowed decomposition (used for dyn\_enhanced in `windowed_qubo` article) recover essentially the same solutions, or does the matrix view reveal new structure?
- Can the same "disjoint squares" view be applied to Motzkin (with the extra nesting allowance encoded as a relaxed compatibility relation)?
