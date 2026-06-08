# Research ideas — potential future directions

Notes for future papers / experiments. Not part of any current submission.

Sources:
- Whiteboard session 2026-06-04 (photos in project archive).
- Code in `src/neighborhoods/` and articles `windowed_qubo`, `hybrid_neighborhoods_icaisc`, `motzkin_soco`.

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
The number of valid subsets of $\{1,\dots,n-1\}$ then satisfies a $k$-step linear recurrence (**k-bonacci numbers**):
$$
a_n \;=\; a_{n-1} + a_{n-2} + \dots + a_{n-k}.
$$

**Concrete sequences** (from whiteboard 2026-06-04):

| $n$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|---|
| Fibonacci $F_n$  | 1 | 1 | 2 | 3 | 5 | 8 | 13 | 21 | 34 | 55 |
| Tribonacci $T_n$ | 1 | 1 | 2 | 4 | 7 | 13 | 24 | 44 | 81 | 149 |

**Asymptotic growth** of $a_n$:

| $k$ | $\lambda_k$ | name              | constraint               |
|-----|-------------|-------------------|--------------------------|
| 2   | **1.6180**  | golden ratio      | Fibonacci (current)      |
| 3   | **1.8393**  | tribonacci const. | Tribonacci               |
| 4   | 1.9276      | tetranacci const. | Tetranacci               |
| $\infty$ | 2.0    | —                 | unconstrained ($2^{n-1}$ subsets) |

**Why study it?**
- Larger $k$ → richer neighborhood at the same per-iteration cost ($O(mn^2)$ for deltas + $O(nk)$ DP — still essentially linear in $n$).
- Trade-off: bigger neighborhood means more swap interactions per iteration, but bigger risk of two swaps fighting over the same Cmax-bottleneck job.
- Open empirical question: does the per-iteration improvement keep growing with $k$, or saturate quickly? Is there an optimal $k^*$?

**DP recurrence.** State tracks the last $k-1$ "taken/skipped" flags:
$$
\mathrm{dp}[i, \text{tail}] = \min\!\bigl(
  \mathrm{dp}[i+1, \text{shift}(\text{tail}, 0)],
  \;\;\Delta_i + \mathrm{dp}[i+1, \text{shift}(\text{tail}, 1)] \cdot [\text{tail not all 1}]
\bigr).
$$

---

## 2. Dynasearch in matrix form — disjoint squares spanned by ones

**Premise.** Dynasearch picks a set of non-overlapping segment swaps $(i,j)$, $i < j$, that minimize $\sum \Delta_{i,j}$. The current implementation enumerates $O(n^2)$ candidate pairs and uses a DP / recursion.

**Matrix encoding.** Map each swap $(i,j)$ to the unit-square cell at position $(i,j)$ in an upper-triangular $n \times n$ binary matrix $X = [x_{ij}]$. A valid composite move is a subset of cells (1's) such that the corresponding segments $[i,j]$ are pairwise disjoint $\Rightarrow$ in matrix view: **disjoint squares spanned by ones**, each square aligned with the diagonal.

Visually (whiteboard sketch):
```
  j: 0 1 2 3 4 5 6 7
i=0  . . . 1 . . . .      swap (0,3)  ━┓
i=1  . . . . . . . .                   ┃
i=2  . . . . . . . .                   ┃ disjoint
i=3  . . . . . . . .                   ┃ squares
i=4  . . . . . 1 . .      swap (4,5)  ━┻━┓
i=5  . . . . . . . 1      swap (5,7)  ━━━┻━ ✗ overlaps with (4,5)
```

**Pairwise non-overlap constraints** (whiteboard concrete form):
$$
x_{i,i+2} + x_{i+1,i+3} \;\leq\; 1, \quad
x_{i+2,i} + x_{i+3,i+1} \;\leq\; 1, \quad \dots
$$
Each constraint forbids one specific overlap pattern; the full set forces compatibility.

**QUBO formulation.** Binary variables $x_{ij} \in \{0,1\}$ for each cell, objective
$$
\min \;\sum_{i<j} \Delta_{i,j} \, x_{ij}
\;+\; P \sum_{\text{overlapping } (i_1,j_1),(i_2,j_2)} x_{i_1 j_1}\, x_{i_2 j_2}.
$$
- Quadratic penalty $P$ enforces compatibility.
- Total variables: $\binom{n}{2}$ — fits a windowed D-Wave decomposition for $n \leq 50$.
- Coupling graph has clean geometric structure (upper-triangular, banded by segment length).

**Open questions.**
- Compare classical DP dynasearch vs.\ QUBO-encoded dynasearch on D-Wave for small $n$.
- Does windowed decomposition (already in `windowed_qubo` for `dyn_enhanced`) recover the same solutions, or does the matrix view reveal new structure?
- Can the same disjoint-squares view extend to Motzkin (with the nesting allowance encoded as a relaxed compatibility relation)?

---

## 3. Quantum Dynamic Programming (special session paper)

**Working title.** "Quantum Dynamic Programming for permutational problems & discrete optimization."

**Co-authors (whiteboard list).** Kotzbach, Gnatowski/Bożejko, S. Trotskyi, D. Choptiany, R. Wojewódzki, I. Dudkiewicz. Confirm exact spelling and order with W. Bożejko.

**Premise.** Classical DP for permutation neighborhoods (Fibonacci, Motzkin, Dynasearch) all share a common pattern: enumerate $O(n^k)$ candidate moves, then pick the optimal compatible subset by a recurrence. The recurrence itself can be *quantum-accelerated* by encoding the DP state as a superposition of partial subsets and performing the min/sum step adiabatically or via QAOA.

**Building blocks.**
- Bra-ket encoding of DP state: $|s\rangle = $ superposition over partial selections; transitions like $|0\rangle\langle 0|$ project onto "skipped" branches.
- Cost factorization $\sum w_i T_i$ encoded as a Hamiltonian, e.g. for tardiness:
  $$T_i = \max(C_i - d_i, 0), \quad C_i = \sum_{j \leq i} p_{\pi(j)}.$$
- Adiabatic schedule reads off the minimum via ground-state evolution, or QAOA mixes layers of cost / driver Hamiltonians.
- Factorization of $\log(N!)$ ↔ encoding cost: total qubit count $\sim n \log n$ for permutation indices.

**Concrete experiments to attempt.**
- Fibonacci DP as adiabatic ground-state search on a chain of $n-1$ qubits with nearest-neighbor coupling penalizing $x_i x_{i+1}$.
- Tribonacci version: same chain, but penalize $x_i x_{i+1} x_{i+2}$ — requires 3-body terms or auxiliary qubits.
- Dynasearch matrix form (section 2): $\binom{n}{2}$ qubits + overlap penalty.
- Compare ground-state energy from D-Wave / IQM Spark / QAOA simulator vs.\ classical DP solution.

**Why a special session?**
- Naturally bridges combinatorial DP (Smutnicki, Glover, Congram) and adiabatic / variational quantum optimization (Farhi, D-Wave samplers).
- Each elementary neighborhood from our prior papers becomes a *case study* — Fibonacci (linear-chain QUBO), Tribonacci (3-body coupling), Dynasearch (banded $\binom{n}{2}$ QUBO), Motzkin (with nested-arc constraints).
- Output of one paper, multiple case studies, multiple co-authors.

---

## Conference deadlines (whiteboard)

For tracking — verify dates against official call-for-papers before committing.

| Venue | Event date | Submission deadline | Theme |
|-------|------------|---------------------|-------|
| SOCO 2026 | 18–19.06 | 16.02 | Soft Computing in Industrial / Environmental |
| ICCCI | — | 15.02 | Computational Collective Intelligence |
| KAPD | — | 30.01 | (production / scheduling?) |
| AISC 2024 | 18.06 | 15.01 | Artificial Intelligence and Soft Computing |
| Fibonacci workshop | 22–27.12 | — | (Fibonacci-themed?) |

Open question: which of these is the target for the **Quantum DP special session**? Likely a Springer LNCS-track event with explicit "Special Session on Quantum Computing" or similar.
