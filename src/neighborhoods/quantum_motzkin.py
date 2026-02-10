"""Quantum Motzkin neighborhood - select non-crossing endpoint swaps using QUBO.

================================================================================
1. OVERVIEW
================================================================================

Given a permutation π = (π₀, π₁, ..., πₙ₋₁) representing the job order in
a Flow Shop, we select a set of endpoint-swap pairs (i,j) that together
minimize the total change in Cₘₐₓ.

Motzkin rules define which pairs can be selected simultaneously:
  - Disjoint pairs:   OK     (0,2) and (4,6) → arcs don't touch
  - Nested pairs:     OK     (0,5) and (2,3) → inner arc inside outer
  - Crossing pairs:   NO     (0,3) and (1,4) → arcs intersect
  - Shared endpoints: NO     (0,3) and (3,5) → share index 3

The problem is formulated as QUBO (Quadratic Unconstrained Binary Optimization),
solvable on a quantum computer (D-Wave) or classically (dimod SA).


================================================================================
2. ALGORITHM STEP BY STEP
================================================================================

STEP 1: Compute Head and Tail matrices — O(m·n)
────────────────────────────────────────────────
    Head[r][j] = completion time of job at position j on machine r
                 (forward propagation, respecting machine dependencies)

    Head[r][j] = max(Head[r-1][j], Head[r][j-1]) + p[r][π[j]]

    Tail[r][j] = remaining time from position j on machine r to the end
                 (backward propagation)

    Tail[r][j] = max(Tail[r+1][j], Tail[r][j+1]) + p[r][π[j]]

    Base Cₘₐₓ = Head[m-1][n-1]

    In code:
        Head = compute_head(pi, processing_times)     # O(m·n)
        Tail = compute_tail(pi, processing_times)     # O(m·n)
        base_c = Head[m-1][n-1]


STEP 2: Compute δₖ for each pair (i,j) — O(m·n³) total
───────────────────────────────────────────────────────
    For each candidate pair k = (iₖ, jₖ), iₖ < jₖ, we compute:

        δₖ = Cₘₐₓ(π after swapping π[iₖ] ↔ π[jₖ]) - Cₘₐₓ(π)

    Endpoint swap of segment [i..j] means:
        [π[i], π[i+1], ..., π[j-1], π[j]]
                    ↓
        [π[j], π[i+1], ..., π[j-1], π[i]]

    Computing δₖ WITHOUT recomputing full Cₘₐₓ (using Head + Tail):

        1. Take state after position i-1: col_prev = Head[·][i-1]
           (or zero vector if i=0)

        2. Propagate the new sequence through segment [i..j]:
           - Position i:   now has job π[j]  (swapped endpoint)
             col[r] = max(col[r-1], col_prev[r]) + p[r][π[j]]
           - Positions i+1..j-1: unchanged, propagate normally
           - Position j:   now has job π[i]  (swapped endpoint)
             col[r] = max(col[r-1], col_prev[r]) + p[r][π[i]]

        3. Combine with Tail from position j+1:
           new_Cₘₐₓ = max_r( col_prev[r] + Tail[r][j+1] )

        4. δₖ = new_Cₘₐₓ - base_Cₘₐₓ

    Complexity per δₖ: O(m · (j-i+1))
    Total for all pairs: O(m · Σᵢ Σⱼ (j-i+1)) = O(m · n³)


STEP 3: Filter candidates (classical optimization) — O(K)
──────────────────────────────────────────────────────────
    Discard pairs with δₖ ≥ 0 (no improvement).
    On a real QPU this step is unnecessary — the solver will set xₖ = 0
    for δₖ ≥ 0 since it never decreases H.


STEP 4: Build QUBO matrix Q — O(K²)
────────────────────────────────────
    The Q matrix encodes the binary optimization problem:

        H(x) = Σₖ Σₗ Q[k,l] · xₖ · xₗ    (xₖ ∈ {0,1})

    Where:
        H(x) = [cost of selected swaps] + [penalty for conflicts]
             = Σₖ δₖ · xₖ  +  P · Σ_{k<l, conflict(k,l)} xₖ · xₗ

    Matrix elements:
        Q[k,k] = δₖ           (linear: cost of selecting swap k)
        Q[k,l] = P             if conflict(k,l)  (crossing or shared endpoint)
        Q[k,l] = 0             if admissible(k,l) (disjoint or nested)

    Penalty weight:  P = Σₖ |δₖ| + 1
        Ensures P > max possible gain from any subset → solver never violates.


STEP 5: Solve QUBO — solver-dependent
──────────────────────────────────────
    Classical (current): dimod.SimulatedAnnealingSampler() with num_reads samples.
    Quantum (future):    DWaveSampler + EmbeddingComposite, O(μs) per read.


STEP 6: Extract selected swaps and validate — O(K²)
────────────────────────────────────────────────────
    1. Extract indices where xₖ = 1
    2. Post-validation: greedy removal of conflicts (SA may violate constraints)
    3. Fallback: if nothing selected, pick the single best swap


STEP 7: Apply swaps to permutation — O(K) + O(m·n)
───────────────────────────────────────────────────
    For each selected (i,j): new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    Motzkin-admissible swaps don't interfere (disjoint or nested endpoints).
    Final Cₘₐₓ is recomputed from scratch for verification.


================================================================================
3. FORMAL QUBO DEFINITION
================================================================================

Let K = number of candidate pairs, xₖ ∈ {0,1} for k = 0,...,K-1.

Hamiltonian (QUBO objective):

    H(x) = Σₖ δₖ · xₖ  +  P · Σ_{k<l} conflict(k,l) · xₖ · xₗ

Where conflict(k,l) ∈ {0,1}:

                    ⎧ 1  if {iₖ,jₖ} ∩ {iₗ,jₗ} ≠ ∅    (shared endpoint)
    conflict(k,l) = ⎨ 1  if iₖ < iₗ < jₖ < jₗ          (crossing)
                    ⎪ 1  if iₗ < iₖ < jₗ < jₖ          (crossing, symmetric)
                    ⎩ 0  otherwise                       (disjoint or nested)

Upper-triangular Q matrix:

    Q[k,l] = ⎧ δₖ                if k = l    (diagonal)
             ⎨ P · conflict(k,l)  if k < l    (off-diagonal)
             ⎩ 0                  if k > l    (lower triangle = 0, BQM convention)

Then:
    H(x) = xᵀ Q x = Σₖ Q[k,k] · xₖ + Σ_{k<l} Q[k,l] · xₖ · xₗ   (since xₖ² = xₖ)

Optimal solution:
    x* = argmin_{x ∈ {0,1}^K} H(x)


================================================================================
4. CONFLICT CONDITION — DETAILS
================================================================================

For two pairs (i₁,j₁) and (i₂,j₂) with i₁<j₁ and i₂<j₂:

    Case 1: Shared endpoint
        {i₁,j₁} ∩ {i₂,j₂} ≠ ∅
        E.g. (0,3) and (3,5) → share index 3
        CONFLICT: same position cannot be endpoint of two different swaps

    Case 2: Crossing
        i₁ < i₂ < j₁ < j₂  (WLOG i₁ < i₂)
        Visually — arcs cross above the axis:
            0───1───2───3───4
            └─────────┘         arc (0,3)
                └─────────┘     arc (1,4)
                    ✗ crossing!

    Case 3: Nesting
        i₁ < i₂ < j₂ < j₁  (inner arc fully inside outer)
        Visually:
            0───1───2───3───4───5
            └─────────────────┘     arc (0,5) — outer
                └─────┘             arc (2,3) — inner
                    ✓ nested → ADMISSIBLE

    Case 4: Disjoint
        j₁ < i₂  (arcs completely separate)
        Visually:
            0───1───2───3───4───5
            └───┘                   arc (0,1)
                        └───┘       arc (3,4)
                    ✓ disjoint → ADMISSIBLE


================================================================================
5. PENALTY P — WHY P = Σ|δₖ| + 1 SUFFICES
================================================================================

Goal: no solution with a conflict should be better than a conflict-free one.

Worst case: solver selects ALL K swaps (even conflicting ones).
    Gain = Σₖ δₖ, best possible gain ≤ Σ_{δₖ<0} |δₖ| ≤ Σₖ |δₖ|

If P > Σₖ |δₖ|, then ONE penalty term P·xₖ·xₗ costs more
than the total gain from ANY subset of swaps.
Hence solver never selects a conflicting pair.

In code: P = Σₖ |δₖ| + 1  ("+1" for numerical margin)


================================================================================
6. COMPARISON WITH OTHER QUBO NEIGHBORHOODS
================================================================================

    ┌───────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │                   │ Q. Adjacent      │ Q. Dynasearch    │ Q. Motzkin       │
    ├───────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Variables         │ n-1              │ n(n-1)/2         │ n(n-1)/2         │
    │ Swap type         │ (i, i+1)         │ (i, j) any       │ (i, j) any       │
    │ Constraint        │ ONE_HOT          │ NO_OVERLAP       │ NO_CROSSING      │
    │                   │ (exactly 1)      │ (no overlaps)    │ (no crossings)   │
    │ Nesting           │ N/A              │ ✗ forbidden      │ ✓ allowed        │
    │ #solutions        │ n-1              │ (varies)         │ Mₙ (Motzkin)     │
    │ Q graph density   │ O(n²) full       │ O(n⁴) dense      │ O(n⁴) sparser   │
    │ Penalty formula   │ P=2·max|δ|+1     │ P=Σ|δ|+1         │ P=Σ|δ|+1         │
    └───────────────────┴──────────────────┴──────────────────┴──────────────────┘

    Key difference Motzkin vs Dynasearch in QUBO:
        Dynasearch: conflict(k,l) = [max(iₖ,iₗ) ≤ min(jₖ,jₗ)]  → penalizes ALL overlaps (including nested)
        Motzkin:    conflict(k,l) = [crossing OR shared endpoint] → nesting allowed, sparser conflict graph


================================================================================
7. COMPLEXITY
================================================================================

    Stage                   | Complexity        | Comment
    ────────────────────────┼───────────────────┼──────────────────────────
    Head + Tail             | O(m·n)            | Two m×n matrices
    Deltas (STEP 2)         | O(m·n³)           | K=O(n²) pairs, O(m·(j-i)) each
    Filtering (STEP 3)      | O(K)              | Simple list
    Build Q (STEP 4)        | O(K²) = O(n⁴)    | K×K pairs, check conflict
    Solve QUBO              | Solver-dependent  | SA: heuristic; QPU: O(μs)
    Validation (STEP 6)     | O(K²)             | Greedy check
    Apply (STEP 7)          | O(K) + O(m·n)     | Swaps + full Cₘₐₓ
    ────────────────────────┼───────────────────┼──────────────────────────
    TOTAL                   | O(m·n³ + n⁴)     | Dominated by deltas or Q

    With L_max: K = O(n·L_max), deltas = O(m·n·L²), Q = O(n²·L²)


================================================================================
8. MOTZKIN NUMBERS — COMBINATORIAL STRUCTURE
================================================================================

    Mₙ = number of admissible configurations (sets of non-crossing pairs)
         from n ordered points.

    Recurrence:
        M₀ = 1,  M₁ = 1
        Mₙ = Mₙ₋₁ + Σ_{k=0}^{n-2} Mₖ · Mₙ₋₂₋ₖ

    Interpretation: point 0 can be:
        - Isolated (not in any pair) → Mₙ₋₁ configurations from points 1..n-1
        - Paired with point k+1 (k=0..n-2):
          → inside arc (0, k+1): Mₖ configurations from points 1..k
          → outside (to the right): Mₙ₋₂₋ₖ configurations from points k+2..n-1

    Values:
        n:   0  1  2  3   4   5    6    7     8     9     10
        Mₙ:  1  1  2  4   9  21   51  127   323   835   2188

    Asymptotics: Mₙ ~ (3/√(12π)) · 3ⁿ / n^{3/2}
"""

from typing import Dict, List, Optional, Tuple

from src.neighborhoods.common import (
    compute_endpoint_swap_delta,
    compute_head,
    compute_tail,
    solve_qubo,
)
from src.permutation_procesing import c_max


def _motzkin_conflict(i1: int, j1: int, i2: int, j2: int) -> bool:
    """Check if two pairs (i1,j1) and (i2,j2) conflict under Motzkin rules.

    Conflict occurs when:
      1. Shared endpoint: {i1,j1} ∩ {i2,j2} ≠ ∅
         → same position cannot be endpoint of two swaps simultaneously
      2. Crossing: i1 < i2 < j1 < j2  or  i2 < i1 < j2 < j1
         → arcs intersect

    Admissible (no conflict) when:
      1. Disjoint: j1 < i2  or  j2 < i1
         → arcs don't touch at all
      2. Nested: i1 < i2 < j2 < j1  (or symmetric)
         → inner arc fits inside outer
         → both swaps exchange only THEIR OWN endpoints, middle untouched
         → therefore no interference

    Mathematically, for i1 < i2 (WLOG):
        - i1 < i2 < j2 < j1  →  nesting  →  OK  (return False)
        - i1 < i2 < j1 < j2  →  crossing →  BAD (return True)
        - j1 < i2             →  disjoint →  OK  (return False)
        - j1 == i2            →  shared   →  BAD (return True)

    Args:
        i1, j1: First pair endpoints (i1 < j1).
        i2, j2: Second pair endpoints (i2 < j2).

    Returns:
        True  → pairs conflict (cannot select both simultaneously).
        False → pairs are admissible (Motzkin-compatible).
    """
    # ── Condition 1: Shared endpoint ──
    # If any index of one pair == any index of the other,
    # the same permutation position would be swapped twice.
    # E.g. (0,3) and (3,5): position 3 is endpoint of BOTH pairs → conflict.
    if i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2:
        return True

    # ── Condition 2: Crossing ──
    # Two arcs cross when one starts inside the other
    # but ends outside.
    # Visually:    i1────j1
    #                  i2────j2
    #               ← arcs intersect
    if i1 < i2 < j1 < j2:
        return True
    if i2 < i1 < j2 < j1:
        return True

    # ── No conflict ──
    # If we reach here, pairs are either disjoint or nested.
    # Both cases are admissible.
    return False


def _validate_motzkin_selection(
    selected: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Post-validation: greedily remove conflicting pairs.

    The QUBO solver (especially SimulatedAnnealingSampler) may return
    a solution violating Motzkin rules — it's a heuristic, not a guarantee.
    Therefore we perform greedy validation after QUBO:

    Algorithm:
        1. Sort pairs by left endpoint (i ascending)
        2. For each pair (i,j) in order:
           - Check if it conflicts with ANY already accepted pair
           - If not → add to admissible set
           - If yes → discard

    Sorting by left endpoint favors pairs closer to the start of permutation.
    This is not an optimal algorithm (that would be NP-hard in general
    for maximum weight independent set), but it's fast and sufficient.

    Complexity: O(K²) worst case (K = number of selected pairs)

    Args:
        selected: List of (i, j) pairs chosen by the QUBO solver.

    Returns:
        List of (i, j) pairs satisfying Motzkin rules.
    """
    if not selected:
        return []

    valid: List[Tuple[int, int]] = []
    for i, j in sorted(selected):
        conflict = False
        for vi, vj in valid:
            if _motzkin_conflict(i, j, vi, vj):
                conflict = True
                break
        if not conflict:
            valid.append((i, j))
    return valid


def quantum_motzkin_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
    L_max: Optional[int] = None,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Main function: quantum Motzkin neighborhood via QUBO.

    Steps overview:
        1. Head + Tail → O(m·n) matrices for fast Cₘₐₓ computation
        2. Deltas δₖ   → Cₘₐₓ change for each pair (i,j)
        3. Filtering    → discard pairs with δₖ ≥ 0 (classical optimization)
        4. Build Q      → QUBO matrix with penalties for crossing/shared
        5. Solve QUBO   → dimod SA (or QPU in the future)
        6. Validation   → greedy removal of any remaining conflicts
        7. Swaps        → apply selected swaps to the permutation

    QUBO: H(x) = Σₖ δₖ·xₖ + P·Σ_{conflict} xₖ·xₗ
        xₖ = 1 → selected swap k = (iₖ, jₖ)
        P = Σ|δₖ| + 1 → penalty for conflict

    Key difference vs quantum_dynasearch:
        Dynasearch: penalizes ALL overlapping intervals
        Motzkin:    penalizes ONLY crossing and shared endpoints
                    → nested pairs are allowed
                    → QUBO conflict graph is sparser
                    → solver finds optimum more easily

    Key difference vs quantum_adjacent (quantum_fibonahi):
        Adjacent:   variables are swaps (i, i+1), constraint ONE_HOT
        Fibonahi:   variables are swaps (i, i+1), constraint NO_OVERLAP (chain)
        Motzkin:    variables are swaps (i, j) any, constraint NO_CROSSING

    Args:
        pi: Current permutation (list of n jobs).
        processing_times: m×n processing times matrix p[r][job].
        num_reads: Number of SA samples (more = better quality, slower).
        L_max: Max segment length (j-i+1). None = all pairs O(n²).
               Classical optimization — unnecessary on QPU.

    Returns:
        Tuple (new_pi, new_cmax, selected_swaps):
            new_pi:         New permutation after applying swaps.
            new_cmax:       New Cₘₐₓ (recomputed from scratch — verification).
            selected_swaps: List of (i, j) pairs that were applied.
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Compute Head and Tail matrices — O(m·n)
    # ══════════════════════════════════════════════════════════════════════
    # Head[r][j] = completion time of job π[j] on machine r
    #   Head[r][j] = max(Head[r-1][j], Head[r][j-1]) + p[r][π[j]]
    # Tail[r][j] = remaining time from position j on machine r to the end
    #   Tail[r][j] = max(Tail[r+1][j], Tail[r][j+1]) + p[r][π[j]]
    # base_c = Head[m-1][n-1] = current Cₘₐₓ
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    m = len(processing_times)
    base_c = Head[m - 1][n - 1]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Compute δₖ for each pair (i,j) — O(m·n³) total
    # ══════════════════════════════════════════════════════════════════════
    # For pair k = (i,j): δₖ = Cₘₐₓ(π with swap π[i]↔π[j]) - Cₘₐₓ(π)
    # We use compute_endpoint_swap_delta(Head, Tail) instead of full
    # Cₘₐₓ recomputation → O(m·(j-i+1)) per pair.
    # L_max limits j-i+1 to reduce the number of QUBO variables.
    all_candidates: List[Tuple[int, int, float]] = []  # (i, j, delta)
    for i in range(n - 1):
        j_max = n - 1 if L_max is None else min(n - 1, i + L_max - 1)
        for j in range(i + 1, j_max + 1):
            delta = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
            all_candidates.append((i, j, delta))

    if not all_candidates:
        return pi.copy(), base_c, []

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Filter candidates — O(K)
    # ══════════════════════════════════════════════════════════════════════
    # Classical optimization: discard δₖ ≥ 0 (no Cₘₐₓ improvement).
    # The solver would set xₖ=0 for these anyway, since δₖ ≥ 0
    # never decreases H. But reducing QUBO size speeds up SA.
    # On a real QPU this step is unnecessary.
    candidates = [(i, j, d) for i, j, d in all_candidates if d < 0]

    # Fallback: no improving candidates
    if not candidates:
        best = min(all_candidates, key=lambda x: x[2])
        if best[2] < 0:
            new_pi = pi.copy()
            new_pi[best[0]], new_pi[best[1]] = new_pi[best[1]], new_pi[best[0]]
            return new_pi, c_max(new_pi, processing_times), [(best[0], best[1])]
        return pi.copy(), base_c, []

    num_vars = len(candidates)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Build QUBO matrix Q — O(K²)
    # ══════════════════════════════════════════════════════════════════════
    # H(x) = Σₖ δₖ·xₖ + P·Σ_{conflict(k,l)} xₖ·xₗ
    #
    # Penalty P = Σ|δₖ| + 1:
    #   Guarantees that one conflict costs more than the total gain
    #   from any subset of swaps. Mathematically:
    #     P > Σ|δₖ| ≥ max gain from subset → solver never violates rules.
    penalty = sum(abs(d) for _, _, d in candidates) + 1
    Q: Dict[Tuple[str, str], float] = {}

    # ── Diagonal elements Q[k,k] = δₖ ──
    # Each diagonal element encodes the linear cost of selecting swap k.
    # δₖ < 0 → improvement (solver wants xₖ=1)
    # δₖ > 0 → worsening (solver wants xₖ=0, but these are filtered above)
    for k in range(num_vars):
        Q[(f"x{k}", f"x{k}")] = candidates[k][2]  # delta_k

    # ── Off-diagonal elements Q[k,l] = P if conflict(k,l) ──
    # Penalty P for CROSSING or SHARED-ENDPOINT pairs.
    # Nested pairs → no penalty (Q[k,l] = 0, entry not added).
    # This is the key difference from quantum_dynasearch, where overlap → penalty.
    for k in range(num_vars):
        i1, j1, _ = candidates[k]
        for l in range(k + 1, num_vars):
            i2, j2, _ = candidates[l]
            if _motzkin_conflict(i1, j1, i2, j2):
                Q[(f"x{k}", f"x{l}")] = penalty

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Solve QUBO — solver
    # ══════════════════════════════════════════════════════════════════════
    # Current implementation: dimod.SimulatedAnnealingSampler (classical)
    # On D-Wave QPU: replace with DWaveSampler + EmbeddingComposite
    # solve_qubo returns dict {"x0": 0/1, "x1": 0/1, ...}
    solution = solve_qubo(Q, num_reads)
    selected_indices = sorted(int(v[1:]) for v, val in solution.items() if val == 1)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Validation + Fallback — O(K²)
    # ══════════════════════════════════════════════════════════════════════
    # SA may return a sub-optimal solution violating constraints.
    # Greedy validation: iterate over pairs sorted by i,
    # keep a pair if it doesn't conflict with any already accepted pair.
    selected_pairs = [(candidates[k][0], candidates[k][1]) for k in selected_indices]
    valid_swaps = _validate_motzkin_selection(selected_pairs)

    # Fallback: if solver selected nothing useful,
    # pick the single swap with the best (lowest) δₖ.
    if not valid_swaps:
        best_k = min(range(num_vars), key=lambda k: candidates[k][2])
        if candidates[best_k][2] < 0:
            valid_swaps = [(candidates[best_k][0], candidates[best_k][1])]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Apply swaps to permutation — O(K) + O(m·n)
    # ══════════════════════════════════════════════════════════════════════
    # Motzkin-admissible endpoint swaps don't interfere:
    #   - Disjoint: operate on different positions
    #   - Nested: both swap only ENDPOINTS — middle untouched
    # Sort by i so the application order is deterministic.
    # At the end, recompute full Cₘₐₓ (verification, not by delta).
    new_pi = pi.copy()
    for i, j in sorted(valid_swaps, key=lambda x: x[0]):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps
