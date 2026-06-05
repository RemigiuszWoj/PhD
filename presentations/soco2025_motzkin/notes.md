# SOCO 2026 — Speaker Notes

*Motzkin Neighborhood Evaluation in Iterated Local Search and Simulated Annealing for the Permutational Flow Shop Problem*
Wojewódzki & Bożejko — CCIS Vol. 3046

Time target: ~15 min talk + 5 min Q&A → ~1.5 min per slide on average.

---

## Slide 1 — Title (~30 s)

- Greet, introduce yourself, mention co-author Prof. Bożejko (Wrocław University of Science and Technology).
- One sentence: this paper introduces the Motzkin neighborhood — a new composite move based on non-crossing arcs — and compares it experimentally with the three classical neighborhoods.
- Acknowledge the conference and the proceedings (CCIS 3046).

---

## Slide 2 — Outline (~20 s)

- Walk through five sections: Introduction → Motzkin neighborhood definition → DP algorithm → experiments → conclusion.
- Position the talk: the *Motzkin* box in the diagram on the next slide is what this paper contributes; everything else is context.

---

## Slide 3 — Introduction (~1 min)

- Restate PFSP in one sentence: same job order on every machine, minimize makespan, NP-hard from m=3.
- Walk along the 4-box arrow: Adjacent (cheap, single swap), Fibonacci (compound disjoint swaps), Dynasearch (deep but expensive segment swaps), Motzkin (this work, non-crossing arcs).
- Frame the question: does Motzkin's combinatorial richness (non-crossing arc structure) translate into better solutions under fixed time?

---

## Slide 4 — Motzkin Numbers and Neighborhood Structure (~1.5 min)

- Motzkin numbers $M_n$ count non-crossing arc configurations on $n$ ordered points; isolated points are allowed.
- Show the small example: 7 points, arcs (0,4), (2,3), (5,6) — none cross, two are at the top level, one is short. This is one admissible composite move.
- The right plot illustrates that $M_n$ grows as $\Theta(3^n / n^{3/2})$ — exponential, so enumeration is hopeless beyond small $n$. We need the DP from the next slide.

---

## Slide 5 — Admissibility Rules (~1.5 min)

- A composite move is a set of (i, j) pairs. Three rules decide admissibility:
  - **Disjoint** (left figure): two arcs sit side by side — OK.
  - **Crossing** (middle, red): forbidden — that is exactly the non-crossing constraint behind the Motzkin count.
  - **Nested** (right): an inner arc fully inside an outer arc — explicitly allowed, distinguishing Motzkin from simpler "non-overlapping" neighborhoods.
- Plus index disjointness: no two pairs share an endpoint.
- Each pair (i, j) means an *end swap* of the segment from position i to position j — the two endpoints exchange and everything between them stays untouched. So a composite move is multiple endpoint-swaps applied at once.

---

## Slide 6 — Constructing the Best Composite Move (~1.5 min)

- Algorithm runs once per local-search iteration:
  1. Precompute Head and Tail completion-time matrices — O(mn).
  2. Compute the makespan delta for every (i, j) pair — O(mn²) using Head+Tail.
  3. DP over those deltas picks the optimal admissible subset — O(n³).
  4. Apply selected swaps and recompute Cmax.
- Trade-off: one Motzkin iteration explores up to $M_n$ configurations in O(n³) time. Adjacent, in contrast, does n−1 cheap swaps in O(mn).
- Practical consequence: Motzkin runs fewer iterations than Adjacent under a fixed time budget, but each iteration carries more information.

---

## Slide 7 — Experimental Setup (~1 min)

- Standard Taillard instances, 12 instance classes (n ∈ {20, 50, 100, 200, 500}, m ∈ {5, 10, 20}).
- Six wall-clock budgets from 100 ms to 10 s — covers both short and long runs.
- ILS with a short tabu list (τ = 10).
- SA: T₀ = 1000, T_min = 1, α = 0.95, reheat factor r = 1.5, stagnation threshold 500 ms.
- Metric: mean RPD to lower bound, in percent.
- All four classical neighborhoods (Adjacent, Fibonacci, Dynasearch, Motzkin) compared on identical configurations.

---

## Slide 8 — Results: Gap vs. Time Limit (~1.5 min)

- Two log-scale plots, ILS on the left, SA on the right. X axis: time limit in ms; Y axis: mean RPD %.
- **Dynasearch (green)** sits at the bottom of both plots — best gap across all budgets.
- **Motzkin (red)** sits between Dynasearch and Adjacent — consistently better than Adjacent at larger $t_{\max}$.
- **Adjacent (blue)** is slow but improves steadily as the budget grows.
- **Fibonacci (orange)** converges fastest (flat line early) but plateaus highest — it finds a local optimum quickly and stagnates.
- Pattern is the same in TS and SA, which suggests it is a property of the neighborhood, not the metaheuristic.

---

## Slide 9 — Statistical Significance (Wilcoxon) (~1 min)

- Wilcoxon signed-rank applied to mean gaps across the 12 instance classes.
- Dynasearch is significantly better than every other neighborhood at every time budget ($p < 0.01$) — the strongest statistical claim in the paper.
- Motzkin beats Fibonacci with statistical significance ($p < 0.05$) once $t_{\max} \geq 5$ s — the Motzkin advantage only emerges when the algorithm has time to amortize the larger per-iteration cost.
- Motzkin vs. Adjacent is **not** statistically significant. Two competing effects: Motzkin runs few rich moves, Adjacent runs many simple ones — they cancel out across 12 configurations. The limited sample size also contributes.

---

## Slide 10 — Conclusion and Future Work (~1.5 min)

- **Summary in two sentences:**
  - Motzkin gives a structured multi-swap neighborhood based on non-crossing arcs, with exponential reach explored at $O(n^3)$ per iteration via DP.
  - Empirically: beats Adjacent at larger $t_{\max}$ in both ILS and SA; falls behind Dynasearch.
- **Future work — three concrete directions:**
  - Improve the computational efficiency of Motzkin (incremental delta evaluation) so it can run more iterations per budget.
  - Quantum-assisted Motzkin evaluation: D-Wave QUBO formulation, windowed decomposition, and QAOA on small instances.
  - Hybrid neighborhood strategies — cyclic / adaptive selection between Motzkin and the others (covered in the ICAISC companion paper).
- Thank the audience and invite questions.

---

## Likely Q&A topics — prep answers

- **"Why call the arcs 'Motzkin'?"**
  — Because the number of admissible composite moves on $n$ points equals the Motzkin number $M_n$. The counting structure (non-crossing, isolated points allowed) is exactly what Motzkin numbers enumerate.

- **"Why does Motzkin allow nesting but Fibonacci does not?"**
  — The Fibonacci version we use is the simpler "non-overlapping adjacent swap" variant ($F_{n+1}$ subsets). Motzkin generalizes by allowing arbitrary $(i, j)$ pairs with nesting, which is why it covers a strictly richer move set.

- **"Did you try Motzkin with a longer time budget than 10 s?"**
  — Within this paper, no — the budgets were chosen to mirror typical metaheuristic comparisons. Longer budgets are tested in the companion windowed-QUBO paper, where Motzkin at $n = 500$ becomes computationally intractable in the classical setting.

- **"How does Motzkin scale at large $n$?"**
  — At $n = 500$ the $O(n^3)$ DP becomes expensive enough that ILS effectively completes a single iteration. Quantum-assisted evaluation (D-Wave QUBO with windowed decomposition) is the planned remedy.

- **"What about the gap to NEH or other state-of-the-art constructive heuristics?"**
  — Out of scope here. The goal of this paper is a controlled neighborhood comparison inside ILS and SA. Comparison to NEH / IGA is future work.
