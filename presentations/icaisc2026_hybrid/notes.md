# ICAISC 2026 — Speaker Notes

*Hybrid Neighborhood Evaluation in Tabu Search and Simulated Annealing for the Permutation Flow Shop Problem*
Wojewódzki & Bożejko

Time target: ~15 min talk + 5 min Q&A → ~1.5 min per slide on average.

---

## Slide 1 — Title (~30 s)

- Greet, introduce yourself, mention co-author Prof. Bożejko (Wrocław University of Science and Technology).
- One sentence on motivation: PFSP is a textbook NP-hard scheduling problem, and the choice of neighborhood structure inside a metaheuristic can matter more than the metaheuristic itself.
- Acknowledge the conference.

---

## Slide 2 — Outline (~20 s)

- Walk through the five sections: problem → three neighborhoods → two metaheuristics (TS, SA) → experiments → conclusion.
- Mention that the talk emphasizes the middle neighborhood — Composite Disjoint-Adjacent, nicknamed *Fibonacci* — because it turns out to be the practical winner for medium-to-large instances.

---

## Slide 3 — Introduction (~1 min)

- Restate PFSP in one sentence: same job order on every machine, minimize makespan, NP-hard from m=3.
- Use the box diagram from left to right:
  - **Adjacent**: the cheapest possible move, only swaps consecutive jobs — fast iterations but each step gains little.
  - **Fibonacci**: middle ground — bundles many disjoint adjacent swaps into one iteration.
  - **Dynasearch**: most expressive — non-overlapping arbitrary segment swaps, but expensive per iteration.
- The whole talk is about which of these wins under a wall-clock time budget.

---

## Slide 4 — Composite Disjoint-Adjacent (a.k.a. Fibonacci) (~1.5 min)

- Definition is purely set-theoretic: pick a subset of swap positions where no two are consecutive — that gives a valid non-overlapping composite move.
- Use the figure: swap at 0–1, 2–3, 5–6 — none of these arcs touch, all applied at the same time.
- Note the counting argument: the number of such subsets in a permutation of length n equals the Fibonacci number F_{n+1}. That is where the nickname comes from — neither the moves nor the positions follow the Fibonacci sequence.
- Optimal subset (smallest sum of deltas) chosen by a single linear-time DP over the precomputed n−1 swap deltas.
- This is essentially a one-iteration approximation of multiple greedy adjacent swaps applied together.

---

## Slide 5 — Cost Comparison (~1.5 min)

- Read the table top to bottom:
  - Adjacent — n−1 candidate swaps, O(mn) per iteration.
  - Fibonacci — same n−1 candidates, but the iteration cost is O(mn²) because each delta requires an O(mn) makespan recomputation (in the basic version evaluated here).
  - Dynasearch — O(n²) candidate segments, O(mn³) per iteration.
- Stress the take-away under the table: Fibonacci's *neighborhood size* is exponential (F_{n+1} subsets), but the iteration cost stays merely quadratic in n. That is what makes it the sweet spot.
- Mention briefly that incremental delta evaluation could bring Fibonacci down to O(mn) per iteration — listed in future work.

---

## Slide 6 — Tabu Search & Simulated Annealing (~1.5 min)

- The two metaheuristics are deliberately classical so that any performance gap comes from the neighborhood.
- TS: tabu list keyed by the move (or composite signature for Fibonacci / Dynasearch), short tenure τ=10, aspiration if a tabu move improves the global best.
- SA: multiplicative cooling. Highlight the **time-based reheating**: if no improvement for 500 ms and the temperature has dropped below T₀, multiply T by r=1.5 (capped at T₀). This counteracts the plateau risk of composite moves.
- Acceptance is standard Metropolis.
- Mention that both algorithms run on identical (n, m, tl) triples for fairness.

---

## Slide 7 — Experimental Setup (~1 min)

- Standard Taillard set, 12 instance classes spanning n ∈ {20, 50, 100, 200, 500} and m ∈ {5, 10, 20}.
- Six wall-clock budgets from 100 ms to 10 s — covers both quick local-search style runs and "full second" budgets.
- SA hyperparameters: T₀=1000, T_final=1, α=0.95, reheat r=1.5, stagnation threshold 500 ms — chosen once and held fixed.
- Metric: relative percentage deviation (PRD) of obtained makespan to the best-known reference, in percent.
- Hardware: MacBook Pro M4 Pro — modest but consistent; absolute timings would be a couple of times faster on a server, but the ranking is platform-independent.

---

## Slide 8 — Results table at tl=5000 ms (~1.5 min)

- Show the rows top to bottom:
  - For small n (20, 50, 100), Adjacent or Dynasearch win — Adjacent because it gets many cheap iterations, Dynasearch because it can still finish one full iteration with depth.
  - At n=200 (both m=10 and m=20), **Fibonacci becomes the best** — the crossover.
  - At n=500, Adjacent edges back slightly for TS, but Fibonacci remains competitive within ~1 pp.
- Two columns per neighborhood — TS and SA values are nearly identical for Fibonacci and Dynasearch (the composite-move oracle leaves SA almost no randomness), while Adjacent shows meaningful TS-vs-SA differences because SA visits many neighbors per second.
- Bold values mark the per-row best.

---

## Slide 9 — Convergence at tl=1000 ms (~1.5 min)

- Two real convergence plots from the paper, taken on representative instances.
- Left (tai200_20, n=200, m=20): Fibonacci (magenta) drops makespan fastest in the first ~100 ms, then plateaus. Adjacent (cyan) makes slow steady progress. Dynasearch (orange) barely starts — only manages a couple of iterations within the 1 s budget.
- Right (tai500_20, n=500, m=20): the gap is more dramatic. Fibonacci settles quickly, Adjacent crawls, Dynasearch is essentially flat after one iteration.
- Use this slide to make the qualitative point: **Fibonacci's early-budget advantage is what makes it the best default for short to medium time limits at large n.**
- Reheating in SA (not shown here) would partially extend Fibonacci's curve downward by escaping the plateau.

---

## Slide 10 — Conclusion and Future Work (~1.5 min)

- Recap in two sentences:
  1. Three neighborhoods evaluated inside TS and SA on Taillard.
  2. Composite Disjoint-Adjacent ("Fibonacci") is the practical winner for n ≥ 200 under short-to-medium budgets; Dynasearch wins only when given ample time.
- Future work — three concrete directions:
  - Incremental delta evaluation to cut Fibonacci's per-iteration cost from O(mn²) toward O(mn).
  - Adaptive switching between neighborhoods at runtime (e.g. start with Fibonacci, switch to Dynasearch when the budget allows).
  - Benchmarking against state-of-the-art metaheuristics (iterated local search, variable neighborhood descent).
- Thank the audience and invite questions.

---

## Likely Q&A topics — prep answers

- **"Why call it Fibonacci if there is no Fibonacci sequence in the move?"**
  — The name reflects the *size* of the neighborhood: the number of non-overlapping subsets of n−1 swap positions equals F_{n+1}. It is a counting coincidence, kept as a memorable internal nickname.

- **"How does SA's reheating compare to standard random restarts?"**
  — Reheating only raises the temperature when stagnation is detected; the trajectory is preserved. A restart would discard the current solution. Reheating is cheaper and tends to keep local structure that is still useful.

- **"What is the gap to state-of-the-art (NEH + tabu, IGA, etc.)?"**
  — Out of scope for this paper — the goal is a controlled neighborhood comparison. State-of-the-art comparison is explicit future work (slide 10).

- **"What about quantum?"**
  — Separate ongoing work (SOCO 2025, windowed QUBO formulations on D-Wave); not part of this ICAISC paper.
