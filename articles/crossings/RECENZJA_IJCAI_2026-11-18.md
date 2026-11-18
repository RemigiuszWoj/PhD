# IJCAI 2027 review: "Should Swaps Cross?" (judged as if campaign results were written in)

**Summary.** The paper orders compound-swap neighborhoods for PFSP/Cmax by allowed overlap (Dynasearch, Motzkin N0, N1, P1, involutions), counts them (J-fraction, a Poisson-type lemma, P1 growth 3.4626^n), gives exact additive selection (interval DP, two-arc-core DP, max-weight matching), and runs ILS on 120 Taillard instances (10 s, 5 seeds). Going further along the axis makes results worse: involutions lose to N0 in 12/12 classes, Dynasearch beats N0 in 7/9 classes. The paper blames the additive surrogate. A one-evaluation "safeguard" brings every family back to Dynasearch level.

**Strengths.** The question is clean and the factor is well isolated: one shared set of candidates and deltas. The tests were pre-registered (Holm, Page) and there is an equal-iterations control. The anatomy data (compound move worse than the best single swap in 11–91% of states) is honest and useful. The combinatorics is tidy. The RNA/pseudoknot analogy is apt.

**Weaknesses (most serious first).**
1. **The main study tests a known-bad acceptance rule.** Applying the additive argmin without checking its true makespan is a design flaw, not a property of the neighborhoods. Once a single O(mn) check is added (E5), all families are indistinguishable from Dynasearch, and N0-safe vs Dyn shows no significant difference in any class. So under a sensible rule the answer to the title question is "it does not matter". The headline "involutions worse in 12/12" mostly measures how badly a larger surrogate argmin misleads an unchecked tabu ILS. Reviewers will call this a strawman.
2. **Compound moves barely help at all.** Dynasearch vs plain Swap differs by 0.1–0.6 pp and is not significant for any n ≤ 100 (only n ≥ 200, where few moves are made). The whole axis sits on top of an effect that is close to zero.
3. **The proposed mechanism does not explain the effect size.** At the pair level, additivity by geometry differs only slightly (e.g. tai100_10: disjoint 0.94, nested 0.90, crossing 0.90). At the move level the gap is huge (realized/predicted 0.73 for Dyn vs 0.07 for N0, negative for involutions). The likely driver is move cardinality: more simultaneous swaps means more compounding of errors. Dynasearch captures only 10–19% of the involution predicted gain at n ≥ 100, so its moves are smaller. Swaps per move are not reported and not controlled. "Overlap breaks additivity" is therefore unproven. "Larger argmin over a biased surrogate is worse" (a winner's-curse effect) fits the data equally well.
4. **The baseline is weak and the metric hard to compare.** RPD is measured to the Taillard LB, from the identity permutation, with 10 s budgets. At n=500 a run makes 1 move, at tai200_20 it makes 6. Values such as 10–21% at m=20 are mostly LB looseness. There is no RPD to BKS, no NEH start, and no IG reference. An IJCAI reader cannot place the results relative to the state of the art.
5. **The exactness claims are undermined by caps.** Cores are capped at 400 candidates and matching at 3000. inv-safe hits the cap in 35–100% of moves at n ≥ 200, which is exactly where it falls behind. The block filter also changes the neighborhoods.
6. **Statistical overreach.** The Page trend (pooled p ≈ 1e-53) is driven by the two ends of the chain: N1 and P1 are not distinguishable from N0, and the drop happens in two steps (Dyn→N0, P1→inv). "Quality falls along the axis" overstates this. With 10 blocks, every Wilcoxon p floors at 0.002, so Holm-adjusted 0.0234 everywhere carries no information about strength. Classes with ≤ 6 moves (n ≥ 200) should not be in the confirmatory family at all. Seeds are averaged, so between-seed variance is never shown.
7. **The theory is light for IJCAI and partly unfinished.** The Poisson lemma is only sketched, with a TODO, and may well be known (Touchard/Riordan, Kasraoui–Zeng). Prop. 4 is trivial. The selection algorithms are routine extensions of interval DP and Edmonds. The c ≥ 2 algorithm is not given. The E0 cost figures are missing.
8. **Loose ends.** The QUBO section has no results. Self-citations ("Motzkin is weak at n ≥ 50 because of this rule") reveal the authors' line of work and implicitly correct papers still under review.

**Questions to authors.** (a) How do the families compare at an equal number of swaps per move, or with the cardinality capped? (b) Why not evaluate the top-k additive moves exactly, i.e. a real VLSN acceptance step? (c) Does any family beat Swap under the safeguard with a meaningful effect size? (d) Is the Poisson lemma in OEIS or the literature? (e) What are the results from an NEH start and as RPD to BKS?

**Novelty.** Exponential neighborhoods searched by DP (Congram–Potts–van de Velde; Ergun–Orlin; Grosso et al.) and by assignment or matching (Sarvanov–Doroshko, Gutin, Deineko–Woeginger, the Ahuja et al. survey) are well known. So is the observation that Dynasearch is only exact when deltas are additive (TWT). What is new is the intermediate crossing levels and the empirical confirmation that non-additivity grows with neighborhood size. That is incremental, and it is largely a negative result. Bożejko–Wodecki multimoves and the authors' own Motzkin paper cover the adjacent ground.

**Fit.** Weak for the IJCAI main track: a single problem, a single metaheuristic, and no general AI insight unless the winner's-curse/surrogate story is lifted to general VLSN. Better venues: GECCO/EvoCOP (ECOM), ICAPS (weakish fit), or Computers & OR / J. Heuristics as a combined compound-moves paper (the authors' own D11).

**Score.** 3/10 (reject); confidence 4/5. P(passing summary reject) ≈ 20–25%; P(acceptance) ≈ 3–5%.

**Authors' risk table (PLAN §11).** Too optimistic. "Compound moves give little over Swap" is listed but treated as survivable. Missing entirely: the strawman-rule objection, the gap between pair-level and move-level additivity, venue fit for IJCAI, and the effect of the cap on exactness. The "negative result with an explanation" plan B depends on the explanation, and the data do not support the geometric explanation.

## Prioritized changes before 11 Jan 2027

**Text only (days):**
1. Reframe: the subject is a biased additive surrogate under neighborhood enlargement. Present the safeguard as a core result, not an appendix, and state plainly that it removes the effect.
2. Replace "quality falls along the axis" with the two-step description. Restrict the confirmatory family to n ≤ 100. Report effect sizes with CIs.
3. Cut the QUBO section. Finish or downgrade the Poisson lemma after a literature check. Report E0 costs and the cap-hit rates per class.
4. State the Swap ≈ Dyn result in the abstract.

**New experiments (all offline or cheap except one):**
5. Swaps per move by family and realized/predicted ratio vs cardinality, from existing anatomy states (≈1 day, no campaign). This is decisive for the mechanism claim.
6. A cardinality-capped or equal-cardinality comparison (≈1–2 days of compute).
7. Top-k exact evaluation (k = 5–10) as a stronger acceptance rule (≈2 days).
8. RPD to BKS, plus an NEH start and an IG reference row for context (≈1 day).
9. Cap-sensitivity ablation at n = 100–200 (≈1 day).

Even with 5–9 done, IJCAI remains a long shot. Unless item 5 yields a general, surprising law, aim for GECCO or a C&OR merger with k-bonacci.
