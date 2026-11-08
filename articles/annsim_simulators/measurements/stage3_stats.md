# Stage 3: ILS on simulators vs Sirius, n = 50…500 (S2, p = 1, snapshot 20261023)

153 Sirius runs × 2 engines = 306 ILS runs, all ok; every Sirius run made 1 move.

| engine | mean PRD sim | mean PRD Sirius | mean Δ (pp) | mean abs Δ | better/tie/worse | Wilcoxon p |
|---|---|---|---|---|---|---|
| qsim | 21.12 | 20.68 | +0.44 | 0.60 | 34/43/76 | 6.4e-08 |
| qulacs_loop | 21.05 | 20.68 | +0.37 | 0.59 | 35/45/73 | 1.1e-06 |

Mean Δ by n — qsim: 50 +0.79, 100 +0.28, 200 +0.23, 500 +0.04; qulacs: 50 +0.58, 100 +0.26, 200 +0.31, 500 +0.06.
Mean Δ by neighborhood — qsim: Adjacent −0.01, Fibonacci +0.04, Motzkin +0.69, Dynasearch +0.93;
qulacs: Adjacent −0.01, Fibonacci +0.09, Motzkin +0.36, Dynasearch +0.96.
