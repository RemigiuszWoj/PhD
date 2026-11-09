# T5: ILS with a simulator in place of Sirius (tai20, inst. 0-3, p = 1)

PRD in percentage points (pp); difference = simulator - Sirius, paired by run (each simulator run
replays one Sirius run with its move budget). Wilcoxon signed-rank (ties dropped), Holm over the
four cells. Bootstrap 95 % CI of the mean (20000 resamples) over runs and over the 12 instances.

| engine | sc | n | PRD sim | PRD Sirius | mean diff | median | CI runs | CI instances | better / tie / worse | p | p Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qsim | S2 | 72 | 32.48 | 30.70 | +1.78 | +0.43 | [+0.91, +2.71] | [+0.28, +3.77] | 18 / 15 / 39 | 0.00025 | 0.00098 |
| qsim | S3 | 72 | 30.83 | 30.70 | +0.13 | +0.00 | [-0.66, +0.94] | [-1.21, +1.70] | 26 / 17 / 29 | 0.89 | 0.94 |
| qulacs_loop | S2 | 72 | 31.98 | 30.70 | +1.28 | +0.00 | [+0.37, +2.24] | [-0.29, +3.41] | 23 / 18 / 31 | 0.041 | 0.12 |
| qulacs_loop | S3 | 72 | 31.13 | 30.70 | +0.43 | +0.00 | [-0.33, +1.22] | [-0.66, +1.83] | 28 / 16 / 28 | 0.47 | 0.94 |

## qsim - Qulacs (shot loop), same runs

| sc | n | mean diff | CI runs | CI instances | ties | p |
|---|---|---|---|---|---|---|
| S2 | 72 | +0.50 | [-0.02, +1.04] | [-0.04, +1.04] | 21 | 0.092 |
| S3 | 72 | -0.30 | [-0.77, +0.15] | [-0.85, +0.19] | 24 | 0.26 |

## Floor: Sirius against itself

Repeated configurations: tai20_5/0/dynasearch, tai20_5/0/motzkin, tai20_10/0/dynasearch, tai20_10/0/motzkin, tai20_20/0/dynasearch, tai20_20/0/motzkin (30 Sirius runs, 1 move each).

- Sirius vs Sirius, two runs of one configuration: mean |diff| = 2.19 pp, 95th percentile 8.60 pp; pooled within-configuration SD = 2.57 pp (per configuration: 2.51, 2.22, 4.07, 3.42, 0.00, 0.00).

Per configuration (same order as above): mean of the 5 simulator runs minus mean of the 5 Sirius runs,
with the Mann-Whitney p of the two sets of 5 (smallest attainable two-sided p = 0.008).

| engine | sc | mean abs(sim - Sirius), same run | pooled SD of the simulator | config. mean diff (p) |
|---|---|---|---|---|
| qsim | S2 | 4.45 | 1.92 | +2.47 (0.11), +0.44 (1), +10.47 (0.02), +6.73 (0.021), +1.29 (0.12), -2.40 (0.0073) |
| qsim | S3 | 3.80 | 2.17 | -1.06 (1), -0.44 (0.83), +5.33 (0.091), +4.13 (0.14), +0.06 (1), -5.84 (0.0075) |
| qulacs_loop | S2 | 4.74 | 2.34 | +0.83 (1), -2.31 (0.14), +11.78 (0.011), +4.90 (0.059), +1.16 (0.12), -3.23 (0.025) |
| qulacs_loop | S3 | 3.49 | 1.85 | -1.31 (0.29), -0.80 (0.67), +6.55 (0.034), +3.69 (0.14), +0.85 (0.66), -2.42 (0.025) |
