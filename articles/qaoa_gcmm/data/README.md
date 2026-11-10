# Data behind the GCMM 2026 paper (Paper 59)

One folder per run; the folder name gives the algorithm, neighborhood,
Taillard file and instance, time limit and seed. Each `result.json` holds the
start and final makespan, the per-move trace and the run settings.

- `qaoa_gate_hw_n20/`: the 80 runs on IBM `ibm_fez` (Table 2).
- `qaoa_gate_n20/`: the same protocol on the noiseless simulator.

The calibrated QAOA angles are in `data/qaoa_angles*.json` at the repository root.
