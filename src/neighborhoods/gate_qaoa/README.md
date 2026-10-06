# gate_qaoa — gate-model QAOA neighborhood family

Bramkowy odpowiednik rodzin `quantum_qubo` / `quantum_qubo_enhanced`.
Ta sama macierz QUBO co na D-Wave (budowana przez `common_qubo`), ale
rozwiązywana obwodem **fixed-angle QAOA** zamiast annealera. Metryka wyjściowa
(RPD względem dolnej granicy Taillarda) jest identyczna, więc wyniki są
bezpośrednio porównywalne z poprzednimi pracami.

Cel końcowy: puścić instancje Taillarda na realnym QPU (`ibm_fez`) i porównać
RPD do wyników annealingowych/klasycznych.

---

## Architektura (warstwy)

```
runner.py → ils/sa → base.get_neighbor("gate_*")          # warstwa 3: wpięcie
   → gate_qaoa/{adjacent,fibonacci,dynasearch,motzkin,windowed}   # warstwa 2: 4 otoczenia
        ├─ common_qubo/  (candidates, assemble, mapback)   # warstwa 0: która Q (wspólna z annealingiem)
        └─ gate_qaoa/    (circuit, angles, solve)          # warstwa 1: jak rozwiązać Q obwodem
                          └─ data/qaoa_angles.json ← experiments/qaoa_calibrate_angles.py  # warstwa 4: kalibracja
```

- **common_qubo** — *która Q* (jedno źródło; annealing i gate dostają identyczną macierz).
- **circuit / angles / solve** — *jak rozwiązać Q obwodem* (kalibracja offline + 1 wywołanie/ruch online).
- **4 otoczenia** — *ruch w permutacji* (mapback do `(new_pi, cmax, moves)`).
- **base/runner** — *pętla + RPD* (jak w poprzednich pracach).

---

## Konfiguracja (`config.yaml`)

```yaml
quantum:
  qaoa_backend: ibm           # ibm | aer_noisy  (brak backendu symulatorowego)
  qaoa_p: 1                   # głębokość QAOA
  qaoa_window_size: 6         # gate_dynasearch/motzkin: rozmiar okna (pełne n); None = pojedyncza Q
  qaoa_overlap_ratio: 0.5     # nakładanie okien
  qaoa_shots: 4096            # tylko aer_noisy / ibm
  L_max_dynasearch: null      # cap długości interwału (ścieżka bez okien)
  L_max_motzkin: null
experiment:
  neighborhoods: [gate_fibonacci, gate_dynasearch, gate_motzkin, gate_adjacent]
```

`ibm` czyta `IBM_TOKEN` / `IBM_CRN` ze środowiska (`.env`); domyślny backend `ibm_fez`.

---

## Przepływ A — kalibracja (offline, RAZ → `data/qaoa_angles.json`)

```
python -m src.experiments.qaoa_calibrate_angles --p-max 5
  _load_instances()                → parser(data/tai*.txt) → [(n, pt), ...]
  dla nb in (adjacent,fibonacci,dynasearch,motzkin):
    build_training(nb, ...)        → okna z Taillarda → List[Q]   (małe QUBO)
      _window_qubo():
        adjacent/fibonacci: compute_deltas → assemble_onehot/tridiagonal_qubo
        dyna/motzkin:       enumerate_interval_candidates(window) → assemble_pairwise_qubo(conflict_fn)
    optimize_neighborhood(training, p_max)          [gate_qaoa/angles.py]
      _prepare():  Q → qubo_to_ising → normalize_ising → ising_hamiltonian → (K,h,J,H_op,l1)
      p=1:  _grid_p1        → min ⟨H_C⟩ na siatce (γ,β)
      p≥2:  _optimize_p     → multi-start Nelder-Mead (seed: warstwa zerowa gwarantuje monotoniczność + INTERP)
  zapis → data/qaoa_angles.json    {neighborhood: {p: {gamma, beta, objective}}}
```

Cel = średnie znormalizowane `⟨H_C⟩` liczone dokładnie na statevectorze (bez
brute-force). Kalibracja na małych oknach, stosowana do pełnych QUBO w runtime —
opiera się na przenaszalności kątów (normalizacja `⟨H⟩` zdejmuje skalę kary P).

---

## Przepływ B — bieg eksperymentu (online, RPD)

```
ExperimentRunner(quantum_config=cfg["quantum"]).run([RunConfig(
    algorithm="ils", neighborhood="gate_fibonacci",
    instance_file="data/tai20_5.txt", instance_number=0, seed=0, time_limit_ms=60000)])

run() → _run_single(cfg):
  parser(...)                       → processing_times, lower_bound (= C* do RPD)
  _run_ils → iterated_local_search(..., neigh_mode="gate_fibonacci", quantum_config=...)

  PĘTLA ILS (aż do time_limit_ms), każda iteracja = JEDEN ruch = JEDNO wywołanie QAOA:
    get_neighbor("gate_fibonacci", current_pi, pt, n, None, quantum_config)   [base.py]
      qp = _extract_quantum_params(...)   → {p, backend, shots, [window_size, L_max]}
      gate_fibonacci_neighborhood(pi, pt, p, backend, shots):
        candidates = enumerate_adjacent_candidates(pi, pt)      [common_qubo]  → [(pos, δ)]
        Q          = assemble_tridiagonal_qubo(candidates)      [common_qubo]  → dict Q
        solution   = solve_qaoa(Q, "fibonacci", p, backend, angles=None, shots)
          _load_angles("fibonacci", p)                          ← data/qaoa_angles.json
          variables,h,J,_ = qubo_to_ising(Q)                    [circuit]  x=(1−Z)/2 → bit=x
          hn,Jn,_         = normalize_ising(h, J)
          ibm: qc=build_qaoa_circuit(...,measure=True); transpile→ibm_fez;
               SamplerV2 (wszystkie okna ruchu w JEDNYM zadaniu) → próbka o min. energii QUBO
          return bitstring_to_assignment(top, variables)        → {"x_k":0/1}
        valid = validate_no_overlap(selected_positions(solution, positions))  (+fallback)
        return apply_swaps(pi, valid), c_max(...), valid
    → ILS: tabu/akceptacja/update best/mushroom kick → iteruj

  RunResult(cmax_best, lower_bound, ...)
    gap_percent() = (cmax_best − lower_bound)/lower_bound × 100      ← RPD
  _persist_result → results/experiments/<ts>/algo=ils__neigh=gate_fibonacci__.../result.json
```

Eksperymenty biegną **wyłącznie na sprzęcie** (`ibm_fez`). Backend symulatorowy
został usunięty z `solve.py`; statevector występuje już tylko w `angles.py`,
przy jednorazowej kalibracji kątów offline.

gate_dynasearch/motzkin z `window_size`: zamiast pojedynczej Q →
`windowed_interval_swaps()` — pętla po nakładających się oknach, per okno
`enumerate_interval_candidates(window) → assemble_pairwise_qubo → solve_qaoa`,
scalenie swapów (analog `quantum_qubo_enhanced`). Konieczne dla pełnych
Taillardów, gdzie pojedyncza Q ma K = O(n²).

---

## Pliki

| plik | rola |
|---|---|
| `circuit.py` | QUBO→Ising, obwód QAOA dowolnego p, mapowanie bitu |
| `angles.py` | offline dobór kątów (⟨H_C⟩, p=1 siatka, p≥2 NM+INTERP+monotoniczność) |
| `solve.py` | `solve_qaoa` / `solve_qaoa_batch` → `{"x_k":0/1}` (kontrakt jak `solve_qubo`); wsad = jedno zadanie |
| `adjacent.py` `fibonacci.py` | pojedyncza Q (K=n−1) |
| `dynasearch.py` `motzkin.py` | pojedyncza Q (L_max) lub okienkowa (window_size) |
| `windowed.py` | dekompozycja okienkowa dla otoczeń interwałowych |
| `../common_qubo/` | wspólna budowa Q + mapback (annealing + gate) |
| `../../experiments/qaoa_calibrate_angles.py` | skrypt kalibracji → `data/qaoa_angles.json` |
| `../../../tests/test_common_qubo_golden.py` | golden test: annealing bez zmian po refaktorze (24/24) |

---

## Status

Zrobione: silnik, 4 otoczenia (z okienkowaniem), wpięcie w `base.py`/`runner.py`,
skrypt kalibracji, golden test, batching okien w jedno zadanie. Zweryfikowane na `ibm_fez`.

Nie zrobione: pełna kalibracja (mamy tylko szybką), przebiegi RPD na symulatorze
i na `ibm_fez`, porównanie do poprzednich artykułów. Ścieżka `ibm` w `solve.py`
zaimplementowana, ale nieuruchamiana (budżet QPU).

Uwaga metodologiczna: jeden ruch na `ibm_fez` trwa ~30 s zegarowych (kolejka +
wykonanie), więc przy każdym budżecie z zakresu 100–10000 ms wykonuje się
dokładnie **jeden ruch**. Wszystkie kolumny czasowe mają wtedy tę samą wartość —
tak samo jak wiersze D-Wave w artykule windowed, które też były jednoiteracyjne.
