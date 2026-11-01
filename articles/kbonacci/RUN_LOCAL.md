# Uruchomienie kampanii k-bonacci na Macu (instrukcja dla lokalnego Claude’a)

Gałąź: `kbonacci-icaps`. Przygotowane i przetestowane w chmurze 01.11.2026.
Zasada: **ten sam protokół co w poprzednich pracach.** ILS i SA bez zmian (stałe z `config.yaml`), start z permutacji identycznościowej,
ziarna 0–4, przebiegi **sekwencyjne**, ta sama maszyna (M4 Pro). Nowa jest tylko klasa sąsiedztw. Plan: `articles/kbonacci/PLAN.md`, sekcja A.

## 0. Przygotowanie (raz)

```bash
git fetch origin && git checkout kbonacci-icaps && git pull
source .venv/bin/activate            # albo .venv311 — to samo środowisko co w poprzednich kampaniach
pip install -r requirements.txt scipy pytest python-dotenv
python scripts/verify_taillard.py    # oczekiwane: 12/12, mismatches: 0
python -m pytest tests -q            # oczekiwane: wszystko zielone (159+ testów)
python scripts/run_kbonacci_campaign.py --dry-run
```

`--dry-run` (domyślnie **tylko ILS**) powinien pokazać ok. 22 h: validate ~1,1 h, validate_repeat ~0,3 h, main ~15,5 h, insert ~1,7 h, ablation ~3,3 h.

**Warunki pomiaru:**
- zasilacz podłączony;
- tryb niskiego zużycia energii wyłączony;
- żadnych innych ciężkich procesów;
- Mac nie może zasnąć, więc używamy `caffeinate`.

## 1. Kampania klasyczna (opcja MAKSIMUM): najpierw tylko ILS

```bash
mkdir -p results/kbonacci
caffeinate -i python scripts/run_kbonacci_campaign.py --stage all 2>&1 | tee -a results/kbonacci/campaign.log
```

- Domyślnie liczony jest **tylko ILS**. Po każdym etapie skrypt drukuje tabelę **średniej liczby ruchów** (iteracji) dla każdego pliku i k.
  Na jej podstawie decydujemy, czy liczyć SA. SA dokłada się później tą samą komendą z `--algos sa` (~22 h; gotowe przebiegi ILS są pomijane).
- Etapy idą po kolei: `validate` → `main` → `insert` → `ablation`. Po każdym automatycznie uruchamia się `extract_budgets.py`,
  który z przebiegów 10 s wyciąga budżety 0,1–5 s.
- **Wznawianie:** wystarczy uruchomić tę samą komendę. Gotowe przebiegi (`result.json`) są pomijane.
- **Postęp:** `find results/kbonacci/main -name result.json | wc -l`. Cały etap main to 10 800 przebiegów.

### Kontrola po etapie `validate` (zatrzymaj się i zgłoś, jeśli nie spełnia)

Skrypt drukuje tabelę `compare`:
- **ILS:** kolumna `|diff|` ≈ 0 (≤ 0,1 pp). ILS jest deterministyczny przy danym ziarnie.
- **SA:** `|diff|` tego samego rzędu co kolumna `noise |diff|`. SA robi reheat po stagnacji mierzonej zegarem, więc nawet dwa przebiegi
  z tym samym ziarnem i budżetem się rozjeżdżają. W chmurze wyszło 1,4–1,7 pp przy szumie 1,4–2,5 pp.

## 2. Analiza

1. **Znaleźć stare wyniki klasyczne** (Adjacent, Fibonacci, Dynasearch, Motzkin; 12 plików; 6 budżetów), np. `results/experiments/2026052*`.
   Sprawdzić kompletność. Audyt z 12.06 wskazywał braki w zagregowanych CSV i brak Motzkina dla n = 500.

   ```bash
   find results/experiments -name result.json | sed -E 's#.*/algo=([a-z]+)__neigh=([a-z_]+)__file=(tai[0-9_]+)__.*__tl=([0-9]+)ms.*#\1 \2 \3 \4#' | sort | uniq -c
   ```

   Każda kombinacja (algorytm, sąsiedztwo, plik, budżet) powinna mieć 50 przebiegów (10 instancji × 5 ziaren).

2. **Uruchomić analizę.** BKS opcjonalnie, jeśli będzie plik `data/taillard_bks.json`.

   ```bash
   python scripts/analyze_kbonacci.py \
     --dirs results/kbonacci/main_budgets results/kbonacci/insert_budgets results/kbonacci/ablation_budgets \
     --reference <katalogi starej kampanii> --out articles/kbonacci/tables
   ```

3. **Commit.** `results/` jest w `.gitignore`, więc commitujemy tylko `articles/kbonacci/tables/`, log kampanii i raport `compare`
   (np. `articles/kbonacci/validation.txt`). Potem push na `kbonacci-icaps`.

## 3. Kwanty (później, po decyzji o kredytach) — etapami MIN → OPT → MAX

Tokeny w `.env`: `DWAVE_API_TOKEN` i `IQM_TOKEN`. IQM wymaga osobnego środowiska `.venv-iqm`, bo `iqm-client` potrzebuje qiskit < 2.2
(patrz `articles/gate_qaoa/NOTES.md`).

```bash
python scripts/run_quantum_tiers.py --backend dwave --dry-run
python scripts/run_quantum_tiers.py --backend dwave --budget-s 400      # wpisać faktyczną miesięczną quotę
python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits 30 --dry-run
python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits 30
```

- Skrypt uruchamia kolejny poziom tylko wtedy, gdy pozostały budżet pokrywa jego estymatę.
- Zużycie zapisuje w `results/quantum_tiers/ledger.json`, osobno dla każdego miesiąca.
- Po wyczerpaniu budżetu kończy pracę. W następnym miesiącu ta sama komenda wznawia od miejsca przerwania.
- **D-Wave:** protokół jak w `windowed_qubo` (ILS/SA, 5 s, num_reads = 100). Ocena kosztu jest konserwatywna: 3 wywołania × 30 ms na przebieg.
- **IQM:** protokół jak w `gate_qaoa` (1 ruch z permutacji identycznościowej, ziarno 0, p = 1, 4096 shotów, okno 6).
  Koszt liczony dokładnie z liczby okien, zanim cokolwiek zostanie wysłane.
