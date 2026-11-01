# Uruchomienie kampanii k-bonacci na Macu (instrukcja dla lokalnego Claude’a)

Gałąź: `kbonacci-icaps`. Przygotowane w chmurze 01.11.2026 i przetestowane: 159 testów pod Pythonem 3.11 i 3.13, testy dymne wszystkich skryptów.
Plan i uzasadnienia: `articles/kbonacci/PLAN.md`, sekcje A.0 i A.

## Zasady (nie zmieniać bez decyzji użytkownika)

1. **Ten sam protokół co w poprzednich pracach.** Nie zmieniać `config.yaml` (stałe ILS/SA), `src/algorithms/ils.py`, `src/algorithms/sa.py`,
   filtra blokowego, ziaren, budżetów ani instancji. Badamy sąsiedztwa, nie metaheurystyki.
2. **Najpierw tylko ILS.** SA dopiero po decyzji użytkownika (`--algos sa`).
3. **Przebiegi sekwencyjne, jedna maszyna.** Zasilacz podłączony, tryb niskiego zużycia energii wyłączony, wieko otwarte,
   żadnych innych ciężkich procesów (kompilacje, przeglądarka z wideo, inne eksperymenty).
4. **Długie komendy uruchamiać w tle przez `nohup`**, niezależnie od sesji Claude’a i limitów czasu narzędzia. Claude tylko monitoruje logi.
5. **Wszystko jest wznawialne:** ta sama komenda pomija gotowe przebiegi (`result.json`). Nie kasować `results/`.
6. **Nie commitować `results/`** (jest w `.gitignore`, duże) ani `.env`. Commitujemy tylko tabele, raporty i metadane (punkt 4).

## 0. Przygotowanie

```bash
git fetch origin && git checkout kbonacci-icaps && git pull
source .venv311/bin/activate 2>/dev/null || source .venv/bin/activate
pip install -r requirements.txt scipy pytest python-dotenv
python scripts/verify_taillard.py            # wymagane: files present 12/12, mismatches: 0
python -m pytest tests -q                    # wymagane: 159 passed (lub więcej), 0 failed
python scripts/run_kbonacci_campaign.py --dry-run
```

`--dry-run` (tylko ILS) powinien pokazać: validate ~1,1 h, validate_repeat ~0,3 h, main ~15,5 h, insert ~1,7 h, ablation ~3,3 h, razem ~22 h.

Jeśli `verify_taillard.py` zgłosi brak plików: instancje są w `data/` z poprzednich kampanii. Bez nich nic nie uruchamiać, tylko zgłosić.

## 1. Walidacja (pierwsza, ~1,4 h)

```bash
mkdir -p results/kbonacci
nohup caffeinate -i python scripts/run_kbonacci_campaign.py --stage validate \
    >> results/kbonacci/validate.log 2>&1 &
echo $! > results/kbonacci/validate.pid
```

Monitorowanie:

```bash
tail -n 5 results/kbonacci/validate.log
find results/kbonacci/validate -name result.json | wc -l    # docelowo 1200
```

Po zakończeniu log zawiera tabelę `compare` (budżety wyciągnięte z logu vs niezależne przebiegi).

**Warunek przejścia:** dla ILS kolumna `|diff|` ≤ 0,1 pp we wszystkich wierszach (ILS jest deterministyczny przy danym ziarnie).
Jeśli nie jest spełniony: **STOP**, nie uruchamiać dalszych etapów, zgłosić tabelę użytkownikowi.

Tabelę `compare` i tabelę liczby ruchów zapisać do `articles/kbonacci/validation.txt`.

## 2. Kampania główna ILS (~20,5 h)

```bash
nohup caffeinate -i python scripts/run_kbonacci_campaign.py --stage main,insert,ablation \
    >> results/kbonacci/campaign.log 2>&1 &
echo $! > results/kbonacci/campaign.pid
```

**Postęp:**

```bash
for s in main insert ablation; do echo "$s: $(find results/kbonacci/$s -name result.json 2>/dev/null | wc -l)"; done
```

Docelowo: main 5400, insert 600, ablation 1200. Kolejność plików: od n = 20 do n = 500.

**Czy proces żyje:** `ps -p $(cat results/kbonacci/campaign.pid)`. Jeśli padł (restart Maca itp.), uruchomić ponownie tę samą komendę; wznowi pracę.

**Czego się spodziewać:**
- `kbonacci_kn` przy n = 500 trwa ok. 30–60 s na przebieg (jedna ocena Dynasearch-owa), mimo limitu 10 s. To normalne, tak samo jak stary Dynasearch.
- Po każdym etapie log drukuje tabelę **średniej liczby ruchów** (ILS, 10 s) dla każdego pliku i k. Skopiować ją do raportu.

**STOP i zgłoszenie, jeśli:**
- w logu pojawi się `Traceback`;
- przez > 30 min nie przybywa `result.json` (poza n = 500 i k = n);
- RPD w raporcie wyjdzie ujemne (dolne ograniczenie nie może być przebite).

## 3. Analiza

1. **Znaleźć stare wyniki klasyczne** (Adjacent, Fibonacci, Dynasearch, Motzkin; 12 plików; 6 budżetów; ILS i SA), zwykle `results/experiments/2026*`.
   Sprawdzić kompletność:

   ```bash
   find results/experiments -name result.json \
     | sed -E 's#.*/algo=([a-z]+)__neigh=([a-z_]+)__file=(tai[0-9_]+)__.*__tl=([0-9]+)ms.*#\1 \2 \3 \4#' \
     | sort | uniq -c | awk '$1 != 50'
   ```

   Pusty wynik oznacza, że wszystko jest kompletne (każda kombinacja ma 50 przebiegów). Inaczej wypisać braki w raporcie, bez dopowiadania.
   Audyt z 12.06 wskazywał m.in. brak Motzkina dla n = 500 i nieważny Dynasearch dla n = 500 (błąd limitu czasu).

2. **Analiza.** BKS dodać, jeśli użytkownik dostarczy `data/taillard_bks.json`.

   ```bash
   python scripts/analyze_kbonacci.py \
     --dirs results/kbonacci/main_budgets results/kbonacci/insert_budgets results/kbonacci/ablation_budgets \
     --reference <katalogi starej kampanii> --out articles/kbonacci/tables
   ```

## 4. Commit i push

```bash
mkdir -p articles/kbonacci/campaign_logs
cp results/kbonacci/*/campaign_meta.jsonl articles/kbonacci/campaign_logs/ 2>/dev/null   # gdy nazwy się powtarzają: dodać prefiks etapu
grep -A40 "Mean number of moves" results/kbonacci/campaign.log > articles/kbonacci/campaign_logs/moves.txt
git add articles/kbonacci/tables articles/kbonacci/validation.txt articles/kbonacci/campaign_logs
git commit -m "k-bonacci: ILS campaign results (tables, validation, logs)"
git push origin kbonacci-icaps
```

Kopia surowych wyników poza repo: `tar czf ~/kbonacci_results_$(date +%Y%m%d).tgz results/kbonacci`.

## 5. Raport dla użytkownika (po każdym etapie)

- liczba gotowych przebiegów, czas, ewentualne błędy;
- tabela `compare` (po walidacji);
- tabela średniej liczby ruchów (plik × k);
- najlepsze k dla każdego n i budżetu (koniec wydruku `analyze_kbonacci.py`) oraz 3–5 najważniejszych obserwacji;
- braki w starych wynikach.

**Nie interpretować ponad dane i nie podejmować decyzji o SA ani o kwantach.** To decyzje użytkownika.

## 6. Później, na polecenie użytkownika

**SA (~22 h):**

```bash
nohup caffeinate -i python scripts/run_kbonacci_campaign.py --stage validate,main,insert,ablation --algos sa \
    >> results/kbonacci/campaign_sa.log 2>&1 &
```

W walidacji SA `|diff|` ma być tego samego rzędu co kolumna `noise |diff|`. SA nie jest deterministyczny: reheat zależy od zegara.

**Kwanty, poziomami MIN → OPT → MAX.** Tokeny w `.env` (`DWAVE_API_TOKEN`, `IQM_TOKEN`; wzór w `.env.example`). Zawsze najpierw `--dry-run`.

```bash
python scripts/run_quantum_tiers.py --backend dwave --dry-run
python scripts/run_quantum_tiers.py --backend dwave --budget-s <miesięczna quota w s> --tiers min
python scripts/extract_budgets.py extract --src results/quantum_tiers/dwave \
    --out results/quantum_tiers/dwave_budgets --budgets 100,500,1000,2000,5000
```

- **IQM** wymaga osobnego środowiska `.venv-iqm`, bo `iqm-client` potrzebuje qiskit < 2.2 (patrz `articles/gate_qaoa/NOTES.md`):

  ```bash
  source .venv-iqm/bin/activate
  python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits <dostępne kredyty> --dry-run
  python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits <dostępne kredyty> --tiers min
  ```

- **Skrypt sam pilnuje budżetu** (`results/quantum_tiers/ledger.json`) i zatrzymuje się przed poziomem, na który nie starcza.
  Kolejny poziom (`--tiers opt`, potem `max`) uruchamiać dopiero na polecenie użytkownika.
- **D-Wave, poziom MAX (n = 200 i 500):** pierwsze przebiegi obserwować. Jeśli jedno wywołanie QPU (embedding) trwa > 60 s albo pojawiają się
  `FAILED (QPU embedding)`, przerwać (`kill $(cat ...pid)` albo Ctrl+C) i zgłosić. Zużyty czas jest w księdze.
