# Uruchomienie kampanii k-bonacci na Macu (instrukcja dla lokalnego Claude’a)

Gałąź: `kbonacci-icaps`. Przygotowane i przetestowane w chmurze 01.11.2026, w kilku iteracjach przeglądu:
162 testy pod Pythonem 3.11 (czyste środowisko z `requirements.txt`) i 3.13, testy dymne wszystkich skryptów.
Plan i uzasadnienia: `articles/kbonacci/PLAN.md`, sekcje A.0 i A.

## Zasady (nie zmieniać bez decyzji użytkownika)

1. **Ten sam protokół co w poprzednich pracach.** Nie zmieniać `config.yaml` (stałe ILS/SA), `src/algorithms/*`, filtra blokowego,
   ziaren, budżetów ani instancji. Badamy sąsiedztwa, nie metaheurystyki.
2. **Najpierw tylko ILS.** SA i obliczenia kwantowe wyłącznie na wyraźne polecenie użytkownika.
3. **Przebiegi sekwencyjne, jedna maszyna.** Zasilacz podłączony, tryb niskiego zużycia energii wyłączony, wieko otwarte,
   żadnych innych ciężkich procesów.
4. **Długie komendy uruchamiać w tle:** `PYTHONUNBUFFERED=1 nohup caffeinate -i …  &`. Kampania jest wtedy niezależna od sesji Claude’a,
   a log aktualizuje się na bieżąco. Claude tylko monitoruje.
5. **Wszystko jest wznawialne.** Ta sama komenda pomija gotowe przebiegi (`result.json`). Nie kasować `results/`.
6. **Nie commitować `results/` ani `.env`.** Commitujemy tylko tabele, raporty i metadane (punkt 5).
7. **Nie aktualizować pakietów Pythona**, bo zmiana szybkości psuje porównywalność czasową. Doinstalowywać tylko brakujące.

## 0. Przygotowanie

```bash
git status --short          # jeśli są lokalne zmiany: STOP, pokazać użytkownikowi (nie stashować, nie kasować)
git fetch origin && git checkout kbonacci-icaps && git pull
source .venv311/bin/activate 2>/dev/null || source .venv/bin/activate   # środowisko poprzednich kampanii
python --version
python -m pytest tests -q
```

**Środowisko:**
- Jeśli testy padają na brakującym module, doinstalować tylko jego, np. `pip install scipy pytest python-dotenv`.
- W ostateczności `pip install -r requirements.txt`, ale **bez `--upgrade`**.
- Wymagane: `162 passed` (lub więcej), 0 failed.

```bash
python scripts/verify_taillard.py        # wymagane: files present 12/12, mismatches: 0
python scripts/run_kbonacci_campaign.py --dry-run
```

`--dry-run` (tylko ILS) ma pokazać: validate ~1,1 h, validate_repeat ~0,5 h, replicate ~0,6 h, main ~15,5 h, insert ~1,7 h, ablation ~3,3 h,
razem ~23 h.

## 1. Walidacja i kontrola replikacji (~2,2 h)

```bash
mkdir -p results/kbonacci
PYTHONUNBUFFERED=1 nohup caffeinate -i python scripts/run_kbonacci_campaign.py --stage validate,replicate \
    >> results/kbonacci/validate.log 2>&1 &
echo $! > results/kbonacci/validate.pid
```

**Monitorowanie** (np. co ~10 min: `sleep 540`, potem sprawdzenie):

```bash
tail -n 3 results/kbonacci/validate.log
for s in validate validate_repeat replicate; do echo "$s: $(find results/kbonacci/$s -name result.json 2>/dev/null | wc -l)"; done
ps -p "$(cat results/kbonacci/validate.pid)" >/dev/null && echo running || echo finished
```

Docelowo: validate 1200, validate_repeat 1000, replicate 200.

**Werdykt walidacji drukuje się sam:**
- `VALIDATION GATE: PASS` oznacza, że wartości wyciągnięte z logu przebiegu 10 s zgadzają się z niezależnymi przebiegami
  w granicach max(0,1 pp; 2 × szum powtórzeń).
- Przy `VALIDATION GATE: FAIL` skrypt sam zatrzymuje kampanię. Wtedy **STOP** i pokazać użytkownikowi tabelę.

**Kontrola replikacji** (czy Mac i Python liczą tak jak w poprzedniej kampanii):
1. Znaleźć katalog starej kampanii klasycznej (zwykle `results/experiments/2026*`) z przebiegami
   `algo=ils__neigh=fibonacci` (albo `fibonahi`) i `dynasearch` dla `tai20_5` i `tai50_10`.
2. Uruchomić porównanie:

   ```bash
   python scripts/extract_budgets.py compare --extracted results/kbonacci/replicate_budgets \
       --independent <katalog starej kampanii>
   ```

3. Werdykt PASS/FAIL w tym porównaniu **nie obowiązuje**, bo brak kolumny szumu. Ocenić dwie kolumny:
   - **iteracje (`iters extr.` vs `iters indep.`):** różnica ≤ ~10% oznacza, że szybkość maszyny jest taka sama;
   - **RPD:** różnica ≤ ~0,3 pp przy 10 s.
4. Przy większych różnicach **STOP** i raport. Mac mógł zmienić szybkość (inny Python, inne pakiety, aktualizacja systemu),
   a wtedy porównanie ze starymi wynikami przy równym czasie jest wątpliwe. Decyduje użytkownik.

Obie tabele (walidacja i replikacja) zapisać do `articles/kbonacci/validation.txt`.

## 2. Kampania główna ILS (~20,5 h). Tylko jeśli punkt 1 przeszedł.

```bash
PYTHONUNBUFFERED=1 nohup caffeinate -i python scripts/run_kbonacci_campaign.py --stage main,insert,ablation \
    >> results/kbonacci/campaign.log 2>&1 &
echo $! > results/kbonacci/campaign.pid
```

**Postęp:**

```bash
for s in main insert ablation; do echo "$s: $(find results/kbonacci/$s -name result.json 2>/dev/null | wc -l)"; done
ps -p "$(cat results/kbonacci/campaign.pid)" >/dev/null && echo running || echo finished
tail -n 3 results/kbonacci/campaign.log
```

Docelowo: main 5400, insert 600, ablation 1200. Kolejność plików: od n = 20 do n = 500.
Jeśli proces padł (np. restart Maca), uruchomić ponownie tę samą komendę. Gotowe przebiegi zostaną pominięte.

**Czego się spodziewać:**
- `kbonacci_kn` przy n = 500 trwa ok. 30–60 s na przebieg mimo limitu 10 s (jedna ocena o koszcie Dynasearch). To normalne.
- Po każdym etapie log drukuje tabelę **średniej liczby ruchów** (ILS, 10 s) dla każdego pliku i k.

**STOP i raport, jeśli:**
- w logu jest `Traceback`;
- przez > 30 min nie przybywa `result.json` (poza n = 500 i k = n);
- log zgłasza `CORRUPT`. Usunąć wskazany katalog przebiegu i wznowić; zgłosić, jeśli się powtarza.

## 3. Analiza

1. **Kompletność starych wyników** (Adjacent, Fibonacci, Dynasearch, Motzkin; 12 plików; 6 budżetów; ILS):

   ```bash
   find results/experiments -name result.json \
     | sed -E 's#.*/algo=([a-z]+)__neigh=([a-z_]+)__file=(tai[0-9_]+)__.*__tl=([0-9]+)ms.*#\1 \2 \3 \4#' \
     | sort | uniq -c | awk '$1 != 50'
   ```

   Pusty wynik oznacza komplet (każda kombinacja ma 50 przebiegów). Inaczej wypisać braki w raporcie, bez dopowiadania.
   Audyt z 12.06 wskazywał m.in. brak Motzkina dla n = 500 i nieważny stary Dynasearch dla n = 500 (błąd limitu czasu).

2. **Analiza:**

   ```bash
   python scripts/analyze_kbonacci.py \
     --dirs results/kbonacci/main_budgets results/kbonacci/insert_budgets results/kbonacci/ablation_budgets \
     --reference <katalogi starej kampanii> \
     --bks data/taillard_pfsp_bks.json --out articles/kbonacci/tables
   ```

   Wynik zawiera:
   - tabele RPD do LB (główna miara, jak dotąd), do UB-1993 z plików instancji i do BKS, każdą per n i per plik;
   - tabele liczby iteracji;
   - `wilcoxon.csv` (z korektą Holma, zbiorczo i per n);
   - `friedman.csv` (z CD Nemenyi);
   - wydruk najlepszego k dla każdego n i budżetu.

## 4. Raport dla użytkownika (po każdym etapie)

- liczba gotowych przebiegów, czas, ewentualne błędy;
- werdykt walidacji i tabela replikacji (po punkcie 1);
- tabela średniej liczby ruchów (plik × k);
- najlepsze k dla każdego n i budżetu oraz 3–5 najważniejszych obserwacji, wyłącznie z liczb;
- braki w starych wynikach.

**Nie interpretować ponad dane i nie podejmować decyzji o SA ani o kwantach.**

## 5. Commit i push

```bash
mkdir -p articles/kbonacci/campaign_logs
for s in validate validate_repeat replicate main insert ablation; do
  [ -f results/kbonacci/$s/campaign_meta.jsonl ] && cp results/kbonacci/$s/campaign_meta.jsonl articles/kbonacci/campaign_logs/${s}_meta.jsonl
done
grep -A40 "Mean number of moves" results/kbonacci/campaign.log > articles/kbonacci/campaign_logs/moves.txt
git add articles/kbonacci/tables articles/kbonacci/validation.txt articles/kbonacci/campaign_logs
git commit -m "k-bonacci: ILS campaign results (tables, validation, logs)"
git push origin kbonacci-icaps
```

Kopia surowych wyników poza repo: `tar czf ~/kbonacci_results_$(date +%Y%m%d).tgz results/kbonacci`.

## 6. Później, tylko na polecenie użytkownika

**SA (~23 h):**

```bash
PYTHONUNBUFFERED=1 nohup caffeinate -i python scripts/run_kbonacci_campaign.py \
    --stage validate,replicate,main,insert,ablation --algos sa >> results/kbonacci/campaign_sa.log 2>&1 &
```

Werdykt walidacji działa tak samo. SA nie jest deterministyczny, bo reheat zależy od zegara, więc szum powtórzeń jest większy i reguła go uwzględnia.

**Kwanty, poziomami MIN → OPT → MAX.** Tokeny w `.env` (`DWAVE_API_TOKEN`, `IQM_TOKEN`; wzór w `.env.example`). Zawsze najpierw `--dry-run`.

```bash
python scripts/run_quantum_tiers.py --backend dwave --dry-run
python scripts/run_quantum_tiers.py --backend dwave --budget-s <miesięczna quota w s> --tiers min
python scripts/extract_budgets.py extract --src results/quantum_tiers/dwave \
    --out results/quantum_tiers/dwave_budgets --budgets 100,500,1000,2000,5000
```

- **IQM** wymaga osobnego środowiska `.venv-iqm` (qiskit < 2.2, patrz `articles/gate_qaoa/NOTES.md`):

  ```bash
  source .venv-iqm/bin/activate
  python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits <dostępne kredyty> --dry-run
  python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits <dostępne kredyty> --tiers min
  ```

- **Budżet:** skrypt zapisuje zużycie w `results/quantum_tiers/ledger.json` i nie startuje poziomu, na który nie starcza.
  Kolejny poziom (`opt`, potem `max`) tylko na polecenie użytkownika.
- **D-Wave, poziom MAX (n = 200 i 500):** obserwować pierwsze przebiegi. Jeśli jedno wywołanie (embedding) trwa > 60 s
  albo pojawia się `FAILED (QPU embedding)`, przerwać i zgłosić.
