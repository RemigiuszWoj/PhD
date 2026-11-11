# Uruchomienie kampanii job shop na Macu (instrukcja dla lokalnego Claude’a)

Gałąź: `jobshop-plan-v2`, worktree `~/Desktop/PhD-jobshop`. Kod przygotowany w chmurze 01.11.2026, przejrzany lokalnie 11.11.2026.
Plan i uzasadnienia: `articles/jobshop/PLAN.md`, **sekcja A.7** (stan, usterki, harmonogram). Wzór: `articles/kbonacci/RUN_LOCAL.md`.

## Zasady (nie zmieniać bez decyzji użytkownika)

1. **Ten sam protokół co w pracach o PFSP.** Nie zmieniać `config.yaml`, kodu PFSP (`src/algorithms/`, `src/neighborhoods/`) ani stałych
   w `src/jobshop/runner.py` (`SA_DEFAULTS`, `ILS_DEFAULTS`), ziaren, budżetów i list instancji. Badamy sąsiedztwa, nie metaheurystyki.
2. **Najpierw tylko ILS.** SA dopiero po decyzji użytkownika (`--algos sa`).
3. **Kampania k-bonacciego ma pierwszeństwo.** Nie startować, dopóki na Macu działa albo czeka inna kampania z budżetem czasowym.
   Przed startem: `ps aux | grep -E "run_.*campaign|run_quantum" | grep -v grep` ma być puste. Cudzych procesów nie zabijać.
4. **Przebiegi sekwencyjne, jedna maszyna.** Zasilacz podłączony, tryb niskiego zużycia energii wyłączony, wieko otwarte,
   żadnych innych ciężkich procesów. W trakcie kampanii nie uruchamiać testów ani kompilacji.
5. **Długie komendy w tle przez `nohup caffeinate -i`.** Claude tylko monitoruje logi.
6. **Wszystko jest wznawialne:** ta sama komenda pomija gotowe przebiegi (`result.json`). Nie kasować `results/`.
7. **Nie commitować `results/` ani `data/`.** Commitujemy tabele, walidację i metadane (punkt 5).

## 0. Przygotowanie

Worktree nie ma własnego środowiska ani danych. Środowisko bierzemy z głównego katalogu, dane PFSP (dla etapu bridge) dowiązujemy.

```bash
cd ~/Desktop/PhD-jobshop && git pull
source ../PhD/.venv311/bin/activate
python scripts/fetch_jsplib.py                       # 162 instancje do data/JSPLIB/instances
mkdir -p data && for f in tai20_5 tai20_10 tai50_10; do ln -sf ~/Desktop/PhD/data/$f.txt data/$f.txt; done
python scripts/verify_taillard_jssp.py               # wymagane: present 80/80, mismatches: 0
python -m pytest tests/test_jobshop.py -q            # wymagane: 25 passed
python scripts/run_jobshop_campaign.py --dry-run --stage all
```

**Blokada znana z 11.11:** w repo nie ma `src/jobshop/data/taillard_jssp.json` (ziarna i BKS; pominęła go reguła `data/` w `.gitignore`).
Bez tego pliku `verify_taillard_jssp.py` kończy się błędem, a test `test_ta01_generator_from_stored_seeds` nie przechodzi (24/25).
Wtedy **nic nie uruchamiać** i zgłosić użytkownikowi. Nie odtwarzać BKS z pamięci.

`--dry-run --stage all` (tylko ILS) powinien pokazać w przybliżeniu:

| Etap | Przebiegi | Czas |
|---|---|---|
| validate + validate_repeat | 1200 + 400 | ~1,1 h + ~0,4 h |
| min | 2400 | ~6,8 h |
| opt | 2000 | ~5,6 h |
| continuum | 3000 | ~4,3 h |
| bridge | 450 | ~1,3 h |
| razem | | ~19,4 h |

Jeśli liczby przebiegów się różnią, zgłosić przed startem.

## 1. Walidacja (pierwsza, ~1,5 h)

```bash
mkdir -p results/jobshop
nohup caffeinate -i python scripts/run_jobshop_campaign.py --stage validate \
    >> results/jobshop/validate.log 2>&1 &
echo $! > results/jobshop/validate.pid
```

Monitorowanie:

```bash
tail -n 5 results/jobshop/validate.log
find results/jobshop/validate results/jobshop/validate_repeat -name result.json | wc -l    # docelowo 1600
```

Po zakończeniu log zawiera tabelę `compare` (budżety wyciągnięte z dziennika ruchów vs niezależne przebiegi) i tabelę średniej liczby ruchów.
Obie zapisać do `articles/jobshop/validation.txt`.

**Warunki przejścia (bramka G1 z planu):**
- kolumna `|diff|` ≤ 0,1 pp we wszystkich wierszach ILS (ILS jest deterministyczny przy danym ziarnie);
- żadnego `Traceback` i żadnego przebiegu z `cmax_best` równym `null`.

Jeśli któryś warunek nie jest spełniony: **STOP**, nie uruchamiać dalszych etapów, zgłosić tabelę.

Plan wymaga w G1 jeszcze dwóch pomiarów (korelacja estymaty z dokładną deltą dla d ≥ 2, udział testu wykonalności w czasie iteracji).
Kampania ich nie liczy. Jeśli użytkownik o nie poprosi, policzyć osobnym skryptem **po** walidacji, nie w jej trakcie.

## 2. Kampania główna ILS (~18 h)

```bash
nohup caffeinate -i python scripts/run_jobshop_campaign.py --stage min,opt,continuum,bridge \
    >> results/jobshop/campaign.log 2>&1 &
echo $! > results/jobshop/campaign.pid
```

**Postęp:**

```bash
for s in min opt continuum bridge; do echo "$s: $(find results/jobshop/$s -name result.json 2>/dev/null | wc -l)"; done
tail -n 3 results/jobshop/campaign.log      # wiersz co 25 przebiegów, z ETA
```

Docelowo: min 2400, opt 2000, continuum 3000, bridge 450. Kolejność klas: od 15×15 do 100×20.

**Czy proces żyje:** `ps -p $(cat results/jobshop/campaign.pid)`. Jeśli padł (restart Maca itp.), uruchomić ponownie tę samą komendę.

**Czego się spodziewać:**
- przy 100×20 jedna ocena trwa kilka milisekund, więc liczba ruchów w 10 s jest o rząd mniejsza niż przy 15×15. To normalne;
- etap bridge zgłosi `SKIPPED`, jeśli brakuje `data/tai20_5.txt` (dowiązania z punktu 0);
- po każdym etapie log drukuje tabelę **średniej liczby ruchów** (klasa × wariant). Skopiować ją do raportu.

**STOP i zgłoszenie, jeśli:**
- w logu pojawi się `Traceback`;
- przez > 30 min nie przybywa `result.json`;
- RPD do BKS wyjdzie ujemne (wynik lepszy od najlepszego znanego oznacza błąd danych albo oceny, nie rekord);
- w etapie continuum przy ρ = 0 pojawi się niewykonalny kandydat (przeczy wnioskowi (d) z twierdzenia o wykonalności).

## 3. Analiza

```bash
python scripts/analyze_jobshop.py --dirs results/jobshop/min_budgets results/jobshop/opt_budgets \
    --out articles/jobshop/tables
python scripts/analyze_jobshop.py --dirs results/jobshop/continuum_budgets results/jobshop/bridge_budgets \
    --out articles/jobshop/tables
```

Skrypt wypisuje: RPD do BKS (klasa × wariant × budżet), liczbę ruchów, test Friedmana z różnicą krytyczną Nemenyiego,
pary Wilcoxona z poprawką Holma, tabelę kontinuum (rozmiar × ρ × wariant) i tabelę mostu do PFSP.

Uwagi do interpretacji (z A.7 planu), które mają trafić do raportu, a nie być rozstrzygane samodzielnie:
- przy budżecie 0,1 s w klasach 50×… i 100×20 wykonuje się kilkanaście ruchów; te komórki opisują start, nie sąsiedztwo;
- w kontinuum porównać warianty także przy równej liczbie ruchów (N5 robi ich przy ρ = 0 kilka razy więcej);
- punkty ρ = 1 dla 20×5 i 50×10 są zdegenerowane (wszystkie warianty trafiają ograniczenie maszynowe).

## 4. Opcje MAX (tylko na polecenie użytkownika)

```bash
python scripts/run_jobshop_campaign.py --dry-run --stage ablation,classic
nohup caffeinate -i python scripts/run_jobshop_campaign.py --stage ablation,classic \
    >> results/jobshop/campaign_max.log 2>&1 &
```

Ablacja (~1,1 h): dokładne delty i remis na ścieżce rozstrzygany poprzednikiem technologicznym, klasy 15×15 i 30×15.
Zestawy klasyczne (~4,6 h): ft, la, orb, abz, swv, yn. Przeszukiwanie tabu (TS-N5, TS-c6) nie jest jeszcze zaimplementowane.

## 5. Commit i push

```bash
mkdir -p articles/jobshop/campaign_logs
for s in validate min opt continuum bridge; do
  cp results/jobshop/$s/campaign_meta.jsonl articles/jobshop/campaign_logs/${s}_meta.jsonl 2>/dev/null
done
grep -A12 "Mean number of moves" results/jobshop/campaign.log > articles/jobshop/campaign_logs/moves.txt
git add articles/jobshop/tables articles/jobshop/validation.txt articles/jobshop/campaign_logs
git commit -m "jobshop: ILS campaign results (tables, validation, logs)"
git push origin jobshop-plan-v2
```

Kopia surowych wyników poza repo: `tar czf ~/jobshop_results_$(date +%Y%m%d).tgz results/jobshop`.

## 6. Raport dla użytkownika (po każdym etapie)

- liczba gotowych przebiegów, czas, ewentualne błędy;
- tabela `compare` (po walidacji);
- tabela średniej liczby ruchów (klasa × wariant);
- tabela RPD dla 10 s i 3–5 najważniejszych obserwacji, w tym wynik par c6 vs N6, c7 vs N7 i k = 3 vs N5;
- tabela kontinuum z zaznaczonymi punktami zdegenerowanymi.

**Nie interpretować ponad dane i nie podejmować decyzji o SA, opcjach MAX ani o kwantach.** To decyzje użytkownika.
Liczb z kampanii nie wpisywać do `main.tex` ani `wersja_pl.tex` bez pokazania zmian w czacie.

## 7. Później, na polecenie użytkownika

**SA (~14 h: walidacja 1,5 h, min 6,8 h, opt 5,6 h; kontinuum i most to kolejne ~5,6 h, po dopisaniu ich do `--stage`):**

```bash
nohup caffeinate -i python scripts/run_jobshop_campaign.py --stage validate,min,opt --algos sa \
    >> results/jobshop/campaign_sa.log 2>&1 &
```

W walidacji SA `|diff|` ma być tego samego rzędu co kolumna `noise |diff|`. SA nie jest deterministyczny: reheat zależy od zegara.

**Kwanty JSSP:** nie wcześniej niż w I kwartale 2027, po kwantach k-bonacciego (plan, A.5 i A.7). Skryptów jeszcze nie ma.
