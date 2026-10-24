# ICAISC 2027 — studium przypadku głębokości QAOA p: plan badań

Stan: 23.10.2026. Gałąź `icaisc-p-study`. Termin zgłoszenia **15.11.2026 23:59 AoE** (= pn. 16.11 rano PL),
cel: wysłać **pt. 13.11**. Recenzja dwustronnie ślepa, ~12 stron LNCS/LNAI.

## Pytanie badawcze

Czy w sąsiedztwach QUBO dla PFSP warto zwiększać liczbę warstw QAOA p, skoro odczyt jest energetyczny
(najlepsza próbka z 4096 shotów)? Rozbijamy to na trzy pytania szczegółowe:

- **Q1 (bez szumu):** jak rośnie jakość rozkładu QAOA z p = 1…10 i gdzie się nasyca, osobno dla sąsiedztw
  rzadkich (Adjacent, Fibonacci: QUBO trójdiagonalne) i gęstych (Dynasearch, Motzkin: QUBO parowe)?
- **Q2 (głębokość czy shoty):** czy przy odczycie energetycznym zysk z p przekłada się na wynik, czy tę samą
  poprawę daje taniej większa liczba shotów? Miara: shoty potrzebne do trafienia optimum okna z pr. 95%.
- **Q3 (szum i sprzęt):** przy jakim p* zysk z głębokości przegrywa z szumem, na symulatorze z szumem
  i na sirius (16 kubitów, gwiazda)?

Na końcu wszystko sprowadzamy do **Cmax i PRD w ILS**, tak jak we wszystkich poprzednich pracach.

## Co już mamy (sprawdzone 23.10)

- Kąty stałe dla p = 1…5, cztery sąsiedztwa: `data/qaoa_angles.json`. Funkcja celu kalibracji nasyca się od
  p ≈ 4 (Fibonacci −0,184 → −0,285 → −0,286 dla p = 1, 4, 5; Motzkin −0,089 → −0,302 → −0,304).
- Kalibracja (`src/experiments/qaoa_calibrate_angles.py`) przyjmuje `--p-max`; p ≥ 2 to Nelder–Mead z siewem
  INTERP, więc funkcja celu nie rośnie z p z konstrukcji.
- `solve.py` przyjmuje p. Backendy: `ibm`, `iqm`, `aer_noisy`.
- **p = 1 na sirius już policzone** (kampania ACIIDS, `articles/gate_qaoa/measurements/sirius_full_taillard_ils.json`):
  tai20_5/10/20, instancje 0–3, cztery sąsiedztwa, ILS 10 s, 4096 shotów; instancja 0 gęstych × 5 powtórzeń.
- Czas symulacji bez szumu (zmierzony): okno 16 kubitów, p = 10 → ~0,05–0,1 s na obwód (statevector
  i Aer 4096 shotów). **Symulacja bez szumu jest praktycznie darmowa.**

## Usterki i braki wykryte przy planowaniu

1. **`aer_noisy` nie ma modelu szumu**: `AerSimulator()` bez `noise_model`, więc mimo nazwy liczy bez szumu.
   W opublikowanych wynikach nie użyty (sprawdzone `git grep`). Do naprawy: rozdzielić na `aer_ideal`
   i `aer_noisy` z jawnym modelem.
2. **Brak fałszywego backendu z 16-kubitową gwiazdą.** W `iqm-client` 35 są Adonis (5), Deneb (6, gwiazda),
   Apollo (20, krata), Aphrodite (54). Sirius (16, gwiazda) trzeba zbudować przez `IQMFakeBackend`
   (architektura gwiazdy jak w Deneb, profil błędów z kalibracji sirius) albo przyjąć model szumu Apollo
   jako zastępczy. **Decyzja w E0.**
3. ~~Limit ruchów w ILS~~ — **niepotrzebny** (decyzja 23.10): ILS zapisuje `move_log` (najlepsze Cmax po
   każdym ruchu), a początek trajektorii przebiegu 10 s jest identyczny z przebiegiem uciętym na k ruchach.
   Porównanie „po tej samej liczbie ruchów” czytamy z `move_log`; `ils.py` bez zmian.
4. **Tabela kątów była liczona z `--perms 1`**, nie z domyślnym `--perms 3` (z domyślnymi nie odtwarza się
   nawet p = 1). Z `--perms 1` p = 1 zgadza się co do bitu we wszystkich czterech sąsiedztwach; p = 2 różni
   się szumem numerycznym Nelder–Meada (≤ 5·10⁻⁵ w funkcji celu). Kalibracja p ≤ 10 idzie z `--perms 1`.
5. **Mały zbiór treningowy kalibracji:** 21 okien (liniowe), 11 (gęste), z tai20_5 inst. 0–1 i tai20_10
   inst. 0, okna w = 6 (liniowe ≤ 5 zmiennych, a w ILS okna mają 16). Przy 20 kątach (p = 10) ryzyko
   dopasowania do treningu. W E1 okna z instancji treningowych raportować osobno; E2 mierzy przenoszenie.
6. **Stały `maxiter = 300` niezależnie od p** — przy p ≥ 5 Nelder–Mead nie dochodził do zbieżności, więc
   dotychczasowe „nasycenie przy p ≈ 4–5” mogło być artefaktem. Poprawka: `--maxiter-per-param` (budżet
   200 · 2p). Domyślne zachowanie bez zmian.
7. **Pierwszy pomiar po kroku 2 (23.10), tai20_5 inst. 0, pierwsze okno, p = 1:**
   - Motzkin (8 zmiennych): P_opt statevector 0,0383, `ideal` 0,0398 (zgodne w granicy szumu shotów),
     `aer_noisy`/Apollo 0,0161; odczyt energetyczny trafia optimum we wszystkich trzech.
   - **Fibonacci (16 zmiennych): P_opt < 5·10⁻⁵, czyli poniżej losowania jednostajnego (8/65 536 ≈ 1,2·10⁻⁴);**
     odczyt daje −64 przy optimum −68, także z szumem. Kąty p = 1 z okien ≤ 5 zmiennych nie przenoszą się
     na okno 16 zmiennych. To mocny argument za E2 i być może główny wynik o p: czy większe p (i kąty
     z dużych okien) to naprawia. Szum zbliża rozkład do jednostajnego, więc może nawet „pomóc” P_opt.
   - Koszt `aer_noisy` na Apollo: okno 16 kubitów ~15 s/obwód przy p = 1 (8 kubitów ~0,4 s). E4/E5 z szumem
     tylko na instancjach 0–3 i z pomiarem czasu przed pełnym uruchomieniem.
8. **ILS startuje z permutacji tożsamościowej** (nie z NEH) — tak jak we wszystkich dotychczasowych
   pracach. Okna w E1 też z permutacji tożsamościowej (pierwszy ruch ILS).

## Zakres

| Wymiar | Wartości |
|---|---|
| Instancje | tai20_5, tai20_10, tai20_20; **symulator: 10 instancji/plik (30)**, sprzęt: instancje 0–3 (12) |
| Sąsiedztwa | Adjacent, Fibonacci (rzadkie), Dynasearch, Motzkin (gęste) — **wszystkie cztery** |
| Okna | jak ACIIDS: liniowe w = 17 (16 zmiennych), gęste w = 6 (15 zmiennych), nakładanie 0,5 |
| p bez szumu | 1, 2, …, 10 |
| p z szumem | 1, 2, …, 6 |
| p na sirius | 1 (mamy), 2, 3 |
| Shoty | 4096 (jak wszędzie); w ILS na symulatorze dodatkowo 256 (pytanie Q2) |
| Ziarna | rzadkie deterministyczne → 1; gęste → 3 powtórzenia na symulatorze |

Dlaczego tylko n = 20: przy oknach o stałym rozmiarze większe n tylko mnoży liczbę okien, a o p nie mówi
nic nowego. Skalowanie w n jest w ACIIDS; tu go nie powtarzamy.

## Eksperymenty

### E0 — infrastruktura (bez QPU) · 24–27.10
- **Krok 1 (zrobione 23.10, liczy się):** kalibracja p = 1…10, `--perms 1 --maxiter-per-param 200`, cztery
  procesy w tle (nohup + caffeinate), wyniki `data/qaoa_angles_p10_<sąsiedztwo>.json`, logi w `logs/`.
  `data/qaoa_angles.json` (ACIIDS/GCMM) nietknięty. Po zakończeniu: p = 1 identyczne ze starą tabelą,
  monotoniczność w p, porównanie nowych p = 2…5 ze starymi (czy plateau było artefaktem).
- **Krok 1b (zrobione 23.10, liczy się):** druga kalibracja, **okna jak w ILS** (`--runtime-windows
  --no-filter-delta`; liniowe w = 17, krok 8, perms 3; gęste w = 6, krok 3, perms 1; po 18 okien,
  K = 11–16 liniowe, 4–15 gęste), p = 1…5, wyniki `data/qaoa_angles_runtime_<sąsiedztwo>.json`.
  Szybki test p = 1 (siatka na dużych oknach vs stare kąty z małych okien, funkcja celu):
  Adjacent −0,105 vs −0,028 (3,7×), Fibonacci −0,176 vs −0,154, Dynasearch −0,088 vs −0,081,
  Motzkin −0,081 vs −0,076. **Stare kąty p = 1 słabo przenoszą się na okna z ILS, najbardziej liniowe.**
  W artykule: dwie tabele kątów (małe okna vs okna z ILS) × p — rozdziela efekt głębokości od przenoszenia.
- **Krok 2 (zrobione 23.10):** `solve.py` — `ideal` (bez szumu), `aer_noisy` z jawnym modelem szumu fałszywego backendu
  IQM (`QAOA_NOISE_BACKEND`), tabela kątów z `QAOA_ANGLES`.
- **Krok 3:** fałszywy sirius (gwiazda 16 + rezonator) albo model zastępczy; sprawdzić, czy topologię
  i kalibrację prawdziwego siriusa da się pobrać z `iqm-client` bez uruchamiania zadania.
- Testy: regresja `tests/test_common_qubo_golden.py`.

### E1 — jakość okna bez szumu, p = 1…10 · 28–29.10
Zbiór okien: dla każdej z 30 instancji permutacja NEH, wszystkie okna pierwszego ruchu, cztery sąsiedztwa.
Optimum okna liczone brute force (≤ 16 zmiennych = 65 536 stanów). Dla każdego okna i p, ze statevectora
(dokładnie, bez szumu próbkowania):
- `P_opt(p)` — masa prawdopodobieństwa na optymalnych ciągach bitów;
- `r(p) = <E>/E_min` — stosunek aproksymacji (okna z E_min = 0, czyli bez ruchu poprawiającego, osobno);
- `S95(p) = ln 0,05 / ln(1 − P_opt)` — shoty do trafienia optimum z pr. 95% (pytanie Q2).

### E2 — kontrola kątów stałych · 29.10
Na podzbiorze okien (np. 40 na sąsiedztwo) kąty optymalizowane osobno dla każdego okna (Nelder–Mead ze
startu z kątów stałych), p = 1…5. Różnica kąty stałe vs optymalne mówi, czy nasycenie przy p ≈ 4 jest cechą
problemu, czy skutkiem przenoszenia kątów. **Bez tego recenzent zarzuci, że plateau to artefakt kalibracji.**

### E3 — koszt obwodu, p = 1…10 · 29–30.10 (darmowe)
Po kompilacji: liczba bramek dwukubitowych (CZ, MOVE), głębokość, głębokość dwukubitowa. Topologie:
gwiazda (sirius, przez obiekt backendu lub fałszywy), krata (Apollo/Garnet). Jedna tabela + wykres liniowy w p.

### E4 — jakość okna z szumem, p = 1…6 · 30.10–1.11
Te same okna co w E1, Aer z modelem szumu (E0). `P_opt`, `r`, `S95` w funkcji p → punkt przegięcia p*.

### E5 — ILS na symulatorze · 31.10–2.11 (w tle)
Ten sam protokół co na sprzęcie: **ILS, budżet 10 s, 4096 shotów**, start z permutacji tożsamościowej.
- Bez szumu: 30 instancji × 4 sąsiedztwa × p = 1…10 (+ 256 shotów dla pytania Q2), gęste × 3 powtórzenia.
- Z szumem: p = 1…6, 4096 shotów, instancje 0–3.
- Z jednego przebiegu (dzięki `move_log`) trzy porównania:

| Wariant | Co pokazuje | Skąd |
|---|---|---|
| sirius, 10 s (1–2 ruchy) | faktyczny wynik przy p = 1, 2, 3 | H2 (p = 1 z ACIIDS) |
| symulator, **ta sama liczba ruchów** co sprzęt | koszt szumu przy tym samym p | prefiks `move_log` |
| symulator, **pełne 10 s** | potencjał p = 1…10 bez latencji QPU | cały przebieg |

- Wynik: Cmax, PRD od dolnego ograniczenia Taillarda, **liczba ruchów w budżecie**, PRD po k ruchach
  (krzywe zbieżności), kolumny budżetów 100…10 000 ms jak w ACIIDS.

**Obserwacja do artykułu: czas ruchu rośnie z p.** Na symulatorze koszt obwodu rośnie z p (zmierzone:
okno 16 kubitów ~0,05 s przy p = 5, ~0,1 s przy p = 10), więc przy p = 10 w 10 s mieści się mniej ruchów niż
przy p = 1. Na sprzęcie czas ruchu to głównie latencja zadania, ale transpilacja i czas wykonania obwodu też
rosną z p — mierzymy, ile. Raportować zawsze liczbę ruchów obok PRD i czas ruchu w funkcji p (osobna
mała figura lub kolumna tabeli); krzywe PRD od liczby ruchów pokazują potencjał bez tego efektu.

### ILS z szumem — budżet w ruchach (decyzja 24.10.2026)
Na fałszywym siriusie jeden obwód liczy się 34–275 s, więc zegar laptopa nie odpowiada zegarowi siriusa
(10 s na sprzęcie = Adjacent 2 ruchy, Fibonacci ~2, gęste 1; na laptopie zawsze 1). **ILS z szumem dostaje
budżet w ruchach: tyle, ile sirius wykonał w 10 s w odpowiadającym przebiegu** (z `move_log` kampanii ACIIDS
dla p = 1; dla p = 2, 3 z H2). Wtedy różnica PRD symulator vs sprzęt pochodzi tylko z modelu szumu.

### H1 — sirius, jakość okna, p = 1, 2, 3 · 3–5.11
Okna z E1 dla instancji 0–3 (12 instancji × 4 sąsiedztwa), 4096 shotów. Empiryczne `P_opt` i trafienie
optimum przez odczyt energetyczny; porównanie z E1 (bez szumu) i E4 (szum) przy tym samym p.
Koszt: ~11 obwodów/instancję/p × 12 × 3 ≈ **400 obwodów ≈ 180–300 kredytów**.

### H2 — sirius, ILS, p = 2, 3 · 3–5.11
Dokładnie protokół ACIIDS (ILS 10 s, 4096 shotów, instancje 0–3, cztery sąsiedztwa), żeby p = 1 wziąć
wprost z istniejącego pliku. Koszt: ~15 obwodów/instancję × 12 × 2 ≈ **360 obwodów ≈ 160–270 kredytów**.
Opcjonalnie 5 powtórzeń gęstych na instancji 0 (jak dla p = 1): +~170 obwodów.

**Razem sprzęt: ~760 obwodów ≈ 340–570 kredytów** (stawka zmierzona 0,45–0,75 kr/obwód). Doładować przed 3.11.
Zabezpieczenia jak w ACIIDS: straż `_assert_real_counts`, zapis po każdym przebiegu, wznowienie.

## Statystyka
- Okna: Wilcoxon par (p vs p − 1, p vs 1) po oknach, korekta Holma.
- ILS: test znaków par po instancjach (jak w ACIIDS) + Friedman po p na symulatorze.
- Raportować mediany i rozstępy, nie tylko średnie; na sprzęcie wprost liczbę instancji.

## Ryzyka
- **Fałszywy sirius** może zjeść E0. Plan B: model szumu Apollo (krata) do E4 + transpilacja na prawdziwy
  obiekt sirius tylko do E3.
- **Kalibracja p = 6…10** (Nelder–Mead, 20 parametrów, okna gęste 15 kubitów) może być wolna. Plan B:
  p ≤ 10 tylko dla rzadkich, gęste do p = 6.
- **ILS na sprzęcie mieści 1–2 ruchy** w 10 s, więc różnice PRD między p będą małe. Główny sygnał p daje
  jakość okna (E1, E4, H1) i ILS na symulatorze (E5); H2 to potwierdzenie, nie główny wynik.
- **5.11 decyzja GCMM** — możliwy camera-ready w tym samym tygodniu co H1/H2.
- **Podwójne użycie danych:** p = 1 na sprzęcie pochodzi z pracy ACIIDS (w recenzji). Trzeba to ujawnić
  i zacytować (patrz PLAN_ARTYKULU.md, anonimizacja), a tekst metody napisać od nowa (autoplagiat).

## Harmonogram

| Daty | Co |
|---|---|
| 24–27.10 | E0 (kod do pokazania w czacie), kalibracja p ≤ 10, pomiar czasów |
| 28–30.10 | E1, E2, E3 |
| 30.10–2.11 | E4, E5 (w tle); szkic sekcji 1–4 |
| do 2.11 | doładowanie IQM |
| 3–5.11 | H1, H2 na sirius |
| 6–11.11 | wyniki, figury, sekcje 5–8 |
| 12.11 | humanizer, przegląd, bibliografia (Crossref), anonimizacja, metadane PDF |
| 13.11 | wysyłka (icaiscsystem.icaisc.eu) |

## Pliki (planowane)
- `src/experiments/qaoa_depth_study.py` — jeden skrypt kampanii (E1–E5, H1–H2 przełącznikiem).
- `articles/qaoa_depth_icaisc/measurements/*.json` — wyniki, zapis po każdym przebiegu.
- `articles/qaoa_depth_icaisc/figures/*.py` — figury (`pdf.fonttype = 42`).
- `data/qaoa_angles_p10.json` — kąty p = 1…10.
