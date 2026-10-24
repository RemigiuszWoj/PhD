# Wyniki do artykułu (ICAISC 2027, studium p) — zapis roboczy

Każda liczba ma źródło (plik lub polecenie). Stan: 24.10.2026.

## W1. Kalibracja kątów stałych dla trzech szerokości okien treningowych

Funkcja celu kalibracji = średnia znormalizowana energia ⟨H_C⟩/‖H‖₁ na oknach treningowych (im niżej, tym
lepiej). Kalibracja: siatka dla p = 1, Nelder–Mead z INTERP dla p ≥ 2, budżet 200 · 2p iteracji
(`--maxiter-per-param 200`), instancje treningowe tai20_5 inst. 0–1 i tai20_10 inst. 0.

| Tabela | Okna treningowe | Pliki |
|---|---|---|
| małe | w = 6, filtr Δ < 0, `--perms 1`; liniowe ≤ 5 zmiennych | `data/qaoa_angles_p10_<nb>.json` |
| duże bez akceleratora | liniowe w = 17 krok 8 (11 i 16 zmiennych, `--perms 3`), gęste w = 6 krok 3 bez filtra (4–15, `--perms 1`) | `data/qaoa_angles_runtime_<nb>_BEZ_AKCELERATORA.json` (liniowe), `data/qaoa_angles_runtime_<nb>.json` (gęste) |
| okna z ILS | przechwycone z sąsiedztw ILS (liniowe z akceleratorem bloków: 1–13 zmiennych, `--perms 3`); gęste = jak wyżej | `data/qaoa_angles_runtime_<nb>.json` |

| Tabela | Sąsiedztwo | p=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| małe | Adjacent | −0,059 | −0,176 | −0,242 | −0,265 | −0,314 | −0,338 | −0,338 | −0,358 | −0,361 | −0,362 |
| małe | Fibonacci | −0,184 | −0,234 | −0,266 | −0,286 | −0,287 | −0,288 | −0,290 | −0,290 | −0,293 | −0,295 |
| małe | Dynasearch | −0,057 | −0,185 | −0,303 | −0,314 | −0,324 | −0,325 | −0,327 | −0,327 | −0,328 | −0,328 |
| małe | Motzkin | −0,088 | −0,187 | −0,294 | −0,318 | −0,329 | −0,329 | −0,336 | −0,339 | −0,340 | −0,343 |
| duże bez akc. | Adjacent | −0,105 | −0,298 | −0,300 | −0,302 | −0,303 | | | | | |
| duże bez akc. | Fibonacci | −0,176 | −0,241 | −0,260 | −0,287 | −0,288 | | | | | |
| duże = ILS | Dynasearch | −0,088 | −0,339 | −0,340 | −0,341 | −0,341 | | | | | |
| duże = ILS | Motzkin | −0,081 | −0,115 | −0,340 | −0,340 | −0,340 | | | | | |
| okna z ILS | Adjacent | −0,087 | −0,173 | −0,200 | −0,219 | −0,225 | | | | | |
| okna z ILS | Fibonacci | −0,137 | −0,185 | −0,247 | −0,251 | −0,256 | | | | | |

Wartości między tabelami **nie są porównywalne** (każda na innym zbiorze okien) — porównanie na wspólnych
oknach daje E1. Kontrole: p = 1 tabeli „małe” identyczne co do bitu z `data/qaoa_angles.json` (ACIIDS,
GCMM) we wszystkich czterech sąsiedztwach; funkcja celu nierosnąca w p we wszystkich tabelach.

**Obserwacje:**
- **Stare plateau przy p ≈ 4 było artefaktem stałego `maxiter = 300`** (stara tabela p = 1…5): Adjacent
  −0,242 → −0,242 (p = 3 → 4) wobec nowej −0,242 → −0,265 → −0,314; Motzkin p = 5: −0,304 (stara) vs
  −0,329 (nowa). Fibonacci nasyca się naprawdę przy p ≈ 4 (−0,286 → −0,295 przy p = 10).
- Na małych oknach zysk z p trwa do p ≈ 8–10 (Adjacent, Motzkin); na dużych oknach funkcja celu nasyca
  się wcześnie: Adjacent i Dynasearch przy p = 2, Motzkin przy p = 3.
- Szybki test p = 1 (siatka na dużych oknach bez akceleratora vs stare kąty, 23.10): Adjacent −0,105 vs
  −0,028 (3,7×), Fibonacci −0,176 vs −0,154, Dynasearch −0,088 vs −0,081, Motzkin −0,081 vs −0,076 →
  kąty z małych okien słabo przenoszą się na większe, najbardziej w sąsiedztwach liniowych.
- Pierwsze okno Fibonacciego z ILS (tai20_5 inst. 0, 9 zmiennych, tabela małe, p = 1): P_opt = 1,75·10⁻⁴,
  losowanie jednostajne 1,95·10⁻³ — **QAOA p = 1 trafia optimum ~11× rzadziej niż losowanie**.
- Czas kalibracji (zegar): małe p ≤ 10: 0,7–2 h na sąsiedztwo; duże bez akc. p ≤ 5: Adjacent 5 h 55 min,
  gęste 2 h 11 min (Motzkin) i 2 h 24 min (Dynasearch); okna z ILS: 9 min (Fibonacci), 30 min (Adjacent).

## W2. Okna kalibracji a okna z ILS (uwaga metodyczna)

ILS buduje okna liniowe z kandydatów przefiltrowanych akceleratorem bloków, więc mają 1–13 zmiennych
(tai20, tożsamość), a nie w − 1 = 16. Okna gęste ILS buduje bez filtra — identyczne z kalibracją
„duże” (sprawdzone co do QUBO). Sprawdzenie: `build_training(..., runtime_windows=True)` ==
`capture_windows(...)` dla wszystkich 4 sąsiedztw.

## W3. Fałszywy sirius

Migawka kalibracji: `data/iqm_sirius_calibration_20261023.json` (zestaw b46eed6f…, 23.10.2026 22:42 UTC),
16 działających kubitów z 24, rezonator COMPR1 (T1 7,5 µs, T2 3,1 µs).

| Bramka | zmierzona błędność (mediana) | sama relaksacja T1/T2 (mediana) | relaksacja > zmierzona |
|---|---|---|---|
| PRX | 0,0011 | 0,0015 | 11/16 |
| CZ | 0,0167 | 0,0205 | 12/16 |
| MOVE | 0,0097 | 0,0129 | 12/16 |

→ tryb `calibrated` (domyślny; wierności dokładnie jak zmierzone, max odchyłka 2·10⁻¹⁴) i `thermal`
(o ~25–35 % bardziej pesymistyczny, górna granica szumu).

Kompilacja (obwód tai20_5 inst. 0, okno 0; fałszywy == prawdziwy sirius we wszystkich przypadkach):

| Okno | p | CZ | MOVE | r | głębokość |
|---|---|---|---|---|---|
| Fibonacci, 16 zm. | 1 | 30 | 30 | 59 | 76 |
| | 3 | 90 | 90 | 145 | 222 |
| | 6 | 180 | 180 | 274 | 441 |
| Motzkin, 15 zm. | 1 | 150 | 158 | 127 | 374 |
| | 3 | 450 | 478 | 351 | 1120 |
| | 6 | 900 | 958 | 687 | 2239 |

(okna z `_window_qubo` bez akceleratora; dla okien z ILS policzyć ponownie w E3.)

Czas symulacji z szumem (fałszywy sirius, 4096 shotów, 1 obwód): Fibonacci 16 zm. 34 / 78 / 128 s
(p = 1 / 3 / 6), Motzkin 15 zm. 53 / 149 / 275 s.

## W4. Backendy symulatora (kontrola)

Pierwsze okno Motzkina (8 zm., filtr Δ < 0), p = 1: P_opt statevector 0,0383, `ideal` 0,0398, `aer_noisy`
(Apollo) 0,0161. Stary `aer_noisy` był bez modelu szumu (nieużyty w opublikowanych wynikach).

## W5. E1 — jakość okien z ILS dla trzech tabel kątów (bez szumu, dokładnie)

Źródło: `articles/qaoa_depth_icaisc/measurements/e1_window_quality.jsonl` (30 240 wierszy; 24.10.2026,
96 s na 8 procesach). 1440 okien przechwyconych z ILS: tai20_5/10/20 × 10 instancji × {tożsamość, 2 losowe
permutacje} × 4 sąsiedztwa; 1195 okien z ruchem poprawiającym (E_min < 0) — tylko one w statystykach.
Mediany P_opt; „trafia opt.” = średnie 1 − (1 − P_opt)^S; „best S” = średnia E[najlepsza z S] / E_min
(1 = optimum, 0 = nic nie robić, < 0 = gorzej niż nic).

Losowanie jednostajne (p = 0): Adjacent P_opt 9,8·10⁻⁴, best4096 0,941; Fibonacci 5,9·10⁻³, 0,995;
Dynasearch 1,2·10⁻⁴, 0,574 (best256 −8,29); Motzkin 1,2·10⁻⁴, 0,730 (best256 −4,91). Mediana K: liniowe 10,
gęste 14.

| Sąsiedztwo | Tabela | p | P_opt / losowanie (mediana) | trafia opt. 256 | 4096 | best256 | best4096 |
|---|---|---|---|---|---|---|---|
| Adjacent | small | 1 | 17,5 | 0,967 | 0,999 | 0,995 | 1,000 |
| | small | 2 / 3 / 5 / 10 | 2,5 / 9,2 / 6,4 / 9,2 | 0,61–0,87 | 0,93–0,99 | 0,92–0,97 | 0,99–1,00 |
| | **big** | 1 / 2 / 3 / 5 | 6,0 / **55** / **72** / **99** | 0,71 / **1,000** / 1,000 / 1,000 | 0,99 / 1,00 | 0,94 / **1,000** | 1,000 |
| | ils | 2 / 3 / 5 | 15,7 / 23,4 / 13,5 | 0,90 / 0,92 / 0,85 | ≥ 0,99 | 0,97–0,99 | 1,000 |
| Fibonacci | small | 1 / 2 / **3** / 4 / 5 / 10 | **0,33** / 1,0 / **5,2** / 0,8 / 1,7 / 0,20 | 0,42 / 0,58 / **0,92** / 0,57 / 0,69 / 0,43 | 0,69 / … / 0,996 / … / 0,68 | 0,75 / … / 0,994 / … / 0,75 | 0,94 / … / 1,000 / … / 0,91 |
| | big | 1 / 3 / 4 | 0,35 / 1,75 / 2,1 | 0,42 / 0,70 / 0,73 | 0,74 / 0,89 / 0,97 | 0,75 / 0,95 / 0,96 | 0,95 / 0,99 / 1,00 |
| | ils | 1 / 2 / 3 / 4 | 0,35 / **0,08** / 4,9 / 3,7 | 0,42 / 0,39 / 0,89 / 0,84 | 0,74 / 0,60 / 0,99 / 0,99 | 0,75 / 0,64 / 0,99 / 0,98 | 0,95 / 0,86 / 1,00 / 1,00 |
| Dynasearch | **small** | 1 / 2 / 3 / 5 / 10 | **0,15** / 121 / 191 / **397** / 372 | 0,10 / 0,90 / 0,97 / 0,99 / 0,99 | 0,30 / 1,00 / 1,00 / 1,00 / 1,00 | **−16,9** / 0,97 / 0,995 / 0,999 / 0,998 | −2,54 / 0,999 / 1,000 |
| | big = ils | 1 / 2 / 3 / 5 | 3,2 / 10,9 / 46 / 24 | 0,32 / 0,52 / 0,75 / 0,64 | 0,73 / 0,85 / 0,90 / 0,85 | −2,0 / 0,78 / 0,91 / 0,84 | 0,88 / 0,95 / 0,98 / 0,97 |
| Motzkin | **small** | 1 / 2 / 4 / 5 / 10 | 1,2 / 77 / 129 / 64 / **236** | 0,20 / 0,79 / 0,91 / 0,80 / **0,975** | 0,53 / 0,99 / 1,00 / 1,00 / 1,00 | **−3,8** / 0,96 / 0,99 / 0,97 / 0,997 | 0,75 / 0,999 / 1,000 |
| | big = ils | 1 / 2 / 3 / 5 | 1,9 / 11 / 2,1 / 4,3 | 0,24 / 0,38 / 0,35 / 0,40 | 0,63 / 0,88 / 0,63 / 0,72 | −1,9 / 0,76 / 0,75 / 0,79 | 0,85 / 0,98 / 0,92 / 0,94 |

(Wiersze z „…” w Fibonaccim: pełne liczby w jsonl; podsumowanie skryptem z sesji 24.10.)

**Wnioski z E1:**
1. **p = 1 jest słabe wszędzie, a w dwóch przypadkach gorsze od losowania**: Fibonacci (0,33× losowania)
   i Dynasearch (0,15×). Przy p = 1 gęste sąsiedztwa przy 256 shotach wybierają średnio ruch **gorszy niż
   brak ruchu** (best256 −16,9 i −3,8): rozkład koncentruje się na stanach łamiących ograniczenia.
2. **Od p = 2 (przy dobrej tabeli) odczyt z 4096 shotów trafia optimum prawie zawsze** (best4096 ≥ 0,999
   dla Adjacent/big, Dynasearch/small, Motzkin/small). W ILS z 4096 shotami różnice między p ≥ 2 będą więc
   małe; widać je przy 256 shotach. → Pytanie „głębokość czy shoty” ma realną odpowiedź.
3. **Która tabela jest najlepsza, zależy od sąsiedztwa**: Adjacent — „big” (duże okna); Dynasearch
   i Motzkin — „small” (małe okna, filtr Δ < 0), choć „big” trenowano na tych samych oknach co ILS.
   Fibonacci — żadna nie jest stabilna; najlepsze p = 3 (small i ils, ~5× losowania).
4. **Funkcja celu kalibracji (średnia energia) nie przekłada się na P_opt**: dla gęstych tabela „big” ma
   lepszą własną funkcję celu (−0,339 przy p = 2 wobec −0,185) i wyraźnie gorsze P_opt na oknach z ILS.
   Minimalizacja ⟨H_C⟩ nagradza unikanie kar, a nie trafianie optimum. Ważna uwaga metodyczna.
5. **Kąty stałe nie dają monotonii w p na oknach testowych** (np. Fibonacci small: p = 3 → 5,2×,
   p = 4 → 0,8×, p = 10 → 0,2×), choć na treningu funkcja celu z konstrukcji nie rośnie.
6. Instancje treningowe nie zawyżają wyników (Fibonacci small p = 1: best4096 0,954 na 17 oknach
   treningowych vs 0,938 na 152 pozostałych).
7. Tabele big i ils mają **identyczne kąty p = 1** dla Adjacent i Fibonacci (ten sam punkt siatki 25 × 13),
   stąd identyczne wiersze p = 1.
