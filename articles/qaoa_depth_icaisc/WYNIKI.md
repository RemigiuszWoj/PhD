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

## W6. E5 — ILS na Taillardzie bez szumu (backend `ideal`, budżet 10 s)

Źródło: `measurements/e5_ils_ideal.jsonl` (8880 przebiegów, 24.10.2026 11:44–14:52, 8 procesów).
30 instancji tai20_5/10/20; PRD od dolnego ograniczenia Taillarda, uśredniony po powtórzeniach w obrębie
instancji, potem mediana po 30 instancjach; w nawiasach średnia liczba ruchów w 10 s. Test znaków par po
instancjach (wygrane/przegrane, p dwustronne).

**Klasyczne DP (ten sam ILS, dokładne sąsiedztwo):** Adjacent 10,57 (27 332 ruchy), Fibonacci 10,14
(87 904), Dynasearch 8,88 (5059), Motzkin 8,95 (6687).

| Sąsiedztwo | S | p=0 | p=1 | p=2 | p=3 | p=4 | p=5 | p=6 | p=8 | p=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Adjacent | 256 | 16,09 (1464) | 17,42 (1001) | 19,26 | 16,25 | 19,24 | 20,39 | 19,93 | 20,77 | 20,82 (289) |
| | 4096 | 19,67 (354) | 18,06 (363) | 20,73 | 20,47 | 21,71 | 20,38 | 21,38 | 22,11 | 22,92 (183) |
| Fibonacci | 256 | 15,49 (1799) | 13,65 (1294) | 14,85 | 15,48 | 14,79 | 16,01 | 20,21 | 17,16 | 18,20 (560) |
| | 4096 | 17,20 (481) | 17,96 (502) | 17,34 | 18,11 | 19,86 | 18,59 | 19,39 | 17,11 | 21,28 (320) |
| Dynasearch | 256 | 24,22 (47) | 26,20 (39) | **14,99** | 15,58 | 15,49 | 16,14 | 15,44 | 16,06 | 17,99 (18) |
| | 4096 | 20,04 (28) | 22,65 (26) | **15,19** | 16,52 | 15,63 | 16,37 | 16,49 | 17,06 | 20,87 (17) |
| Motzkin | 256 | 23,81 (48) | 25,27 (41) | 14,32 | **14,00** | 15,74 | 14,37 | 14,71 | 16,37 | 17,55 (20) |
| | 4096 | 20,83 (30) | 21,84 (28) | 16,92 | 16,54 | 16,23 | **15,86** | 16,07 | 16,09 | 18,68 (18) |

(tabela „small”; p = 7, 9 w jsonl.) Tabela „big” (Adjacent) i „ils” (Fibonacci) nie poprawiają wyników
liniowych (np. Adjacent big p = 3: 17,89 / 21,51; Fibonacci ils p = 3: 15,60 / 18,13 dla S = 256 / 4096).

**Testy znaków (tabela small):**
- Dynasearch: p = 1 vs p = 0 przegrywa 4/26 (S = 256, p < 0,001) i 3/27 (S = 4096, p < 0,001); p = 3 vs p = 1
  wygrywa 30/0 i 29/1; najlepsze p = 2 vs p = 0: 30/0 i 27/3 (p < 0,001).
- Motzkin: p = 1 vs p = 0: 11/19 (p = 0,20) i 8/22 (p = 0,016); p = 3 vs p = 1: 30/0 i 27/3; najlepsze p
  (3 / 5) vs p = 0: 30/0 i 29/1.
- Adjacent, Fibonacci: brak istotnej poprawy z p; p = 10 vs p = 1 przegrywa (Adjacent 4/25 przy S = 256,
  p < 0,001; Fibonacci 7/22, p = 0,008).

**PRD po tej samej liczbie ruchów (15 ruchów, gęste; prefiks `move_log`), S = 256 / 4096:**
Dynasearch p = 0: 29,07 / 22,73; p = 1: 29,17 / 25,43; p = 2: 20,03 / 20,24; p = 3: 19,16 / 19,90; p = 6:
18,00 / 20,12; p = 8: 18,87 / 18,10. Motzkin p = 0: 27,65 / 23,57; p = 1: 29,47 / 24,72; p = 2: 19,02 / 19,20;
p = 6: 19,51 / 17,98. (Liniowe: część przebiegów przy wysokim p ma < 150 ruchów — próg do dobrania.)

**Wnioski z E5:**
1. **Sąsiedztwa gęste: głębokość pomaga, i to mocno.** p = 1 jest gorsze od losowania (Dynasearch
   istotnie), a już p = 2 obniża PRD z ~24 do ~15 % (S = 256) — wynik zgodny z E1. Najlepsze p = 2–5;
   powyżej PRD znowu rośnie, bo przy budżecie 10 s głębszy obwód to mniej ruchów (47 → 18).
2. **Przy równej liczbie ruchów zysk z p ≥ 2 zostaje** (Dynasearch 15 ruchów: 29 → 18–20 %), więc nie jest
   artefaktem liczby ruchów; spadek przy dużym p w 10 s to koszt czasu, nie jakości.
3. **Sąsiedztwa liniowe: p nie pomaga; losowanie (p = 0) jest tak samo dobre.** Okna liniowe mają medianę
   10 zmiennych (2¹⁰ = 1024 stany), więc 256–4096 losowych próbek z odczytem energetycznym prawie
   przeszukuje okno w całości; rozkład QAOA nie ma tu czego poprawić. Wyższe p tylko spowalnia ruch.
4. **Więcej shotów na symulatorze to mniej ruchów**: w sąsiedztwach liniowych S = 4096 daje gorsze PRD niż
   S = 256 (Adjacent p = 0: 19,67 vs 16,09), bo losowanie 4096 próbek kosztuje czas (354 vs 1464 ruchy).
5. **Klasyczne DP jest wyraźnie lepsze** (PRD 8,9–10,6 % przy tysiącach ruchów) — w artykule bez tezy
   o przewadze; wartość wyniku leży w tym, kiedy i ile warstw QAOA się opłaca.

## W7. E3 — koszt obwodu w funkcji p (gwiazda vs krata)

Źródło: `measurements/e3_circuit_cost.jsonl` (192 okna z ILS: tai20 × instancje 0–3 × tożsamość × 4 sąsiedztwa;
kompilacja `transpile_to_IQM`, optimization_level 1; kąty z tabeli „small”). Mediany liczby bramek CZ:

| Sąsiedztwo | topologia | p = 1 | p = 3 | p = 6 | p = 10 |
|---|---|---|---|---|---|
| Adjacent | gwiazda (sirius) | 72 | 216 | 432 | 720 |
| | krata (Apollo) | 118 | 396 | 800 | 1356 |
| Fibonacci | gwiazda | 15 | 45 | 90 | 150 |
| | krata | 15 | 45 | 90 | 150 |
| Dynasearch | gwiazda | 164 | 492 | 984 | 1640 |
| | krata | 306 | 990 | 2036 | 3443 |
| Motzkin | gwiazda | 134 | 402 | 804 | 1340 |
| | krata | 256 | 832 | 1712 | 2886 |

Koszt rośnie dokładnie liniowo z p. Fibonacci (QUBO trójdiagonalne = łańcuch) mapuje się na kratę bez SWAP-ów,
więc obie topologie dają to samo. Dla QUBO gęstych (Adjacent one-hot, Dynasearch, Motzkin) gwiazda potrzebuje
~1,6–2,1× mniej CZ niż krata (krata dokłada SWAP-y); gwiazda płaci za to bramkami MOVE (~tyle samo co CZ).
Gęste okno przy p = 6 to ~800–1000 CZ na gwieździe — na tym tle E4 pokaże, ile z tego zostaje po szumie.

## W8. E4 — jakość okien z szumem (fałszywy Sirius)

Źródło: `measurements/e4_window_noise.jsonl` (1482 obwody, 24.10 15:12 – 25.10 11:19, z przerwą 4:27–10:03 po
błędzie zakresu p w trybie thermal). Okna pierwszego ruchu z ruchem poprawiającym, instancje 0–3 (161 okien),
4096 shotów, tryb `calibrated`; `thermal` tylko instancja 0, p ∈ {1, 3, 6} (Adjacent „big”: 1, 3). Miara główna:
odsetek okien, w których odczyt energetyczny (najlepsza z 4096 próbek) trafia optimum; bez szumu = średnie
hit_4096 z E1 na tych samych oknach. Figura: `figures/fig3_noise_*.pdf`.

| Sąsiedztwo (tabela) | p=0 | p=1 | p=2 | p=3 | p=4 | p=5 | p=6 |
|---|---|---|---|---|---|---|---|
| Adjacent (big), szum | 0,88 | 0,88 | **1,00** | **1,00** | 0,83 | 0,88 | — |
| Fibonacci (small), szum | 0,96 | 0,74 | 0,91 | **1,00** | 0,87 | 0,96 | 0,91 |
| Dynasearch (small), szum | 0,54 | 0,47 | **0,75** | 0,72 | 0,70 | 0,61 | 0,58 |
| Motzkin (small), szum | 0,53 | 0,53 | **0,91** | 0,79 | 0,72 | 0,60 | 0,56 |

Bez szumu (te same okna) gęste mają 1,00 dla każdego p ≥ 2. P_opt z szumem w gęstych oknach spada o rząd
wielkości i więcej (Dynasearch p = 3: 7,3·10⁻⁴ z szumem wobec 2,6·10⁻² bez), często do rozdzielczości
1/4096 = 2,4·10⁻⁴.

**Wnioski z E4:**
1. **Z szumem optimum głębokości to p* = 2 dla sąsiedztw gęstych** (Dynasearch 0,75, Motzkin 0,91); powyżej
   każda warstwa (~150–180 CZ + MOVE na gęste okno) obniża trafienie i przy p = 6 wraca ono do poziomu
   losowania (0,54–0,58). Bez szumu zysk utrzymywał się do p = 10.
2. **Sąsiedztwa liniowe znoszą szum dobrze** — okna są małe (mediana 10 zm.), obwody płytkie (15–72 CZ przy
   p = 1), a losowanie i tak trafia optimum w 88–96 % okien; najlepsze p = 2–3 (trafienie 1,00).
3. **p = 1 jest równie słabe z szumem jak bez** (gęste ≈ losowanie) — szum nie zmienia oceny p = 1.
4. **Tryb thermal** na instancji 0 daje podobny obraz jak calibrated (mała próba; czasem lepiej, czasem gorzej),
   więc wniosek o p* nie zależy od wyboru modelu szumu.
5. W Fibonaccim szum bywa „pomocny” (p = 2, 5, 6: wyżej niż bez szumu) — zbliża rozkład do jednostajnego,
   a ten przy małych oknach trafia optimum często (spójne z W5/W6).

## W9. E5n — ILS z szumem (fałszywy Sirius) przy liczbie ruchów Siriusa

Źródło: `measurements/e5n_ils_noisy.jsonl.gz` (624 przebiegi, 25.10 11:19 – 26.10 19:57, 8 procesów). Każdy z 72
przebiegów Siriusa p = 1 (tai20_5/10/20, inst. 0–3; w gęstych instancja 0 każdego pliku ma 5 powtórzeń, pozostałe po 1)
powtórzony w symulatorze z szumem (`calibrated`, 4096 shotów) z budżetem w ruchach równym liczbie ruchów Siriusa:
1 ruch (gęste), 2 ruchy (Adjacent, Fibonacci; Fibonacci raz 3). p = 0 (losowanie) … 6, tabela „small”; dodatkowo
Adjacent „big” (p ≤ 5) i Fibonacci „ils” (p ≤ 5). Start ILS jest deterministyczny (permutacja tożsamościowa),
więc porównania w parach mierzą wyłącznie jakość 1–2 ruchów. PRD: średnia po powtórzeniach w instancji, potem
mediana po 12 instancjach; testy znaków po instancjach (wygrane/przegrane, p dwustronne).

| Sąsiedztwo | p=0 | p=1 | p=2 | p=3 | p=4 | p=5 | p=6 | Sirius p=1 |
|---|---|---|---|---|---|---|---|---|
| Adjacent (small) | 34,91 | 34,49 | 34,49 | 34,91 | 34,91 | 34,49 | 34,49 | 34,91 |
| Adjacent (big) | | 34,49 | 34,49 | 34,49 | 34,49 | 34,49 | | |
| Fibonacci (small) | 29,31 | 30,71 | 29,80 | 28,77 | 30,05 | 30,04 | 31,46 | 29,85 |
| Fibonacci (ils) | | 31,22 | 30,47 | 30,42 | 29,00 | 30,84 | | |
| Dynasearch (small) | 32,99 | 35,05 | 30,74 | **29,45** | 31,44 | 32,21 | 36,15 | 31,77 |
| Motzkin (small) | 33,59 | 35,67 | 34,40 | 35,10 | 33,36 | **33,25** | 34,62 | 32,07 |

Bez szumu przy tej samej liczbie ruchów (prefiks `move_log` z E5, S = 4096): Dynasearch 33,53 / 35,23 / 32,43 /
33,23 / 30,78 / 32,40 / 32,37 (p = 0…6), Motzkin 34,96 / 34,98 / 31,78 / 32,30 / 32,78 / 31,87 / 31,96.

**Testy znaków (small):**
- Dynasearch: p = 1 vs p = 0 przegrywa 2/10 (p = 0,039); p = 2 vs p = 1 wygrywa 11/1 (p = 0,006); p = 3, 5 vs
  p = 1: 9/3 (p = 0,15); p = 4: 9/2 (p = 0,065); p = 6: 6/4 (p = 0,75).
- Motzkin: p = 4 i p = 6 vs p = 1: 9/2 (p = 0,065); pozostałe p ≥ 0,23; p = 1 vs p = 0: 3/7 (p = 0,34).
- Adjacent, Fibonacci: brak istotnych różnic (Adjacent: różnice ≤ 0,4 pp, przeważają remisy).
- Symulator z szumem vs Sirius, p = 1: Adjacent 4/0 (p = 0,125), Fibonacci 2/4 (0,69), Dynasearch **2/9**
  (p = 0,065; po przebiegach 4/19, p = 0,003), Motzkin 3/7 (0,34) — symulator jest dla p = 1 gorszy od sprzętu.
- Sirius p = 1 vs losowanie z szumem (p = 0): Dynasearch 6/5, Motzkin 6/4, Fibonacci 3/5 (wszystkie p ≥ 0,73).
- Szum vs brak szumu przy tej samej liczbie ruchów: żadna różnica nieistotna (wszystkie p ≥ 0,13).

**Wnioski z E5n:**
1. **Dynasearch potwierdza p* = 2–3 także w ILS z szumem**: p = 1 jest gorsze od losowania, p = 2 wyraźnie
   lepsze od p = 1, a przy p = 6 PRD wraca do poziomu p = 1 — ten sam obraz co w E4 (okna) i E5 (bez szumu).
2. **Przy 1–2 ruchach na 12 instancjach test ma małą moc**: poza Dynasearch różnice mieszczą się w kilku pp
   i nie są istotne. Tabelę 3 opisujemy jako „ten sam kierunek, małe różnice”, bez tezy o istotności poza
   Dynasearch.
3. **Model `calibrated` jest dla p = 1 bardziej pesymistyczny niż sprzęt, a Sirius przy p = 1 zachowuje się
   jak losowanie** (gęste: 6/5 i 6/4 wobec p = 0). Najprostsze wyjaśnienie: prawdziwy szum jest silniejszy
   i spłaszcza rozkład do jednostajnego, a ten w gęstych sąsiedztwach jest lepszy od p = 1 (spójne z W8.5).
   Dla H2: zysk z p = 2 na sprzęcie może być mniejszy, niż pokazuje symulator.
4. Ziarna są odtwarzalne: łańcuch uruchamiał E5n z `PYTHONHASHSEED=0` (kod bez tego się nie uruchomi),
   a `abs(hash(key))` w Pythonie 3.11 z `.venv-iqm` daje dokładnie zapisane ziarna (sprawdzone 26.10).
