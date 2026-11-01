# Katalog sąsiedztw i plan badań

Stan: 19.10.2026. Wszystkie liczności zweryfikowane brute-force (n ≤ 9–14) skryptami z sesji.

## Wspólna konwencja

Ruch to **zbiór elementarnych przestawień na rozłącznych fragmentach permutacji**. W QUBO: zmienna = kandydat
(przestawienie), przekątna = delta makespanu, kara poza przekątną = konflikt dwóch kandydatów. Pojedyncze ruchy
(Adjacent, Swap, Insert) mają QUBO one-hot (wybierz dokładnie jeden).

Gdy kandydatem jest zamiana końców przedziału [i, j], ruch jest **inwolucją** (zbiorem rozłącznych zamian), a
rodziny różnią się tylko geometrią łuków (i, j).

## Hierarchia (łańcuch zawierań)

Adjacent (1 zamiana) ⊂ **Fibonacci** ⊂ **tribonacci** ⊂ … ⊂ **k-bonacci** ⊂ **Dynasearch**
⊂ **Motzkin** ⊂ (≤ c krzyżowań) ⊂ **inwolucje**

Trzy osie parametrów:
- **k** — maksymalna długość łuku: Adjacent/Fibonacci (k = 2) → k-bonacci → Dynasearch (k = n);
- **c** — dopuszczalna liczba krzyżowań łuków: Motzkin (c = 0) → wszystkie inwolucje (c = ∞);
- **d** — odległość Hamminga, czyli liczba zamian ≤ d/2 (Hamming „z zamian”).

Czwarta oś, poza inwolucjami: **okno de Montmorta** — w oknie każde zadanie zmienia miejsce, więc dochodzą
**rotacje**, czyli ruchy wstawiania.

## Katalog

| # | Sąsiedztwo | Ruch | Liczba ruchów | QUBO | Status |
|---|---|---|---|---|---|
| 1 | **Adjacent** | jedna zamiana sąsiedniej pary | n − 1 | one-hot | mamy |
| 2 | Swap | jedna zamiana dowolnej pary (= Hamming d = 2) | n(n − 1)/2 | one-hot | punkt odniesienia |
| 3 | Insert | przesunięcie jednego zadania | (n − 1)² | one-hot | punkt odniesienia (klasycznie najsilniejszy dla PFSP) |
| 4 | **Fibonacci** | rozłączne zamiany sąsiadów | F(n + 1), ~1,618ⁿ | trójdiagonalne | mamy |
| 5 | **tribonacci / k-bonacci** | rozłączne zamiany końców przedziałów długości ≤ k, środek zablokowany | k-bonacci, ~1,839ⁿ (k = 3) | pasmowe, ≈ (k − 1)n zmiennych | **ICAPS 14.12.2026** |
| 6 | **Dynasearch** | rozłączne przedziały dowolnej długości | 2ⁿ⁻¹ | parowe (gęste) | mamy |
| 7 | **Motzkin** | łuki bez krzyżowań, zagnieżdżanie wolno | liczby Motzkina, ~3ⁿ | parowe | mamy |
| 8 | ≤ c krzyżowań (N_c) | jak Motzkin + do c przecięć łuków w całym ruchu | ~M_n·(n/3)^c/c!, wzrost wciąż 3ⁿ | parowe | **plan: [`crossings/PLAN.md`](crossings/PLAN.md)** (sprawdzić OEIS) |
| 8b | P₁ („pojedyncze krzyżowania”) | każdy łuk krzyżuje się co najwyżej raz, par krzyżujących się może być wiele | ~3,4626ⁿ | parowe | w planie krzyżowań |
| 9 | Inwolucje | dowolne rozłączne zamiany | liczby inwolucji (szybciej niż wykładniczo) | parowe, konflikt = wspólny koniec (najrzadsze QUBO z parowych) | w planie krzyżowań; wybór = skojarzenie o maks. wadze |
| 10 | **Hamming z zamian** | ≤ d/2 rozłącznych zamian na dowolną odległość | n = 20, d = 4: 14 726 | parowe + ograniczenie liczności (zmienne pomocnicze) | **Zakopane 15.11.2026** |
| 11 | Hamming pełny | dowolna permutacja w odległości ≤ d (z cyklami) | Σ C(n, j)·!j | przypisaniowe, n² zmiennych | później, trudne |
| 11b | Hamming „repermutacja” (wariant z ROADMAP) | wybór d pozycji i optymalne ich przestawienie | — | przypisaniowe, ~d² zmiennych | wariant do rozważenia |
| 12 | **Okno de Montmorta** | rozłączne okna, w każdym deranżacja (każde zadanie zmienia miejsce) | k = 3: c(n) = c(n−1) + c(n−2) + 2c(n−3), wzrost dokładnie 2ⁿ | zmienna = (okno, deranżacja), konflikt = nachodzące okna | nowe, obiecujące (zawiera wstawienia) |

### Liczności dla porównania (zweryfikowane)

| n | Fibonacci | tribonacci | Dynasearch | Motzkin | ≤ 1 krzyż. | ≤ 2 krzyż. | P₁ | inwolucje | de Montmort k = 3 | de Montmort k = 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 8 | 13 | 16 | 21 | 26 | 26 | 26 | 26 | 18 | 34 |
| 7 | 21 | 44 | 64 | 127 | 204 | 225 | 204 | 232 | 73 | 185 |
| 9 | 55 | 149 | 256 | 835 | 1 717 | 2 221 | 1 753 | 2 620 | 293 | 1 077 |
| 10 | 89 | 274 | 512 | 2 188 | 5 038 | 7 108 | 5 263 | 9 496 | — | — |

Deranżacje: !2 = 1, !3 = 2, !4 = 9, !5 = 44. Hamming pełny przy n = 20: d = 2 → 191, d = 4 → 46 076
(z ruchem pustym). Uwaga: „liczby Hamminga” w matematyce to 2ᵃ3ᵇ5ᶜ — w tekstach pisać „sąsiedztwo w odległości
Hamminga”.

## Decyzje i sprawy otwarte

- **19.10.2026:** na razie **bez krzyżowania**; w k-bonaccim środek przedziału jest **zablokowany**.
- **01.11.2026:** plan rodziny krzyżowań (Motzkin → N₁ / P₁ → inwolucje): [`crossings/PLAN.md`](crossings/PLAN.md). Pilot: przy wyborze
  addytywnym krzyżowania szkodzą (nieaddytywność rośnie, gdy zamiany na siebie nachodzą), a **bezpiecznik** (ruch złożony albo najlepsza
  pojedyncza zamiana, wg rzeczywistego C_max) poprawia Motzkina przy 50×10 z 17,3% do 5,9% RPD, czyli do poziomu Dynasearch w nowym kodzie
  (5,75% bez bezpiecznika; Dynasearchowi bezpiecznik szkodzi). Dotyczy to też wcześniejszych wyników Motzkina.
- **Do rozstrzygnięcia z promotorem — co robi blok długości 3 w tribonaccim:**
  (a) zamiana końców (i, i+2), środek stoi — liczności dokładnie k-bonacci, k = n daje dokładnie Dynasearch
  (wersja z 19.10);
  (b) jedna rotacja (3-cykl) — też dokładnie tribonacci (wersja z ROADMAP, lipiec);
  (c) obie rotacje — to już okno de Montmorta (waga !3 = 2).
- **Nowość:** sprawdzić, czy prace promotora (Sci Rep 2026 o wykładniczym sąsiedztwie Fibonacciego, IFAC 2025
  o nieprzecinających się przedziałach) nie zawierają już tej hierarchii.
- **Koszt delt:** zamiana zadań odległych o L − 1 pozycji jest droższa w ocenie niż zamiana sąsiednia — policzyć
  koszt przez Head/Tail.
- **Addytywność:** DP i QUBO zakładają sumowanie się delt — przybliżenie, którego słabość widać w ACIIDS
  (nieaddytywne delty w sąsiedztwach gęstych).

## Plan badań dla rodzin parametrycznych (k-bonacci, c krzyżowań, Hamming d, de Montmort)

**Faza 0 — implementacja i walidacja (bez QPU).** Jeden moduł na rodzinę, sparametryzowany (k, c, d), nie kopia
na każdą wartość. Testy: liczność sąsiedztwa = wzór (brute-force dla n ≤ 10), DP najlepszego ruchu = brute-force.

**Faza 1 — pilot na małych zbiorach.** tai20_5, tai20_10, tai20_20 i tai50_10, 4 instancje, 5 ziaren, ILS
(opcjonalnie SA), budżety 100–10 000 ms. Przegląd parametrów:
- k ∈ {2, 3, 4, 5, 6, 8, n} (k = 2 to Fibonacci, k = n to Dynasearch — kontrola zgodności z tym, co mamy);
- c ∈ {0, 1, 2, 3, ∞};
- d ∈ {2, 4, 6, 8};
- okno de Montmorta k ∈ {2, 3, 4}.

Mierzymy: PRD, czas na iterację, rozmiar sąsiedztwa, liczbę zamian w ruchu, krzywe zbieżności. Wynik fazy:
wybór 2–3 wartości parametru na rodzinę (najlepsza + skrajne).

**Faza 2 — pełny Taillard.** 12 plików × 10 instancji, wybrane parametry, protokół jak w `windowed_qubo` (ILS,
5 ziaren, te same budżety), żeby wyniki były porównywalne z wcześniejszymi pracami.

**Faza 3 — strona kwantowa (opcjonalnie, za darmo).** Rozmiar i szerokość pasma QUBO w funkcji parametru, QAOA na
symulatorze, głębokość obwodu po kompilacji na fałszywych backendach (IBM, IQM). Sprzęt tylko punktowo.

## Przydział do publikacji

| Termin | Konferencja | Treść |
|---|---|---|
| 15.11.2026 | ICAISC 2027, Zakopane (recenzja dwustronnie ślepa) | Hamming z zamian (#10) |
| 7.12 / 14.12.2026 | ICAPS 2027 (A*) | hierarchia k-bonacci (#4–#6) w zapisie macierzowym (rozłączne kwadraty) |
| 18.12.2026 | ANNSIM 2027 (przewodniczący: promotor), 4 strony | głębokość QAOA p przy szumie (sąsiedztwa gęste) |
| 11.01.2027 | IJCAI 2027 (A*) | zapas dla k-bonacci, jeśli nie zdążymy na ICAPS |
| ~I/II 2027 (do potwierdzenia) | GECCO 2027, zapas SOCO 2027 | krzyżowania: Motzkin → N₁ / P₁ → inwolucje (#7–#9); plan w [`crossings/PLAN.md`](crossings/PLAN.md) |
| 2027 | — | okno de Montmorta (#12), job shop |
