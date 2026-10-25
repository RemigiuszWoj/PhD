# Artykuł A*: od Fibonacciego do Dynasearcha — rodzina sąsiedztw k-bonacci, klasycznie i kwantowo

Plan z 25.10.2026 (wersja 1). Cel: konferencja A* — **ICAPS 2027** (abstrakt 7.12, artykuł 14.12.2026) albo
**IJCAI 2027** (abstrakt 4.01, artykuł 11.01.2027); decyzja otwarta (rekomendacja: IJCAI — jakość A* nie zmieści
się w grudniowym tłoku z ICAISC 15.11 i ANNSIM 18.12). Format obu: styl AAAI/IJCAI, ~7–8 stron + bibliografia,
recenzja dwustronnie ślepa (sprawdzić w CFP).

**Przed startem: uzgodnić z promotorem** podział z jego szkicem „Tribonacci, Tetranacci and k-nacci
neighborhoods” (TWT, D-Wave; pamięć `project_tribonacci_supervisor`), współautorstwo i cytowanie.

## 1. Rodzina sąsiedztw i teza

N_k(π): złożenia rozłącznych, nieprzecinających się zamian końców przedziałów o długości ≤ k, środek przedziału
stoi (wariant (a), zgodny z promotorem). Łańcuch zawierań:

Adjacent ⊂ **Fibonacci (k = 2) ⊂ tribonacci (k = 3) ⊂ … ⊂ k-bonacci ⊂ Dynasearch (k = n)**

Z k rosną: liczność (~1,618ⁿ → 2ⁿ⁻¹), koszt wyboru najlepszego ruchu (DP O(nk)), rozmiar QUBO (~(k−1)n
zmiennych) i jego szerokość pasma, a więc głębokość obwodu QAOA i trudność embeddingu na wyżarzaczu.

**Pytanie główne:** jak k wpływa na jakość rozwiązania (PRD) przy klasycznym i kwantowym przeszukiwaniu
sąsiedztwa i gdzie leży k* — osobno dla klasyki, wyżarzacza i komputera bramkowego.

**Teza robocza:** klasycznie najlepsze k rośnie z n i budżetem (duże sąsiedztwa się opłacają, gdy DP jest
tani); kwantowo opłaca się tylko małe k (tribonacci, k = 4), bo szerokość pasma QUBO szybko wyczerpuje kubity
i głębokość obwodu (E4 z ICAISC: z szumem gęste QUBO tracą zysk już od p > 2).

## 2. Wkład teoretyczny (wymóg A*)

- Liczność |N_k(n)| = liczby k-bonacciego (rekurencja c(n) = Σ_{L=1..k} c(n−L)), dowód kombinatoryczny;
  tempo wzrostu φ_k → 2; |N_n| = 2ⁿ⁻¹ (Dynasearch). Weryfikacja brute force już jest (`katalog_sasiedztw.md`).
- Zawierania N_k ⊂ N_{k+1}; średnica grafu sąsiedztwa w funkcji k.
- Algorytm DP O(nk) dla najlepszego ruchu przy addytywnych deltach + jawne potraktowanie nieaddytywności Cmax
  w PFSP (DP wybiera według sumy delt → dokładna ocena wybranego ruchu).
- QUBO wyboru zamian dla N_k: liczba zmiennych, graf konfliktów (graf przedziałów), szerokość pasma ~k;
  koszt warstwy QAOA (liczba RZZ) i długość łańcuchów embeddingu w funkcji k.

## 3. Eksperymenty

Wartości k: **{2, 3, 4, 5, 6, 8, 12, n}** (k = 2 Fibonacci i k = n Dynasearch jako kontrola zgodności z tym, co mamy).

| # | Co | Zakres Taillarda | Koszt |
|---|---|---|---|
| **C0** | implementacja jednego modułu sparametryzowanego k (klasyczne DP, QUBO), testy: liczność = wzór (n ≤ 12), DP = brute force | — | dni pracy |
| **C1** | pilot: ILS, 4 instancje, 5 ziaren, budżety 100–10 000 ms | tai20_5/10/20, tai50_10 | godziny |
| **C2** | **klasyka, pełny Taillard**: ILS i SA (+ IG z N_k jako fazą lokalną), wszystkie k, wyrównany czas | **120 instancji, n = 20…500** | ~1–2 doby CPU (bez kredytów) |
| **C3** | krajobraz: minima lokalne, autokorelacja, baseny w funkcji k | podzbiór (n = 20, 50) | godziny |
| **Q1** | rozmiar QUBO, szerokość pasma, liczba sprzężeń vs k; CZ/MOVE i głębokość po kompilacji (gwiazda Sirius, krata) | **pełny Taillard** (tylko liczenie) | darmowe |
| **Q2** | **wyżarzacz D-Wave**, okienkowe QUBO (pipeline windowed_qubo), k ∈ {2, 3, 4} | pełny Taillard tam, gdzie pozwala kwota; inaczej n ≤ 100 | kwota D-Wave — sprawdzić stan konta |
| **Q3** | **QAOA na symulatorze** (bez szumu + fałszywy Sirius), k ∈ {2, 3, 4}, p = 1–3 | n = 20 (30 instancji), n = 50 (podzbiór) | doby CPU (jak E5/E5n) |
| **Q4** | **Sirius (sprzęt)**: tribonacci (k = 3) na tai20 inst. 0–3, p = 1–2 | 12 instancji | ~100–250 kr (do policzenia `--dry-run`) |

Ograniczenie kubitów w Q3/Q4: zmiennych w oknie ≈ (k−1)(w−1) ≤ 16 → okno w ≈ 17 (k = 2), 9 (k = 3), 6 (k = 4);
im większe k, tym mniejsze okna i więcej obwodów na ruch — to też wynik (koszt kwantowy rośnie z k).

**Pełny Taillard nie dla wszystkiego:** klasyka (C2) i analiza rozmiaru (Q1) — tak; D-Wave (Q2) — w miarę kwoty;
QAOA (Q3) — n = 20 (+ podzbiór n = 50); sprzęt (Q4) — tylko tai20 inst. 0–3.

## 4. Porównanie (rdzeń artykułu)

Jedna tabela i jedna figura spinające wszystko: **PRD w funkcji k** dla klasyki (DP), wyżarzacza i QAOA
(symulator / Sirius) na wspólnym podzbiorze instancji, przy wyrównanym budżecie lub tej samej liczbie ruchów;
do tego koszt (czas, kubity, CZ, długość łańcuchów) w funkcji k. Miary: PRD od dolnego ograniczenia Taillarda,
test Friedmana po k, testy par (Wilcoxon / znaków) z korektą Holma, krzywe zbieżności.

## 5. Plan artykułu (~8 stron + bibliografia)

| § | Treść |
|---|---|
| 1 Introduction | sąsiedztwa wykładnicze w PFSP, k jako pokrętło, wkład prozą |
| 2 Related Work | VLSN, Dynasearch, Fibonacci (promotor), Motzkin, kwantowe przeszukiwanie sąsiedztw |
| 3 The k-bonacci family | definicja, liczność (dowód), zawierania, DP O(nk), nieaddytywność |
| 4 Quantum formulation | QUBO dla N_k, pasmo, koszt QAOA i embeddingu w funkcji k |
| 5 Experiments | protokół C2, Q2, Q3, Q4 |
| 6 Results | PRD vs k (klasyka, wyżarzacz, QAOA), krajobraz, koszt kwantowy vs k |
| 7 Discussion / Conclusions | k*: klasyka vs kwanty; ograniczenia |

Zasady jak w ICAISC: wstęp prozą, pełne zdania, humanizer, bez nagłówka nad nagłówkiem, anonimizacja.

## 6. Harmonogram (do potwierdzenia po wyborze terminu)

| Okres | Co |
|---|---|
| po ICAISC (od 14.11) | C0, C1, Q1; rozmowa z promotorem |
| koniec XI | C2 (w tle), Q3 |
| początek XII | Q2, Q4 (kredyty), C3 |
| do 14.12 (ICAPS) lub 11.01 (IJCAI) | analiza, figury, tekst |

Kolizje: ICAISC 15.11, ANNSIM 18.12 (doby obliczeń), GCMM camera-ready (XI).

## 7. Ryzyka

| Ryzyko | Plan B |
|---|---|
| nakładanie z artykułem promotora | podział TWT/PFSP i tematy (teoria k, krajobraz, kwanty bramkowe) uzgodnione przed pisaniem |
| nieaddytywność delt Cmax | DP jako filtr + dokładna ocena; opisane jako wkład, nie ukryte |
| kwota D-Wave | Q2 na n ≤ 100 albo pominięty; kwanty przez Q3/Q4 |
| nie wygramy z IG | pytanie „czy N_k poprawia IG jako faza lokalna”, bez tezy o SOTA |
| tłok terminów | IJCAI 11.01 zamiast ICAPS |

## 8. Do decyzji
1. ICAPS (14.12) czy IJCAI (11.01)?
2. IG jako punkt odniesienia (N_k w fazie lokalnej IG)?
3. Zakres kwantowy: Q2 (D-Wave) — czy jest kwota; Q4 (Sirius) — ile kredytów.
4. Termin rozmowy z promotorem o podziale z jego artykułem.
