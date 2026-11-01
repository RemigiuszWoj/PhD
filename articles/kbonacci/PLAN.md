# Artykuł: sąsiedztwa tribonacci / k-bonacci ze studium parametru k

Stan: 01.11.2026. Plan + research przed implementacją. Kontekst: `articles/katalog_sasiedztw.md`, `ROADMAP.md` (front C).

## 0. Status (01.11.2026, gałąź `kbonacci-icaps`)

- ✅ Moduł `src/neighborhoods/classical/kbonacci.py`: delty Head/Tail, filtr blokowy, pozycyjne DP O(n·k), statystyki addytywności.
  Tryby w `get_neighbor`: `tribonacci`, `tetranacci`, `kbonacci_k<K>`, `kbonacci_kn`.
- ✅ Testy `tests/test_kbonacci.py` (96 przypadków): liczność = brute force (n ≤ 9), DP = brute force,
  k = 2 ≡ DP Fibonacciego, k = n ≡ Dynasearch, delty dokładne. Cały zestaw: 121/121.
- ✅ Dane Taillard PFSP (12 plików × 10 instancji) pobrane do `data/` (gitignored) i **zweryfikowane generatorem Taillarda**
  (0 rozbieżności). Źródło: aosabook/500lines. UB pochodzą z 1993 roku, więc do artykułu trzeba aktualnych best-known UB.
- ✅ Pilot `scripts/run_kbonacci_pilot.py`, wyniki w `articles/kbonacci/pilot/`.
- ✅ Szkice `main.tex` (EN) i `wersja_pl.tex` (PL). Nie były kompilowane, bo w sesji nie ma LaTeX-a.
- ❌ Prace promotora (Sci Rep 2026, IFAC 2025): nie udało się ich przeczytać, bo doi.org i nature.com są zablokowane w sieci sesji,
  a wyszukiwarka nie znalazła IFAC. **Potrzebne PDF-y od promotora.**

### Wyniki pilota (ILS, 1000 ms, 4 inst. × 3 ziarna)

| | tai20_5 | tai20_10 | tai50_10 |
|---|---|---|---|
| Adjacent | 1,59 | 3,26 | 11,73 |
| k = 2 (Fib) | 1,31 | 2,46 | 8,37 |
| k = 3 (trib) | 0,92 | 1,84 | 5,94 |
| k = 4 | 0,89 | 1,43 | 5,51 |
| k = 6 | 0,74 | 1,09 | 4,85 |
| k = 8 | 0,72 | 1,35 | 4,66 |
| k = n | **0,55** | **0,99** | **3,62** |
| Dynasearch (stary, rekurencyjny) | 0,94 | 1,49 | 5,30 |

(RPD [%] do UB.)
- **Przy 1000 ms i n ≤ 50 wygrywa k = n.** Wewnętrzne k\* trzeba szukać przy n ≥ 100 i krótszych budżetach (Faza 1).
  Jeśli go nie będzie, wkładem jest krzywa kompromisu i szybkie DP, a nie „magiczne k”.
- **Nowe DP robi z Dynasearch 2–3× więcej iteracji** (63 vs 23 przy n = 50) i daje lepsze RPD (3,62 vs 5,30).
  Uwaga: stary `dynasearch_full` przy n = 50 przekracza domyślny limit rekurencji Pythona (main.py podnosi go do 2·10⁶).
- **Addytywność:** predykcja jest dokładna w 42% ocen dla k = 2 i w 10–12% dla k ≥ 6. Zawyżenie poprawy rośnie od 10% (k = 2) do ~50% (k ≥ 8).
- **Liczba zamian w ruchu (2–5) prawie nie zależy od k.** Rośnie natomiast średnia odległość zamiany (1 → ~4,5 przy k = 8).

## 1. Wybór konferencji

| | ICAPS 2027 (Columbia SC, 27.06–2.07.2027) | IJCAI 2027 (Kioto 7–13.08.2027) |
|---|---|---|
| Termin (NIEPOTWIERDZONY, trackery) | abstrakt ~7.12.2026, artykuł ~14.12.2026 | abstrakt ~4.01.2027, artykuł ~11–15.01.2027 |
| Czasu od dziś | ~6 tygodni | ~10 tygodni |
| Dopasowanie tematu | **bardzo dobre** (scheduling, local search) | dobre, ale recenzent „szerokie AI” może uznać PFSP za wąskie → potrzebny wątek AI (adaptacyjne k) |
| Format (z poprzednich edycji, do sprawdzenia w CFP) | AAAI style, 8 stron + referencje, double-blind | IJCAI style, 7 stron + 2 str. referencji, double-blind |

Oficjalne strony (`icaps27.icaps-conference.org`, `ijcai.org`) są zablokowane w sieci tej sesji, więc daty trzeba sprawdzić ręcznie.
**Nie wolno wysłać tego samego tekstu na obie naraz**: notyfikacja ICAPS (~koniec lutego) przychodzi po terminie IJCAI.

**Rekomendacja:** celujemy w **ICAPS**, zgodnie z katalogiem. Decyzja zapada w **punkcie kontrolnym ~22.11**. Jeśli pilot (Faza 1) da czytelną krzywą
jakość(k) i k-bonacci jest konkurencyjny z Insert/IG, piszemy na ICAPS. Jeśli nie, przesuwamy się na IJCAI i
zyskujemy ~4 tygodnie na pełny Taillard, porównanie z IG i wątek adaptacyjnego k.

## 2. Co już jest w repo (kluczowe znalezisko)

- `src/neighborhoods/classical/dynasearch.py` ma już parametr **`L_max`** (długość przedziału j−i+1 ≤ L_max).
  Wariant (a) k-bonacciego, czyli zamiana końców przedziału o długości ≤ k ze środkiem zablokowanym, to **dokładnie Dynasearch z L_max = k**.
  k = 2 daje Fibonacciego, k = n daje Dynasearch. Prototyp jest więc prawie za darmo.
- Mamy Head/Tail (`compute_endpoint_swap_delta`), akcelerator blokowy NPI, ILS (BackTrackJumpList) i SA z reheatem,
  runner eksperymentów, Wilcoxona (`tools/wilcoxon_analysis.py`) i dane porównawcze dla Adj/Fib/Dyn/Motz.
- **Brakuje danych Taillard PFSP**: `data/` jest w `.gitignore`, a w klonie jest tylko `qaoa_angles.json`. Trzeba je pobrać
  (np. `tai20_5.txt` … `tai500_20.txt`) razem z **górnymi ograniczeniami (best-known UB)**.
- DP w `dynasearch_full` przy L_max = k skanuje kandydatów liniowo (`while next_pos`). Dla k-bonacciego potrzebne jest DP pozycyjne O(n·k).

## 3. Definicja rodziny (USTALONA 01.11.2026)

**Ruch elementarny = zamiana końców (i, j), odległość d = j − i ∈ {1, …, k−1}, środek przedziału stoi.**
Ruch złożony to zbiór takich zamian na rozłącznych przedziałach [i, j].

| Sąsiedztwo | dopuszczalne odległości zamian | k | liczność (n pozycji) | wzrost |
|---|---|---|---|---|
| Fibonacci | 1 (sąsiednie) | 2 | F(n+1) | 1,618 |
| tribonacci | 1, 2 | 3 | T(n) (n = 10: 274) | 1,839 |
| tetranacci | 1, 2, 3 | 4 | (n = 10: 401) | 1,928 |
| k-bonacci | 1, …, k−1 | k | a_n = a_{n−1} + … + a_{n−k} | → 2 |
| Dynasearch | 1, …, n−1 | n | 2^{n−1} | 2 |

Liczność: permutację dzielimy na kafelki długości 1 (pozycja bez ruchu) albo L = d + 1 ∈ {2, …, k} (zamiana końców kafelka),
co daje dokładnie rekurencję k-bonacciego. Implementacyjnie to Dynasearch z `L_max = k`. Wkładem jest więc **rodzina sparametryzowana k
i studium k**, a nie pojedynczy nowy ruch: szybkie DP O(n·k) zamiast O(n³), koszt delt O(m·n·k²) zamiast O(m·n³),
ciągły suwak między Fibonaccim a Dynasearchem. To trzeba wyraźnie zaznaczyć w tekście, bo recenzent zauważy związek z Dynasearchem.

### Możliwe rozszerzenie (nie w tym artykule, ewentualnie future work)

Inne jądra ruchu w kafelku dają ważoną rekurencję a_n = Σ w_L a_{n−L}:
a w każdym bloku stosujemy jeden ruch z **jądra** W_L. Liczność to ważona rekurencja k-bonacciego
a_n = Σ_{L=1..k} w_L · a_{n−L}, gdzie w_L = |W_L|. Ruchy w bloku są nierozkładalne, więc różne podziały dają różne permutacje.
Sprawdziłem to policzeniem (n = 10, zgodne z katalogiem: 274 dla tribonacciego).

| Jądro W_L (L ≥ 3) | w_L | n = 10, k = 3 | wzrost k = 3 | wzrost k → ∞ | interpretacja |
|---|---|---|---|---|---|
| (a) zamiana końców, środek stoi | 1 | 274 | 1,8393 | 2 | Fib → Dynasearch (oś łuków) |
| (b) obie rotacje (= wstawienie o L−1 pozycji w lewo/prawo) | 2 | 585 | **2,0000** | ~2,20 | Fib → **złożone wstawienia** |
| (c) zamiana końców + obie rotacje | 3 | 1046 | 2,1304 | ~2,36 | (a) ∪ (b) |

Wzrost dla k = 4, 5, 6, 8: (a) 1,928 / 1,966 / 1,984 / 1,996; (b) 2,123 / 2,171 / 2,190 / 2,203.

Rotacja kafelka to ograniczone wstawienie (Insert), czyli ruch najsilniejszy dla PFSP, więc (b) to naturalny kolejny artykuł
(„złożone krótkie wstawienia”). Moduł warto napisać z parametrem jądra, ale w tym artykule używamy tylko (a).

## 4. Pytania badawcze (studium k)

- **RQ1. Jakość vs k:** czy PRD maleje monotonicznie z k, czy się nasyca? Czy istnieje k\*, i czy zależy od n, m i budżetu czasu?
- **RQ2. Koszt vs k:** koszt jednej oceny to delty O(m·n·k) z Head/Tail (dla zamiany odległej o L koszt ~m·L) plus DP O(n·k).
  Mierzymy czas na iterację i liczbę iteracji w budżecie. Kompromis „bogactwo ruchu vs liczba iteracji” jest sednem artykułu.
- **RQ3. Addytywność:** DP sumuje delty niezależnych ruchów, a w PFSP Cmax **nie jest addytywny** (problem widoczny już w ACIIDS).
  Mierzymy błąd |Σδ − ΔCmax_rzecz| w funkcji k. To uczciwy wynik, który recenzent A* doceni.
- **RQ4. Struktura ruchu:** rozkład liczby elementarnych ruchów w wybranym ruchu złożonym oraz rozkład wybranych długości L.
  Pokazuje, czy duże k jest w ogóle używane.
- **RQ5 (wątek AI, głównie dla IJCAI):** adaptacyjne k online (bandyta UCB/ALNS nad k ∈ {2, 3, 4, 6, 8}) vs najlepsze stałe k.
  Daje odpowiedź na pytanie „jak dobrać k bez strojenia”.
- **RQ6 (opcjonalnie, krótka sekcja):** QUBO k-bonacciego jest pasmowe (szerokość ~k−1, ~(k−1)n zmiennych).
  Pokazujemy tylko szerokość pasma i długość łańcuchów embeddingu w funkcji k, bez QPU, jako pomost do wcześniejszych prac.

## 5. Plan eksperymentów

**Faza 0: implementacja i walidacja (do ~12.11)**
1. Pobrać Taillard PFSP i best-known UB (format jak `src/parser.py`).
2. Moduł `src/neighborhoods/classical/kbonacci.py`: delty Head/Tail dla zamian końców o odległości ≤ k−1, DP pozycyjne O(n·k) (opcjonalnie top-k jak w Fibonaccim),
   akcelerator blokowy (warunek ∩ granic bloków z [i, j]). Wpięcie w `CLASSICAL_MODES` i runner (`kbonacci_k3_rot` itp.).
3. Testy: liczność = rekurencja (brute force n ≤ 10); DP = brute force najlepszego ruchu na małych n; k = 2 ≡ Fibonacci;
   wariant (a) z k = n ≡ Dynasearch.
4. Wydajność: czysty Python może być wąskim gardłem. Rozważyć numpy/numba dla delt, ale **jednakowo dla wszystkich sąsiedztw**
   (uczciwe porównanie przy budżecie czasu).

**Faza 1: pilot (do ~22.11, punkt kontrolny)**
- tai20_5, tai20_10, tai20_20, tai50_10, tai50_20, tai100_10; 4 instancje × 5 ziaren; ILS (+ SA); budżety 100, 1000, 5000 ms.
- k ∈ {2, 3, 4, 5, 6, 8, 12, n}; punkty odniesienia: Adjacent, Insert (pojedynczy, z akceleracją Taillarda), Dynasearch, Motzkin.
- Wynik: krzywe PRD(k), czas/iteracja(k), błąd addytywności(k). Na tej podstawie wybór 3–4 wartości k i decyzja ICAPS/IJCAI.

**Faza 2: pełny Taillard (ICAPS: ~23.11–3.12; IJCAI: do ~20.12)**
- 12 plików × 10 instancji (n = 20…500), 5 ziaren, budżety jak w `windowed_qubo` (porównywalność z poprzednimi pracami).
- **RPD względem best-known UB** (A* tego oczekuje). Dodatkowo RPD do LB dla zgodności z naszymi wcześniejszymi tabelami.
- Silny baseline: **Iterated Greedy (Ruiz & Stützle 2007)** z lokalnym przeszukiwaniem Insert oraz IG, w którym LS = k-bonacci.
  Bez IG recenzent A* najpewniej odrzuci pracę jako „słabe baseline’y”.
- Statystyka: Wilcoxon + Friedman z post-hoc Nemenyi / Holm i wykres critical-difference.

**Faza 3: tylko IJCAI**: adaptacyjne k (RQ5) + ewentualnie sekcja QUBO (RQ6).

**Zasoby:** kampania w pełni klasyczna, więc bez quoty D-Wave. Pełny Taillard to ~120 instancji × 5 ziaren × kilka budżetów × ~10 wariantów,
czyli rzędu dni CPU. Liczyć na M4 Pro lub w kilku sesjach chmurowych równolegle.

## 6. Szkic artykułu (8 stron)

1. **Wstęp:** sąsiedztwa złożone / VLSN, luka między Fibonaccim (ubogi, O(n)) a Dynasearch/Motzkinem (bogate, O(n²–n³)).
   k-bonacci jako ciągły suwak między nimi.
2. **Pokrewne prace:** Dynasearch (Congram/Potts/van de Velde 2002; Ergun & Orlin 2006; Grosso i in. 2004), VLSN (Ahuja i in. 2002),
   multimoves (Bożejko & Wodecki 2007), własności blokowe (Nowicki & Smutnicki 1996, Grabowski & Wodecki 2004),
   PFSP IG (Ruiz & Stützle 2007), parametryzowane lokalne przeszukiwanie (Balzereit i in., arXiv 2409.00771),
   prace promotora: Sci Rep 2026 (wykładnicze sąsiedztwo Fibonacciego), IFAC 2025.
   **Przed submisją sprawdzić, że nie zawierają już tej hierarchii** (sprawa otwarta z katalogu).
3. **Rodzina k-bonacci:** definicja z jądrem, twierdzenie o liczności (ważona rekurencja, stała wzrostu jako pierwiastek
   1 = Σ w_L x^{−L}), hierarchia Adj ⊂ Fib ⊂ trib ⊂ … ⊂ Dyn, poprawność DP, złożoność O(m·n·k + n·k), akcelerator blokowy.
4. **Studium k:** RQ1–RQ4 (+ RQ5).
5. **Wyniki na pełnym Taillardzie** i porównanie z IG/Insert.
6. **Wnioski:** rekomendacja k\* oraz ograniczenia (addytywność).

Double-blind: własne prace (Fibonacci, Motzkin, windowed QUBO) cytujemy w trzeciej osobie. Repo musi być anonimowe
(anonymous.4open.science), bo repo na GitHubie jest publiczne i ma imię i nazwisko.

## 7. Harmonogram (wariant ICAPS)

| Do | Zadanie |
|---|---|
| 5.11 | dane Taillard + UB; sprawdzenie CFP; przegląd prac promotora (Sci Rep 2026, IFAC 2025) |
| 12.11 | moduł kbonacci + testy (Faza 0) |
| 22.11 | pilot + **punkt kontrolny ICAPS vs IJCAI** |
| 3.12 | pełny Taillard + IG |
| 7.12 | abstrakt w systemie |
| 14.12 | artykuł (pisanie równolegle od ~25.11) |

## 8. Sprawy otwarte

1. ~~Jądro (a)/(b)/(c)~~: ustalone, (a) zamiany końców o odległości ≤ k−1.
2. Prace promotora (Sci Rep 2026, IFAC 2025): czy pokrywają tę hierarchię?
3. Daty i limity stron z oficjalnych CFP.
4. Implementacja IG: czy jest gdzieś w historii repo, czy piszemy od zera (~1–2 dni)?
5. Kolizja tematyczna z ICAISC 2027 (Hamming z zamian, 15.11) i ANNSIM (18.12): kto ma czas na co.
6. Uwaga z `windowed_qubo`: Fibonacci w SA wypada słabo (15% vs Adjacent 11% przy n = 20). Warto sprawdzić, czy przyczyną nie jest
   brak ruchu wymuszonego / top-k, zanim zbudujemy na nim rodzinę.
