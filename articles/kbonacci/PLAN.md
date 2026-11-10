# k-bonacci: plan artykułu i badań, wersja 3 (11.11.2026)

Gałąź `kbonacci-icaps`, folder `~/Desktop/PhD-kbonacci/articles/kbonacci/`.
Zaakceptowany 11.11.2026. Zastępuje plan v2 z sekcją A, który leży w `PLAN_v2_archiwum.md`
(tam są szczegóły decyzji A.0–A.6, wyceny kwantowe i dziennik przeglądów).

## 1. Stan na 11.11.2026

| Element | Stan |
|---|---|
| Kod sąsiedztwa, DP, Insert, wersje D-Wave i IQM | gotowe, 163 testy przechodzą |
| Skrypty kampanii, ekstrakcji budżetów, analizy, kwantów | gotowe, przetestowane dymnie |
| **Kampania klasyczna (ILS)** | **nieuruchomiona** (brak `results/kbonacci/`) |
| Pilot (01.11) | 3 klasy, 1 s, mała próba |
| Anatomia ruchu (nowe, 11.11) | policzona dla n ≤ 100, ok. 990 stanów, `anatomy/anatomy.json` |
| Teoria | spisana z dowodami w `main.tex`, sprawdzona numerycznie (`scripts/kbonacci_theory_checks.py`) |
| Szkic | `main.tex` (EN, v2), `wersja_pl.tex` (PL, v2) |
| Terminy (potwierdzone na stronach konferencji 10.11) | ICAPS: abstrakt 7.12, artykuł 14.12.2026. IJCAI: abstrakt 4.01, artykuł 11.01.2027, 7 stron + 2 strony bibliografii, opłata 100 USD za zgłoszenie |

## 2. Co chcemy zbadać

Pytanie główne: **jak daleko powinna sięgać zamiana w ruchu złożonym dla PFSP?**
Protokół jest ten sam co w poprzednich pracach (ILS, start z permutacji identycznościowej, budżety 0,1–10 s,
ziarna 0–4, RPD do LB), więc jedynym czynnikiem jest sąsiedztwo.

| # | Pytanie | Skąd odpowiedź | Wstępnie (pilot, anatomia) |
|---|---|---|---|
| Q1 | Jak jakość zależy od k przy stałym budżecie? Czy najlepsze k jest wewnętrzne (2 < k < n) i jak przesuwa się z n i z budżetem? | kampania `main` | pilot: przy n ≤ 50 i 1 s wygrywa k = n |
| Q2 | Ile iteracji mieści się w budżecie dla każdego k? | kampania `main` | pilot: 3329 → 63 iteracji (50×10, k = 2 → n) |
| Q3 | Ile daje jeden ruch i ile kosztuje? Czy zysk na milisekundę przewiduje najlepsze k? | anatomia | zysk rośnie z k (100×10: 45 → 134), ale zysk/ms spada z ok. 145 (k ≤ 3) do ok. 1 (k = n) |
| Q4 | Jak duży jest błąd addytywności i od czego zależy? | anatomia + twierdzenie 1, stwierdzenie 4 | zależy głównie od m: przy m = 20 ruch daje 33–57% obiecanego zysku, prognoza jest optymistyczna w 78–91% stanów |
| Q5 | Jak daleko sięgają zamiany wybierane przy k = n? | anatomia | mediana 3–11 pozycji dla m ≥ 10, długi ogon |
| Q6 | Jak N_k wypada wobec Adjacent, Fibonacciego, Dynasearch, Motzkina i Insert? | kampania + stare wyniki + `insert` | — |
| Q7 | Ile kosztuje przybliżony filtr z poprzednich prac wobec filtra dokładnego? | ablacja `_cp` | filtr przybliżony gubi 2–7% zamian poprawiających |
| Q8 (opcjonalne) | Czy QUBO k-bonacciego mieści się na D-Wave i IQM i jak k zmienia koszt? | poziomy MIN → OPT → MAX | — |

**Zmiana względem v2:** hipoteza H2 („korzyść z k nasyca się w skali długości bloku”) nie znajduje potwierdzenia
w anatomii: mediana k_sat dla n = 100 wynosi n. Proponuję zastąpić ją pytaniem Q3 (zysk na jednostkę czasu), które dane wspierają.

## 3. Teoria (wszystko w `main.tex`, rozdziały 4–5)

| Wynik | Treść | Dowód | Kontrola numeryczna |
|---|---|---|---|
| Lemat 1 | ruch N_k = macierz blokowo-diagonalna z blokami S_L, L ≤ k; rozkład jednoznaczny | pełny | C1 |
| Stw. 1 | \|N_k\| = a_n (rekurencja k-bonacciego), funkcja tworząca, 2 − λ_k = λ_k^(−k), **2 − 2^(1−k) ≤ λ_k < 2 − 2^(−k)** | pełny (nowy dowód ograniczeń) | C1, C2 |
| Stw. 2 (nowe) | a_n ≈ c_k·λ_k^n ze stałą c_k w postaci zamkniętej; **udział w Dynasearch ≈ exp(−n/2^(k+1))** | szkic (biegun dominujący) | C2, C3 |
| Stw. 3 | filtr ścieżki krytycznej jest bezstratny; ≤ (m−1)(k−1)(k+2)/2 kandydatów; ocena O(mn + m²k³) | pełny | C4 (38 853 zamian poprawiających, 0 zgubionych) |
| DP pozycyjne | B[p] = min(B[p−1], min_d B[p−d−1] + δ), O(nk) | z lematu 1 | testy jednostkowe |
| Tw. 1 (nowe) | C_max po ruchu = max po ścieżkach z sumy zmian; DP liczy sumę maksimów; równość, gdy jedna ścieżka jest krytyczna we wszystkich konfiguracjach | pełny | C5 (9 440 ruchów złożonych) |
| Stw. 4 (nowe) | rozcięcie ruchu w kolumnie: wzór head–tail, ograniczenia dwustronne, **warunek, przy którym suma delt jest optymistyczna**, warunek równości | pełny | C6 |
| Przykłady 1–2 (nowe) | najmniejsze instancje (n = 4), w których suma delt zawyża i zaniża | wyliczenie | C7 |

Uruchomienie kontroli: `python scripts/kbonacci_theory_checks.py` (ok. 5 s, wynik `ALL CHECKS PASSED`).

Uwaga: dla m = 2 i czasów ≤ 5 nie istnieje instancja n = 4, w której dwie poprawiające zamiany sąsiednie dają sumę optymistyczną.
To może być ogólna własność m = 2. Nie twierdzimy tego w artykule; ewentualnie do sprawdzenia później.

## 4. Eksperymenty

### 4.1 Kampania klasyczna (bez zmian względem sekcji A.0 starego planu)

| Etap | Zakres | Czas (Mac, ILS) |
|---|---|---|
| `validate` | tai20_5 + tai50_10, k ∈ {3, 8}, 6 budżetów niezależnie + powtórki | ~1,6 h |
| `replicate` | stare Fibonacci i Dynasearch, 2 klasy, 10 s | ~0,6 h |
| `main` | 12 plików, k ∈ {2, 3, 4, 5, 6, 8, 12, 16, n}, 10 s | ~15,5 h |
| `insert` | 12 plików, Insert | ~1,7 h |
| `ablation` | 12 plików, k ∈ {3, 8} z filtrem dokładnym | ~3,3 h |

Instrukcja: `RUN_LOCAL.md`. Mniejsze budżety z logu ruchów (`scripts/extract_budgets.py`). SA dopiero po decyzji.

### 4.2 Anatomia ruchu (nowy skrypt, `scripts/kbonacci_anatomy.py`)

Stany z zejść (start, dwa wczesne, optimum lokalne), wszystkie k na tych samych stanach, oba filtry.
Zrobione dla n ≤ 100 (kilkanaście minut na jednym rdzeniu). Do zrobienia: n = 200 z k ≤ 16
(`--files tai200_10,tai200_20 --ks 2,3,4,6,8,12,16`).

### 4.3 Analiza i rysunki

- `scripts/analyze_kbonacci.py`: tabele RPD (LB, UB-1993, BKS), iteracje, Wilcoxon + Holm, Friedman + Nemenyi, najlepsze k.
- `scripts/plot_kbonacci.py` (nowy): RPD(k) dla każdego n i budżetu, iteracje(k), mapa najlepszego k, addytywność, odległości, koszt.

### 4.4 Kwanty

Bez zmian: poziomy MIN → OPT → MAX po klasyce (`scripts/run_quantum_tiers.py`), D-Wave ≤ 225 s QPU, IQM ok. 200 kredytów.

## 5. Struktura artykułu

| § | Sekcja | Stan w `main.tex` | Rysunki i tabele |
|---|---|---|---|
| 1 | Introduction | napisane, wkład prozą | — |
| 2 | Related Work | napisane; TODO: prace promotora | — |
| 3 | Problem and Notation | napisane (+ ścieżki kratowe) | — |
| 4 | The k-bonacci Neighborhood | napisane: definicja, lemat 1, stw. 1–3, DP | plansze, kafelki, łańcuch, udział |
| 5 | Additivity | napisane: tw. 1, stw. 4, przykłady | ścieżki |
| 6 | Experimental Protocol | napisane (+ anatomia) | — |
| 7 | Results | anatomia wstępnie; kampania: szkielet | addytywność; po kampanii RPD(k), iteracje, najlepsze k, tabela główna |
| 8 | Conclusions | TODO po kampanii | — |

Szkic ma teraz 8 stron w układzie roboczym, z dowodami w tekście. Pod IJCAI (7 stron) część dowodów trafi do suplementu.
O cięciach decydujemy na końcu.

## 6. Harmonogram

| Kiedy | Co |
|---|---|
| do 16.11 | komplet GCMM (pierwszeństwo). W tle na Macu: `validate` + `replicate` (2,2 h), potem `main` + `insert` + `ablation` (20,5 h) |
| 17–22.11 | analiza, rysunki, anatomia n = 200; rozmowa z promotorem (punkt 8); **decyzja 22.11: ICAPS czy IJCAI** |
| 23.11–3.12 | rewizja BPASTS ma pierwszeństwo (termin 3.12); równolegle uzupełnianie rozdziału Results |
| ICAPS | abstrakt 7.12, artykuł 14.12 (koliduje z ANNSIM 18.12) |
| IJCAI | abstrakt 4.01, artykuł 11.01; czas na SA albo kwanty |

Moja ocena: przy GCMM (16.11), BPASTS (3.12) i ANNSIM (18.12) realny jest IJCAI. ICAPS tylko wtedy, gdy kampania
ruszy w tym tygodniu, a promotor szybko odpowie w sprawie podziału.

## 7. Ryzyka

| Ryzyko | Ocena | Co z nim robimy |
|---|---|---|
| „To tylko Dynasearch z limitem odległości” | wysokie | teoria (stw. 2–4, tw. 1), studium k, anatomia |
| Start z permutacji identycznościowej, stałe budżety, RPD do LB | wysokie na A* | decyzja wiążąca zostaje; RPD do BKS jako druga miara; w tekście jawnie: badamy sąsiedztwa przy protokole z poprzednich prac. Opcja: dodatkowy etap ze startem NEH tylko dla rodziny k (ok. +16 h) |
| Główna kampania używa filtra przybliżonego, a twierdzenia mówią o dokładnym | średnie | opisane wprost w rozdziale 4; ablacja `_cp`. Opcja: ablacja dla wszystkich k zamiast dwóch (ok. +12 h) |
| Nakładanie się ze szkicem promotora (tribonacci/k-nacci, TWT, D-Wave) | średnie | rozmowa przed pisaniem dalszych części; zdanie „we introduce” do uzgodnienia |
| Testy Wilcoxona na parach (instancja, ziarno) | średnie | ziarna tej samej instancji nie są niezależne; proponuję dodać test na średnich po ziarnach (120 par) obok dotychczasowego |
| Brak SA w pierwszej wersji | niskie | decyzja po ILS |

## 8. Sprawy otwarte

1. **Promotor:** podział z jego szkicem k-nacci, cytowanie, współautorstwo; czy Sci Rep 2026 i IFAC 2025 zawierają już hierarchię; lista BKS.
2. **Filtr blokowy:** opis w poprzednich pracach („nie może poprawić”) jest nieścisły; decyzja z promotorem.
3. **Bibliografia:** trzy pozycje własne to wciąż placeholdery; Miles 1960, Schilling 1990, Ergun i Orlin 2006, Grosso i in. 2004 do sprawdzenia w Crossref.
   Balzereit i in. poprawione (tytuł i autorzy z arXiv).
4. **Stare wyniki:** kompletność (Motzkin n = 500, Dynasearch n = 500).
5. **Tabela 2 (pilot)** używa RPD do UB, a tabela główna RPD do LB; pilot zniknie po kampanii.

## 9. Zmiany szkicu v1 → v2 (zaakceptowane 11.11.2026)

| # | Miejsce | Było | Jest |
|---|---|---|---|
| 1 | Introduction, wkład | lista `enumerate` z czterema punktami | ciągła proza; dodane: wzór exp(−n/2^(k+1)), bezstratny filtr, opis tego, co optymalizuje DP |
| 2 | Related Work | Balzereit: „radius–runtime trade-off … parameterized-complexity viewpoint” | poprawny opis pracy (szeregowanie z rodzinami przezbrojeń, promień przeszukiwania); TODO o równoległej pracy grupy promotora |
| 3 | Problem and Notation | tylko Head/Tail | + akapit o ścieżkach kratowych (potrzebny w rozdziale 5) |
| 4 | Stw. 1 | „λ_k = 2 − Θ(2^(−k))”, szkic dowodu | jawne ograniczenia z pełnym dowodem |
| 5 | Hierarchia | tekst | + rysunek łańcucha (`fig_chain`) |
| 6 | Pokrycie Dynasearch | odwołanie do „classical longest-run law” | + stwierdzenie 2 ze wzorem i szkicem dowodu; przedział 0,78–0,88 wyprowadzony |
| 7 | Ocena, filtr | jedno zdanie + TODO | akapit „Filtering”: stwierdzenie 3 z dowodem; filtr poprzednich prac nazwany heurystyką (gubi 2–6% zamian poprawiających na optimach lokalnych) |
| 8 | Additivity | jeden akapit z TODO | twierdzenie 1, stwierdzenie 4, rysunek, dwa przykłady |
| 9 | Protocol | — | + akapit „Move anatomy” |
| 10 | Results | pilot + „Planned analyses” | + akapit z wynikami anatomii i rysunek; analizy: test na średnich po instancjach, Friedman |
| 11 | Bibliografia | Balzereit z TODO | pełny wpis; + Miles 1960, Schilling 1990 (do weryfikacji) |

Zmiany są naniesione w `main.tex` i w `wersja_pl.tex`.

## 10. Pliki dodane 11.11.2026

| Plik | Co to |
|---|---|
| `articles/kbonacci/PLAN.md` | ten plan (stary: `PLAN_v2_archiwum.md`) |
| `articles/kbonacci/main.tex`, `wersja_pl.tex` | szkic v2 (PDF-y w `build/`) |
| `articles/kbonacci/figures/fig_chain.tex`, `fig_paths.tex` | rysunki TikZ: łańcuch zawierań, ścieżki i rozcięcie |
| `articles/kbonacci/figures/fig_additivity.pdf`, `fig_distances.pdf`, `fig_cost.pdf` | rysunki z anatomii |
| `articles/kbonacci/anatomy/anatomy.json`, `anatomy_summary.txt` | dane i tabela anatomii |
| `scripts/kbonacci_theory_checks.py` | kontrola numeryczna wszystkich twierdzeń |
| `scripts/kbonacci_anatomy.py` | anatomia ruchu na stanach przeszukiwania |
| `scripts/plot_kbonacci.py` | rysunki z tabel kampanii i z anatomii |
