# k-bonacci: plan artykułu i badań, wersja 4 (14.11.2026)

Wersja 4 uaktualnia stan po kampanii głównej i etapie `large`.

Gałąź `kbonacci-icaps`, folder `~/Desktop/PhD-kbonacci/articles/kbonacci/`.
Zaakceptowany 11.11.2026. Zastępuje plan v2 z sekcją A, który leży w `PLAN_v2_archiwum.md`
(tam są szczegóły decyzji A.0–A.6, wyceny kwantowe i dziennik przeglądów).

## 1. Stan na 14.11.2026

| Element | Stan |
|---|---|
| Kod sąsiedztwa, DP, Insert, wersje D-Wave i IQM | gotowe, 163 testy przechodzą |
| Skrypty kampanii, ekstrakcji budżetów, analizy, kwantów | gotowe, przetestowane dymnie |
| **Kampania klasyczna (ILS)** | policzone: `validate` (PASS), `replicate` (skończony 11.11, zgodny ze starą kampanią: przy 10 s w granicach 0,01 pp), `main`, `insert`, `ablation`, `ablation_all`, `relative`, `large`. Od 14.11 ok. 11:50 trwają `large500` (tai500_20, 300 s, ok. 12,5 h), a potem `large_m10` (tai200_10, k ∈ {48, 64, n}, 60 s, ok. 2,5 h); koniec spodziewany w niedzielę 15.11 ok. 3:00, potem łańcuch sam wznawia kampanię crossings |
| Pilot (01.11) | zastąpiony kampanią główną (tabela pilota usunięta z artykułu) |
| Anatomia ruchu (11.11) | policzona dla n ≤ 100, ok. 990 stanów, `anatomy/anatomy.json`; n = 200 do zrobienia |
| Teoria | spisana z dowodami w `main.tex`, sprawdzona numerycznie (`scripts/kbonacci_theory_checks.py`) |
| Szkic | `main.tex` (EN) ma rozdział Results i roboczy Conclusions, napisane 14.11 na filtrze dokładnym; `wersja_pl.tex` (PL) do zsynchronizowania |
| Terminy (potwierdzone na stronach konferencji 10.11) | ICAPS (14.12.2026) nie jest już realny. IJCAI: abstrakt 4.01, artykuł 11.01.2027, 7 stron + 2 strony bibliografii, opłata 100 USD za zgłoszenie |

## 2. Co chcemy zbadać

Pytanie główne: **jak daleko powinna sięgać zamiana w ruchu złożonym dla PFSP?**
Protokół jest ten sam co w poprzednich pracach (ILS, start z permutacji identycznościowej, budżety 0,1–10 s,
ziarna 0–4, RPD do LB), więc jedynym czynnikiem jest sąsiedztwo.

| # | Pytanie | Skąd odpowiedź | Stan na 14.11 |
|---|---|---|---|
| Q1 | Jak jakość zależy od k przy stałym budżecie? Czy najlepsze k jest wewnętrzne (2 < k < n) i jak przesuwa się z n i z budżetem? | kampania `main` | najlepsze k jest wewnętrzne; przy 10 s i filtrze dokładnym wynosi 12 (n ≤ 100), 8 (n = 200), 6 (n = 500); zależy od m i od budżetu |
| Q2 | Ile iteracji mieści się w budżecie dla każdego k? | kampania `main` | n = 500, 10 s: ok. 3500 iteracji (k = 2), 1400 (k = 8), 370 (k = 16), 1 (k = n) |
| Q3 | Ile daje jeden ruch i ile kosztuje? Czy zysk na milisekundę przewiduje najlepsze k? | anatomia | zysk rośnie z k (100×10: 45 → 134), ale zysk/ms spada z ok. 145 (k ≤ 3) do ok. 1 (k = n); porównanie z kampanią do zrobienia |
| Q4 | Jak duży jest błąd addytywności i od czego zależy? | anatomia + twierdzenie 1, stwierdzenie 4 | zależy głównie od m: przy m = 20 ruch daje 33–57% obiecanego zysku, prognoza jest optymistyczna w 78–91% stanów; w kampanii najlepsze k maleje z m (m = 5: każde k ≥ 12 jednakowo; m = 10: 8–16 przy 10 s; m = 20 i n ≥ 100: 5–8 przy każdym budżecie) |
| Q5 | Jak daleko sięgają zamiany wybierane przy k = n? | anatomia | mediana 3–11 pozycji dla m ≥ 10, długi ogon (liczby z anatomii bez zmian; porównanie z kampanią do zrobienia) |
| Q6 | Jak N_k wypada wobec Adjacent, Fibonacciego, Dynasearch, Motzkina i Insert? | kampania + stare wyniki + `insert` | Insert policzony: lepszy na większości klas z n ≤ 200, gdy budżet sięga kilku sekund; k-bonacci prowadzi na 500×20 do 60 s (7 pp przy 10 s, 2,6 pp przy 60 s); wiersze Adjacent, Fibonacci, Dynasearch, Motzkin ze starej kampanii jeszcze nie weszły do tabel |
| Q7 | Ile kosztuje przybliżony filtr z poprzednich prac wobec filtra dokładnego? | ablacja `_cp` | filtr dokładny jest lepszy dla każdego k ≤ 16, każdego n i budżetu (0,1–3,7 pp przy 10 s, o 14–111% więcej iteracji); przy filtrze przybliżonym najlepsze k przy 10 s wynosiło 10–30, przy dokładnym 6–12; filtr przybliżony gubi 2–6% zamian poprawiających |
| Q8 (opcjonalne) | Czy QUBO k-bonacciego mieści się na D-Wave i IQM i jak k zmienia koszt? | poziomy MIN → OPT → MAX | — |
| Q9 | Czy przy n ≥ 200 najlepsze k rośnie z budżetem i czy Insert dogania rodzinę k przy 60 s? | etap `large` | 200×10: Insert wygrywa od 5 s i przy 60 s (1,28 wobec 1,42 dla k = 32), najlepsze k rośnie do 32; 200×20: najlepsze k prowadzi o 0,05 pp przy 60 s (k = 6 i 8 remis); 500×20: najlepsze k przechodzi z 6 na 8 między 20 a 30 s; 300 s otwarte (`large500`) |

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
| `relative` | 12 plików, k = ⌈α·n⌉, α = 10%…90% (bez wartości z `main`), 10 s | ~14 h |
| `ablation_all` | 12 plików, filtr dokładny dla pozostałych k z `main` (2, 4, 5, 6, 12, 16, n) | ~11,7 h |
| `large` | tai200_10, tai200_20, tai500_20 (30 instancji), filtr dokładny, k ∈ {4, 6, 8, 12, 16, 24, 32} + Insert, 60 s | ~20 h |
| `large500` | tai500_20, filtr dokładny, k ∈ {8, 12} + Insert, 300 s | ~12,5 h |
| `large_m10` | tai200_10, filtr dokładny, k ∈ {48, 64, n}, 60 s | ~2,5 h |

Etap `relative` jest policzony (po `ablation`, filtr przybliżony). Służy rysunkowi `fig_best_k_rel` (x = n, y = k/n, najlepsze k i remisy Wilcoxona).

Etap `large` ma rozstrzygnąć, czy małe k wygrywa przy dużych n z natury, czy tylko dlatego, że 10 s to krótki budżet.
Mniejsze budżety (0,1–30 s) odczytujemy z logu ruchów przebiegów 60 s. Ruszył 13.11 o 15:25 przed dokończeniem crossings,
które łańcuch sam wznawia po nim. Instancje n ≥ 1000 bierzemy dopiero wtedy, gdy trend będzie wyraźny.

Wynik `large` (14.11): o najlepszym k decyduje liczba maszyn. Przy m = 20 zostaje ono małe także przy 60 s
(k = 8 dla 200×20 i 500×20), a przy m = 10 rośnie z budżetem (200×10: k8 → k12 → k16 → k24 → k32, największe policzone).
Insert wygrywa na 200×10 od 5 s, remisuje na 200×20 przy 60 s i przegrywa na 500×20 przy 60 s o 2,6 pp, ale różnica maleje.
Stąd dwa etapy domykające, uruchomione 14.11 przed dokończeniem crossings (łańcuch sam je wznawia):
`large500` sprawdza, czy Insert dogania przy n = 500, a `large_m10` szuka dna krzywej przy m = 10.

Instrukcja: `RUN_LOCAL.md`. Mniejsze budżety z logu ruchów (`scripts/extract_budgets.py`). SA dopiero po decyzji.

### 4.2 Anatomia ruchu (nowy skrypt, `scripts/kbonacci_anatomy.py`)

Stany z zejść (start, dwa wczesne, optimum lokalne), wszystkie k na tych samych stanach, oba filtry.
Zrobione dla n ≤ 100 (kilkanaście minut na jednym rdzeniu). Do zrobienia: n = 200 z k ≤ 16
(`--files tai200_10,tai200_20 --ks 2,3,4,6,8,12,16`).

### 4.3 Analiza i rysunki

- `scripts/analyze_kbonacci.py`: tabele RPD (LB, UB-1993, BKS), iteracje, Wilcoxon + Holm, Friedman + Nemenyi, najlepsze k.
- Tabele etapu `large` są w `articles/kbonacci/tables/large/` (kopia z 14.11; odtworzenie: `analyze_kbonacci.py --dirs results/kbonacci/large_budgets --bks data/taillard_pfsp_bks.json --out articles/kbonacci/tables/large`). Wiersze trzech tabel w artykule powstały skryptem z plików CSV, nie ręcznie; skrypt trzeba jeszcze przenieść do `scripts/`.
- `scripts/plot_kbonacci.py` (nowy): RPD(k) dla każdego n i budżetu, iteracje(k), mapa najlepszego k, addytywność, odległości, koszt.

### 4.4 Kwanty

Decyzja 11.11: **D-Wave MIN + OPT** (≤ 54 s QPU łącznie, ok. 19% miesięcznej quoty), po klasyce. MIN w listopadzie,
OPT w grudniu po odnowieniu quoty. **IQM tylko MIN** (ok. 6 kredytów, z darmowej puli prywatnego konta), jeśli saldo pozwoli.
MAX poza artykułem, jako kierunek dalszej pracy. Skrypt: `scripts/run_quantum_tiers.py` (poziomy po kolei; MAX po prostu nie startuje).

| Poziom | Zakres | D-Wave | IQM Sirius |
|---|---|---|---|
| MIN | tai20_5, k ∈ {3, 4}, ILS + SA (IQM: instancja 0) | ≤ 18 s QPU, 200 przebiegów, ~0,4 h | ~6 kr (12 obwodów) |
| OPT (przyrost) | + k = 8 na tai20_5; + tai20_10, tai50_10, tai100_10 (ILS, k ∈ {3, 4}) | ≤ 36 s QPU, 400 przebiegów, ~0,8 h | — (poza zakresem) |

Quota D-Wave to **ok. 287 s/mies.** (projekt fdMX, korekta z lipca 2026), nie 400 s; domyślny `--budget-s` w skrypcie poprawiony.
Reguła z audytu czerwcowego („maks. 10% quoty na całe badania”, ok. 29 s) mieściła tylko MIN; OPT zatwierdzony przez użytkownika 11.11.
Aktualnego salda IQM ani zużycia D-Wave w listopadzie nie znamy (brak dostępu do dashboardów, licznik w repo widzi tylko ten kod),
więc przed każdym poziomem najpierw `--dry-run` i sprawdzenie salda przez użytkownika.

## 5. Struktura artykułu

| § | Sekcja | Stan w `main.tex` | Rysunki i tabele |
|---|---|---|---|
| 1 | Introduction | napisane, wkład prozą | — |
| 2 | Related Work | napisane; TODO: prace promotora | — |
| 3 | Problem and Notation | napisane (+ ścieżki kratowe) | — |
| 4 | The k-bonacci Neighborhood | napisane: definicja, lemat 1, stw. 1–3, DP | plansze, kafelki, łańcuch, udział |
| 5 | Additivity | napisane: tw. 1, stw. 4, przykłady | ścieżki |
| 6 | Experimental Protocol | napisane (+ anatomia) | — |
| 7 | Results | napisane na filtrze dokładnym (akapity Reach, Cost of reach, Machines, Long budgets, Filter, Insert, Move anatomy); do dodania: rysunki RPD(k) i iteracje(k) przeliczone dla filtra dokładnego (`plot_kbonacci.py` pomija teraz wiersze `_cp`), wiersze referencyjne, wyniki 300 s i k ∈ {48, 64, n}, testy na średnich po instancjach | trzy tabele (główna, najlepsze k per klasa, długie biegi), jeden rysunek (addytywność) |
| 8 | Conclusions | wersja robocza; do uaktualnienia po `large500` i `large_m10` | — |

**Decyzja 13.11:** tabela główna i wnioski liczone na filtrze dokładnym (`_cp`), stary filtr jako ablacja (jedna tabela i akapit
o tym, ile kosztował). Teza robocza: zasięg powinien być krótki i zależy od liczby maszyn; dokładny filtr poprawia każde k ≤ 16 i przesuwa najlepsze k w dół.

Szkic ma teraz 8 stron w układzie roboczym, z dowodami w tekście. Pod IJCAI (7 stron) część dowodów trafi do suplementu.
O cięciach decydujemy na końcu.

## 6. Harmonogram

| Kiedy | Co |
|---|---|
| do 16.11 | `large500` i `large_m10` kończą się 15.11; analiza obu, tabele `large*` przeniesione z katalogu roboczego do `articles/kbonacci/tables` |
| 17–22.11 | rysunki z filtrem dokładnym, wiersze referencyjne, testy na średnich po instancjach (120 par) i między k, Friedman z Nemenyi CD, anatomia n = 200, rozmowa z promotorem; **decyzja 22.11: IJCAI czy GECCO** |
| 23.11–3.12 | rewizja BPASTS ma pierwszeństwo (termin 3.12) |
| od 1.12 | D-Wave OPT po odnowieniu quoty |
| IJCAI | abstrakt 4.01, artykuł 11.01; czas na SA albo kwanty |

ICAPS (14.12) odpada; IJCAI albo GECCO, decyzja 22.11.

## 7. Ryzyka

| Ryzyko | Ocena | Co z nim robimy |
|---|---|---|
| „To tylko Dynasearch z limitem odległości” | wysokie | teoria (stw. 2–4, tw. 1), studium k, anatomia |
| Start z permutacji identycznościowej, stałe budżety, RPD do LB | wysokie na A* | decyzja wiążąca zostaje; RPD do BKS jako druga miara; w tekście jawnie: badamy sąsiedztwa przy protokole z poprzednich prac. Opcja: dodatkowy etap ze startem NEH tylko dla rodziny k (ok. +16 h) |
| Główna kampania używa filtra przybliżonego, a twierdzenia mówią o dokładnym | rozwiązane 13.11 | `ablation_all` policzył filtr dokładny dla wszystkich k; wyniki główne na `_cp` (decyzja w §5) |
| Nakładanie się ze szkicem promotora (tribonacci/k-nacci, TWT, D-Wave) | średnie | rozmowa przed pisaniem dalszych części; zdanie „we introduce” do uzgodnienia |
| Testy Wilcoxona na parach (instancja, ziarno) | średnie | artykuł podaje obecnie test na parach (instancja, ziarno) z zastrzeżeniem, że jest optymistyczny; test na średnich po ziarnach (120 par) jest wymagany przed wysłaniem |
| Przy n = 500 i 10 s żaden wariant nie jest zbieżny (k = n robi 1 iterację, Insert 11), więc „najlepsze k” może być skutkiem budżetu | niskie po etapie `large` | 60 s potwierdza małe k dla m = 20 (k = 8); dla m = 10 optimum rośnie z budżetem, więc tezę formułujemy per m; 300 s otwarte (`large500`) |
| Brak SA w pierwszej wersji | niskie | decyzja po ILS |
| Zmieszanie n i m w benchmarku (brak 200×5, n = 500 tylko z m = 20) | średnie | twierdzenia formułujemy per klasa, tabela najlepszego k per klasa |
| Insert wygrywa z rodziną na większości klas z n ≤ 200 | wysokie | artykuł mówi to otwarcie; rodzina k jest lepsza dla dużych instancji i krótkich budżetów; ostateczne sformułowanie po `large500` |
| Kolizje oznaczeń znalezione w przeglądzie (macierze S_L a ruchy złożone S, S_1, S_2; macierz ogonowa T a T_0; Δ i α użyte podwójnie) | średnie | zmienić nazwy przed wysłaniem; wymaga przebudowy `fig_paths` |
| Bezpiecznik (lepszy wg rzeczywistego C_max z ruchu złożonego i najlepszej pojedynczej zamiany) | niskie | decyzja 11.11: poza kampanią główną, bo zmienia regułę wyboru ruchu, a nie otoczenie, i psuje porównywalność ze starymi wynikami. Ewentualnie osobny, oznaczony wariant `kbonacci_kn_safe` (ok. +2,3 h), do rozważenia po wynikach; to samo ustalenie przekazane do wątku o krzyżowaniach. Offline: licznik w anatomii, jak często ruch złożony jest gorszy od najlepszej pojedynczej zamiany |
| Brak porównania ze stanem wiedzy PFSP (iterated greedy Ruiza i Stützlego) | wysokie na A* | w tekście wprost: badamy otoczenie przy stałym protokole, nie bijemy rekordów; IG cytowany jako stan wiedzy, bez doliczania |
| Część kwantowa mała, wygląda na doklejoną | średnie | jeden akapit w wynikach: małe k daje małe QUBO (MIN + OPT D-Wave, MIN IQM) |
| Domknięcie łańcucha „od góry” | średnie | tylko Motzkin (zagnieżdżanie) ze starych wyników i z wcześniejszych prac. **Nie** dokładamy wyników z pilota crossings, bo zdradzają główny wynik artykułu o inwolucjach (decyzja 11.11). Krzyżowanie: jedno zdanie w zakończeniu, bez liczb |

## 8. Sprawy otwarte

1. **Promotor:** podział z jego szkicem k-nacci, cytowanie, współautorstwo; czy Sci Rep 2026 i IFAC 2025 zawierają już hierarchię; lista BKS.
2. **Filtr blokowy:** opis w poprzednich pracach („nie może poprawić”) jest nieścisły; decyzja z promotorem.
3. **Bibliografia:** trzy pozycje własne to wciąż placeholdery; Miles 1960, Schilling 1990, Ergun i Orlin 2006, Grosso i in. 2004 do sprawdzenia w Crossref.
   Balzereit i in. poprawione (tytuł i autorzy z arXiv).
4. **Stare wyniki:** kompletność (Motzkin n = 500, Dynasearch n = 500).
5. **Tabela pilota:** zrobione, pilot usunięty z artykułu.
6. **Miejsce publikacji:** kryterium oparte na (log₂n+2)/n odrzucają dane: najlepsze k wynosi 12, 12, 8, 6 dla n = 50…500 i zależy od m;
   N_6 pokrywa poniżej 2% Dynasearch przy n = 500 i mimo to wygrywa. Decyzja IJCAI czy GECCO zależy teraz od `large500`
   (czy Insert dogania przy 300 s) i od siły zależności od m. Tezy „zysk Dynasearch siedzi w k ≤ 3–4” nie formułujemy.
7. **`nemenyi_cd` jest NaN** w `tables/friedman.csv`: poprawić w `analyze_kbonacci.py`.
8. **Etap `relative`** policzono tylko ze starym filtrem: albo powtórzyć trzy ułamki z filtrem dokładnym (koszt do oszacowania), albo zostawić zastrzeżenie w tekście.
9. **Szkic dowodu Stw. 2:** składnik błędu musi być jednostajny względem k (w artykule oznaczone TODO).
10. **Licznik bezpiecznika w anatomii** (jak często ruch złożony jest gorszy od najlepszej pojedynczej zamiany): zrobić z istniejących stanów albo porzucić.
11. **RPD do UB-1993 i do BKS** są w tabelach, ale artykuł jeszcze ich nie używa; trzeba podać źródło BKS.

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
