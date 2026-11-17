# k-bonacci: plan artykułu i badań, wersja 4 (14.11.2026)

Wersja 4 uaktualnia stan po kampanii głównej i etapie `large`.

Gałąź `kbonacci-icaps`, folder `~/Desktop/PhD-kbonacci/articles/kbonacci/`.
Wersja 3 zaakceptowana 11.11.2026; wersja 4 (14.11) czeka na akceptację. Zastępuje plan v2 z sekcją A, który leży w `PLAN_v2_archiwum.md`
(tam są szczegóły decyzji A.0–A.6, wyceny kwantowe i dziennik przeglądów).

## 1. Stan na 14.11.2026

| Element | Stan |
|---|---|
| Kod sąsiedztwa, DP, Insert, wersje D-Wave i IQM | gotowe, 163 testy przechodzą |
| Skrypty kampanii, ekstrakcji budżetów, analizy, kwantów | gotowe, przetestowane dymnie |
| **Kampania klasyczna (ILS)** | policzone: `validate` (PASS), `replicate` (skończony 11.11, zgodny ze starą kampanią: przy 10 s w granicach 0,01 pkt proc.), `main`, `insert`, `ablation`, `ablation_all`, `relative`, `large`. `large500` i `large_m10` skończone 15.11 (tabele w `tables/large500`, `tables/large_m10`). Od 15.11 12:20 trwa `relative_cp` (ok. 14 h, koniec ok. 16.11 2:30); łańcuch sam wznawia potem crossings. |
| Pilot (01.11) | zastąpiony kampanią główną (tabela pilota usunięta z artykułu) |
| Anatomia ruchu (11.11, 16.11) | policzona dla n ≤ 100 (ok. 990 stanów, `anatomy/anatomy.json`) i dla n = 200, k ≤ 16 (194 stany, `anatomy/anatomy_n200.json`, `anatomy_summary_n200.txt`; 20 s obliczeń) |
| Teoria | spisana z dowodami w `main.tex`, sprawdzona numerycznie (`scripts/kbonacci_theory_checks.py`) |
| Szkic | `main.tex` (EN) ma rozdział Results i roboczy Conclusions, napisane 14.11 na filtrze dokładnym; `wersja_pl.tex` (PL) zsynchronizowana 14.11 i uzupełniona o braki wobec wersji angielskiej |
| Terminy (potwierdzone na stronach konferencji 10.11) | ICAPS (14.12.2026) nie jest już realny. IJCAI: abstrakt 4.01, artykuł 11.01.2027, 7 stron + 2 strony bibliografii, opłata 100 USD za zgłoszenie |

## 2. Co chcemy zbadać

Pytanie główne: **jak daleko powinna sięgać zamiana w ruchu złożonym dla PFSP?**
Protokół jest ten sam co w poprzednich pracach (ILS, start z permutacji identycznościowej, budżety 0,1–10 s,
ziarna 0–4, RPD do LB), więc jedynym czynnikiem jest sąsiedztwo.

| # | Pytanie | Skąd odpowiedź | Stan na 14.11 |
|---|---|---|---|
| Q1 | Jak jakość zależy od k przy stałym budżecie? Czy najlepsze k jest wewnętrzne (2 < k < n) i jak przesuwa się z n i z budżetem? | kampania `main` | najlepsze k jest wewnętrzne; przy 10 s i filtrze dokładnym wynosi 12 (n ≤ 100), 8 (n = 200), 6 (n = 500); zależy od m i od budżetu |
| Q2 | Ile iteracji mieści się w budżecie dla każdego k? | kampania `main` | n = 500, 10 s: ok. 3500 iteracji (k = 2), 1400 (k = 8), 370 (k = 16), 1 (k = n) |
| Q3 | Ile daje jeden ruch i ile kosztuje? Czy zysk na milisekundę przewiduje najlepsze k? | anatomia | zysk rośnie z k (100×10: 45 → 134), ale zysk/ms spada z ok. 145 (k ≤ 3) do ok. 1 (k = n); porównanie przy równej liczbie iteracji zrobione z istniejących tabel (np. 500×20 po ok. 690 iteracjach: RPD 7,6 dla k = 4, 6,7 dla k = 6, 6,4 dla k = 8; długie biegi po 4000 iteracjach: 5,2 dla k = 4 i 3,3 dla k = 12); zysk na milisekundę z anatomii NIE przewiduje najlepszego k (jest najwyższy przy k = 2, które w ILS jest najgorsze) |
| Q4 | Jak duży jest błąd addytywności i od czego zależy? | anatomia + twierdzenie 1, stwierdzenie 4 | zależy głównie od m: przy m = 20 ruch daje 33–57% obiecanego zysku, prognoza jest optymistyczna w 78–91% stanów; w kampanii najlepsze k maleje z m (m = 5: każde k ≥ 12 jednakowo; m = 10: 8–16 przy 10 s; m = 20 i n ≥ 100: 5–8 przy każdym budżecie) |
| Q5 | Jak daleko sięgają zamiany wybierane przy k = n? | anatomia | mediana 3–11 pozycji dla m ≥ 10, długi ogon (liczby z anatomii bez zmian; porównanie z kampanią do zrobienia) |
| Q6 | Jak N_k wypada wobec Adjacent, Fibonacciego, Dynasearch, Motzkina i Insert? | kampania + stare wyniki + `insert` | Insert policzony: lepszy na większości klas z n ≤ 200, gdy budżet sięga kilku sekund; k-bonacci prowadzi na 500×20 do 60 s (7,4 pkt proc. przy 10 s, 2,6 pkt proc. przy 60 s); wiersze Adjacent, Fibonacci, Dynasearch, Motzkin ze starej kampanii jeszcze nie weszły do tabel |
| Q7 | Ile kosztuje filtr granic bloków z poprzednich prac wobec filtra dokładnego? | ablacja `_cp` | filtr dokładny jest lepszy dla każdego k ≤ 16, każdego n i budżetu (0,1–3,7 pkt proc. przy 10 s, o 14–111% więcej iteracji); przy filtrze granic bloków najlepsze k przy 10 s wynosiło 10–30, przy dokładnym 6–12; filtr granic bloków gubi 2–6% zamian poprawiających |
| Q8 (opcjonalne) | Czy QUBO k-bonacciego mieści się na D-Wave i IQM i jak k zmienia koszt? | poziomy MIN → OPT → MAX | — |
| Q9 | Czy przy n ≥ 200 najlepsze k rośnie z budżetem i czy Insert dogania rodzinę k przy 60 s? | etap `large` | 200×10: Insert wygrywa od 5 s i przy 60 s (1,28 wobec 1,42 dla k = 32), najlepsze k rośnie do 32; 200×20: najlepsze k prowadzi o 0,05 pkt proc. przy 60 s (k = 6 i 8 remis); 500×20: najlepsze k przechodzi z 6 na 8 między 20 a 30 s; 300 s otwarte (`large500`) **Wynik 15.11:** 500×20 przy 300 s: k = 8 2,02, k = 12 2,10, Insert 2,26; Insert odrabia większość różnicy (0,9 pkt proc. przy 120 s, 0,2 przy 300 s), ale nie dogania; k = 8 wyprzedza k = 12 przy każdym budżecie. 200×10 przy 60 s: k = 32, 48, 64 w granicach 0,05 pkt proc. (1,42 / 1,41 / 1,38), k = n 1,48, Insert 1,28. |

**Zmiana względem v2:** hipoteza H2 („korzyść z k nasyca się w skali długości bloku”) nie znajduje potwierdzenia
w anatomii: mediana k_sat dla n = 100 wynosi n. Proponuję zastąpić ją pytaniem Q3 (zysk na jednostkę czasu); odpowiedź jest negatywna (zysk na milisekundę nie przewiduje najlepszego k), a wyjaśnia to porównanie przy równej liczbie iteracji.

## 3. Teoria (wszystko w `main.tex`, rozdziały 4–5)

| Wynik | Treść | Dowód | Kontrola numeryczna |
|---|---|---|---|
| Lemat 1 | ruch N_k = macierz blokowo-diagonalna z blokami S_L, L ≤ k; rozkład jednoznaczny | pełny | C1 |
| Stw. 1 | \|N_k\| = a_n (rekurencja k-bonacciego), funkcja tworząca, 2 − λ_k = λ_k^(−k), **2 − 2^(1−k) ≤ λ_k < 2 − 2^(−k)** | pełny (nowy dowód ograniczeń) | C1, C2 |
| Stw. 2 | a_n ≈ c_k·λ_k^n ze stałą c_k w postaci zamkniętej; **udział w Dynasearch ≈ exp(−n/2^(k+1))** | szkic (biegun dominujący) | C2, C3 |
| Stw. 3 | filtr ścieżki krytycznej (klasyczna własność blokowa) jest dokładny dla sumy delt; ≤ (m−1)(k−1)(k+2)/2 kandydatów; ocena O(mn + min(mnk², m²k³)) | pełny | C4 (38 853 zamian poprawiających, 0 zgubionych) |
| DP pozycyjne | B[p] = min(B[p−1], min_d B[p−d−1] + δ), O(nk) | z lematu 1 | testy jednostkowe |
| Tw. 1 (nowe) | C_max po ruchu = max po ścieżkach z sumy zmian; DP liczy sumę maksimów; równość, gdy jedna ścieżka jest krytyczna we wszystkich konfiguracjach | pełny | C5 (9 440 ruchów złożonych) |
| Stw. 4 (nowe) | rozcięcie ruchu w kolumnie: wzór head–tail, ograniczenia dwustronne, **warunek, przy którym suma delt jest optymistyczna**, warunek równości | pełny | C6 |
| Przykłady 1–2 (nowe) | najmniejsze instancje (n = 4), w których suma delt zawyża i zaniża | wyliczenie | C7 |

Uruchomienie kontroli: `python scripts/kbonacci_theory_checks.py` (ok. 5 s, wynik `ALL CHECKS PASSED`).

Uwaga: dla m = 2 i czasów ≤ 5 nie istnieje instancja n = 4, w której dwie poprawiające zamiany sąsiednie dają sumę optymistyczną.
To może być ogólna własność m = 2. Nie twierdzimy tego w artykule; ewentualnie do sprawdzenia później.

## 4. Eksperymenty

### 4.1 Kampania klasyczna (etapy do `ablation` jak w sekcji A.0 starego planu, dalsze dodane 12–14.11)

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
| `relative_cp` | 12 plików, k = ⌈α·n⌉ jak w `relative`, filtr dokładny, 10 s | ~14 h |

Etap `relative` jest policzony (po `ablation`, filtr granic bloków). Służy rysunkowi `fig_best_k_rel` (x = n, y = k/n, najlepsze k i remisy Wilcoxona).

Etap `large` ma rozstrzygnąć, czy małe k wygrywa przy dużych n z natury, czy tylko dlatego, że 10 s to krótki budżet.
Mniejsze budżety (0,1–30 s) odczytujemy z logu ruchów przebiegów 60 s. Ruszył 13.11 o 15:25 przed dokończeniem crossings,
które łańcuch sam wznawia po nim. Instancje n ≥ 1000 bierzemy dopiero wtedy, gdy trend będzie wyraźny.

Wynik `large` (14.11): o najlepszym k decyduje liczba maszyn. Przy m = 20 zostaje ono małe także przy 60 s
(k = 8 dla 200×20 i 500×20), a przy m = 10 rośnie z budżetem (200×10: k8 → k12 → k16 → k24 → k32, największe policzone).
Insert wygrywa na 200×10 od 5 s, na 200×20 przy 60 s przegrywa o 0,05 pkt proc. (w granicy remisu), a na 500×20 przy 60 s przegrywa o 2,6 pkt proc., ale różnica maleje.
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

**Decyzja 13.11:** tabela główna i wnioski liczone na filtrze dokładnym (`_cp`), filtr granic bloków jako ablacja (jeden akapit
o tym, ile kosztował). Teza robocza (decyzja 14.11): głównym układem porównań jest budżet czasu, tak jak w poprzednich pracach. Przy ustalonym budżecie najlepsze k jest małe, maleje z liczbą maszyn i rośnie z budżetem; udział ruchów Dynasearch nie przewiduje jakości; filtr dokładny poprawia każde k ≤ 16 dla każdego rozmiaru instancji. Porównanie przy równej liczbie iteracji zostaje jako dodatek (osobny akapit w wynikach): dłuższy zasięg jest wtedy lepszy w każdej klasie, więc zasięg ogranicza koszt ruchu, a nie błąd addytywności.

Szkic ma teraz 8 stron w układzie roboczym, z dowodami w tekście. Pod IJCAI (7 stron) część dowodów trafi do suplementu.
O cięciach decydujemy na końcu.

## 6. Harmonogram

| Kiedy | Co |
|---|---|
| do 16.11 | `large500` i `large_m10` kończą się 15.11; analiza obu; przenieść skrypt generujący wiersze tabel do `scripts/`; rozmowa z promotorem (punkty 8.1 i 8.2) |
| 17–22.11 | rysunki z filtrem dokładnym, wiersze referencyjne, testy na średnich po instancjach (120 par) i między k, Friedman z Nemenyi CD; anatomia n = 200 i każdy bieg z zegarem (SA, D-Wave MIN) dopiero po końcu kampanii crossings; **decyzja 22.11: IJCAI czy GECCO**; decyzja o SA 22.11 |
| 23.11–3.12 | rewizja BPASTS ma pierwszeństwo (termin 3.12) |
| 4–18.12 | ANNSIM ma pierwszeństwo (termin 18.12); k-bonacci tylko D-Wave OPT |
| 19.12–3.01 | cięcie do 7 stron, suplement z dowodami, szablon IJCAI, anonimizacja, bibliografia własna; SA (ok. 16 h) tylko w oknie 19–23.12, inaczej wiersze SA znikają z protokołu; abstrakt 4.01 |
| 5–11.01 | tylko korekta, żadnych nowych obliczeń; artykuł 11.01 |
| od 1.12 | D-Wave OPT po odnowieniu quoty |

ICAPS (14.12) odpada; IJCAI albo GECCO, decyzja 22.11.

## 7. Ryzyka

| Ryzyko | Ocena | Co z nim robimy |
|---|---|---|
| „To tylko Dynasearch z limitem odległości” | wysokie | teoria (stw. 2–4, tw. 1), studium k, anatomia |
| Start z permutacji identycznościowej, stałe budżety, RPD do LB | wysokie na A* | decyzja wiążąca zostaje; RPD do BKS jako druga miara; w tekście jawnie: badamy sąsiedztwa przy protokole z poprzednich prac. Opcja: dodatkowy etap ze startem NEH tylko dla rodziny k (ok. +16 h) |
| Główna kampania używa filtra granic bloków, a twierdzenia mówią o dokładnym | rozwiązane 13.11 | `ablation_all` policzył filtr dokładny dla wszystkich k; wyniki główne na `_cp` (decyzja w §5) |
| Nakładanie się ze szkicem promotora (tribonacci/k-nacci, TWT, D-Wave) | średnie | rozmowa przed pisaniem dalszych części; zdanie „we introduce” do uzgodnienia |
| Testy Wilcoxona na parach (instancja, ziarno) | średnie | artykuł podaje obecnie test na parach (instancja, ziarno) z zastrzeżeniem, że jest optymistyczny; test na średnich po ziarnach (120 par) jest wymagany przed wysłaniem |
| Przy n = 500 i 10 s żaden wariant nie jest zbieżny (k = n robi 1 iterację, Insert 11), więc „najlepsze k” może być skutkiem budżetu | niskie po etapie `large` | 60 s potwierdza małe k dla m = 20 (k = 8); dla m = 10 optimum rośnie z budżetem, więc tezę formułujemy per m; 300 s otwarte (`large500`) |
| Brak SA w pierwszej wersji | niskie | decyzja po ILS |
| Nowość teorii: licznik k-bonacciego, ograniczenia λ_k, prawo najdłuższej serii i własność blokowa są znane | średnie | w tekście przedstawione jako znane; wkładem jest parametryzacja przez k i badanie; do uzgodnienia z promotorem, czy „Theorem 1” zostaje twierdzeniem, czy obserwacją |
| Filtr dokładny jest dokładny tylko dla sumy delt, nie dla rzeczywistego C_max ruchu złożonego, a ruch awaryjny (najlepsza pojedyncza zamiana) liczy się po pozostałych kandydatach | niskie | opisane w tekście (akapit po stwierdzeniu o filtrze) |
| Porównanie z Insert zależy od szybkości implementacji (Insert 0,87 s na iterację dla 500×20) | średnie | podać język implementacji i czas jednej iteracji |
| Komórki, w których jedna iteracja przekracza budżet (Insert przy n = 500 poniżej 1 s, k = n przy n = 500) | niskie | opisane w protokole (podana wartość pierwszej iteracji) |
| Siatki k różnią się między filtrami (filtr dokładny nie ma wartości między 16 a n w kampanii głównej), więc zdanie „filtr obniża najlepsze k” dotyczy tylko policzonych siatek | niskie | sformułowanie w tekście ograniczone do policzonych siatek |
| Zmieszanie n i m w benchmarku (brak 200×5, n = 500 tylko z m = 20) | średnie | twierdzenia formułujemy per klasa, tabela najlepszego k per klasa |
| Insert wygrywa z rodziną na większości klas z n ≤ 200 | wysokie | artykuł mówi to otwarcie; rodzina k jest lepsza dla dużych instancji i krótkich budżetów; ostateczne sformułowanie po `large500` |
| Kolizje oznaczeń znalezione w przeglądzie (macierze S_L a ruchy złożone S, S_1, S_2; macierz ogonowa T a T_0; Δ i α użyte podwójnie) | średnie | zmienić nazwy przed wysłaniem; wymaga przebudowy `fig_paths` |
| Bezpiecznik (lepszy wg rzeczywistego C_max z ruchu złożonego i najlepszej pojedynczej zamiany) | niskie | decyzja 11.11: poza kampanią główną, bo zmienia regułę wyboru ruchu, a nie otoczenie, i psuje porównywalność ze starymi wynikami. Ewentualnie osobny, oznaczony wariant `kbonacci_kn_safe` (ok. +2,3 h), do rozważenia po wynikach; to samo ustalenie przekazane do wątku o krzyżowaniach. Offline: licznik w anatomii, jak często ruch złożony jest gorszy od najlepszej pojedynczej zamiany |
| Brak porównania ze stanem wiedzy PFSP (iterated greedy Ruiza i Stützlego) | wysokie na A* | w tekście wprost: badamy otoczenie przy stałym protokole, nie bijemy rekordów; IG cytowany jako stan wiedzy, bez doliczania |
| Część kwantowa mała, wygląda na doklejoną | średnie | jeden akapit w wynikach: małe k daje małe QUBO (MIN + OPT D-Wave, MIN IQM) |
| Domknięcie łańcucha „od góry” | średnie | tylko Motzkin (zagnieżdżanie) ze starych wyników i z wcześniejszych prac. **Nie** dokładamy wyników z pilota crossings, bo zdradzają główny wynik artykułu o inwolucjach (decyzja 11.11). Krzyżowanie: jedno zdanie w zakończeniu, bez liczb |

## 8. Sprawy otwarte

1. **Promotor:** podział z jego szkicem k-nacci, cytowanie, współautorstwo; czy Sci Rep 2026 i IFAC 2025 zawierają już hierarchię; lista BKS.
2. **Filtr granic bloków:** opis w poprzednich pracach („nie może poprawić”) jest nieścisły; decyzja z promotorem.
3. **Bibliografia:** trzy pozycje własne to wciąż placeholdery; Miles 1960, Schilling 1990, Ergun i Orlin 2006, Grosso i in. 2004 do sprawdzenia w Crossref.
   Balzereit i in. poprawione (tytuł i autorzy z arXiv).
4. **Stare wyniki:** kompletność (Motzkin n = 500, Dynasearch n = 500).
5. **Miejsce publikacji:** kryterium oparte na (log₂n+2)/n odrzucają dane: najlepsze k wynosi 12, 12, 8, 6 dla n = 50…500 i zależy od m;
   N_6 pokrywa poniżej 2% Dynasearch przy n = 500 i mimo to wygrywa. Termin GECCO 2027 do sprawdzenia na stronie konferencji przed 22.11. Decyzja IJCAI czy GECCO zależy teraz od `large500`
   (czy Insert dogania przy 300 s) i od siły zależności od m. Tezy „zysk Dynasearch siedzi w k ≤ 3–4” nie formułujemy.
6. **`nemenyi_cd` jest NaN** w `tables/friedman.csv`: poprawić w `analyze_kbonacci.py`. **Zrobione 16.11:** `analyze_kbonacci.py` liczy q Nemenyi z rozkładu rozstępu studentyzowanego, testy Wilcoxona na średnich po instancjach (`wilcoxon_inst.csv`), testy najlepszego k z pozostałymi (`wilcoxon_k.csv`), wiersze odniesienia ze starej kampanii (`--reference ~/Desktop/PhD/results/experiments/202605*`; tylko majowe katalogi, bo lipcowe to eksperyment z dywersyfikacją; przebiegi > 1,5 × budżet odpadają, komórka tylko z kompletem). Wynik: na średnich po instancjach Insert vs najlepsze k przy 10 s istotny tylko dla n = 100 (p = 0,005); Friedman 120 instancji: Insert i k ∈ {6, 8, 12, 16} (filtr dokładny) w jednej grupie (CD 2,86). Stary Fibonacci = k = 2 z filtrem granic bloków co do 0,1 pkt proc. Stare Dynasearch i Motzkin mają przekroczone budżety dla n ≥ 100 (stary Dynasearch 200: 14,83 vs k = n 9,98) – decyzja użytkownika 16.11: stare wyniki bierzemy takie, jakie są, bez liczenia od nowa. Do tabeli weszły wszystkie cztery wiersze w całości. Przebiegów przekraczających budżet NIE odrzucamy: obie kampanie sprawdzają limit między iteracjami, więc przebieg kończy rozpoczęty ruch (konwencja „z dokładnością do jednej iteracji”, jak w poprzednich artykułach). Kreska tylko tam, gdzie stara kampania nie liczyła (Dynasearch i Motzkin, n = 500, 0,1 s i 1 s).
7. **Etap `relative`** policzono tylko z filtrem granic bloków: albo powtórzyć trzy ułamki z filtrem dokładnym (koszt do oszacowania), albo zostawić zastrzeżenie w tekście.
8. **Szkic dowodu Stw. 2:** składnik błędu musi być jednostajny względem k (w artykule oznaczone TODO).
9. **Licznik bezpiecznika w anatomii** (jak często ruch złożony jest gorszy od najlepszej pojedynczej zamiany): zrobić z istniejących stanów albo porzucić.
10. **RPD do UB-1993 i do BKS** są w tabelach, ale artykuł jeszcze ich nie używa; trzeba podać źródło BKS.
11. **Obiecane w planie, a nieobecne w artykule ani wśród TODO:** RPD do UB-1993 w tabeli, `fig_best_k_rel` i mapa najlepszego k, zdanie o tym, że nie bijemy rekordów (IG jako stan wiedzy), zmiana kolidujących oznaczeń. Każde dopisać jako TODO w artykule albo skreślić.

12. **Test Wilcoxona między wartościami k:** dziś jest tylko względem Insert.
13. **Lista tabu dla ruchu złożonego:** co przechowuje; dopisać w protokole.
14. **Pięć ziaren przy n = 500:** czy dają różne przebiegi przed pierwszym zaburzeniem (sprawdzić w logach).
15. **Cytowania zamiast szkicu dowodu stwierdzenia o udziale:** Feller oraz Gordon–Schilling–Waterman.
16. **Etap uzupełniający dla pozostałych k (filtr dokładny, 10 s, decyzja użytkownika).** Czas wynika z budżetu: jedno k na jednej klasie to 50 przebiegów × 10 s ≈ 8,5 min, jedno k na wszystkich klasach z n ≥ 50 (9 klas) ≈ 1,25 h, na 12 klasach ≈ 1,7 h; narzut uruchamiania ok. 5–10%. Warianty: (A) minimum, k ∈ {20, 24, 32} dla n = 50 i 100 oraz k = 20 dla 200×10, 200×20 i 500×20: 1050 przebiegów, ok. 3 h; (B) siatka k ∈ {10, 20, 24, 32} dla wszystkich klas z n ≥ 50 (bez par już policzonych): ok. 4,5 h; (C) pełne lustro etapu `relative` z filtrem dokładnym (k = 10–90% n): ok. 14 h. Dla n = 20 wartości k ≥ 20 pokrywają się z k = n. **Decyzja 15.11: wariant C.** Etap `relative_cp` (4350 przebiegów, ok. 14 h) ruszył 15.11 o 12:20 (commit `46d47fe4`), koniec ok. 16.11 o 2:30; łańcuch `results/kbonacci/chain_relcp.sh` sam wznawia potem crossings (wstrzymane na 700/1200 etapu ablation). Etapy `large500` i `large_m10` skończone 15.11, do analizy.
    **Wynik (16.11.2026):** etap `relative_cp` policzony (4350 przebiegów; 56 zepsutych uśpieniem komputera w nocy 15/16.11 powtórzono, kontrola `sleepcheck.py` czysta), tabele w `tables/relative_cp/`, wpisany do Results EN/PL. Tezy stoją: dla m = 20 i n ≥ 100 RPD nie maleje od k = 10 do k = n; 200×10 najlepsze k = 20 (płasko 16–40); m = 5 płasko dla k ≥ 12; 500×20 od k = 250 jeden ruch w 10 s. Zmienione: kolumna 10 s w tabeli najlepszego k, Insert lepszy w 7 z 11 klas (50×10 przechodzi na k = 15, 20×10 różnica < 0,01 pkt proc.), najlepsze k przy filtrze dokładnym 6–15 na tej samej siatce co filtr granic bloków (10–30).
17a. **Stan 16.11 wieczór:** bibliografia zweryfikowana w Crossref (Ergun–Orlin, Grosso i in., Miles, Schilling; Balzereit i in. nadal tylko arXiv), własne prace wpisane (ICAISC 2026 w druku, SOCO 2026 z DOI, preprint SSRN); RPD do BKS w Results; opis listy tabu i języka w protokole. Zostało 14 `\todo`: decyzje (SA, część kwantowa, podział z promotorem, łańcuch w pracach grupy), cytowania Fellera i Gordona–Schillinga–Watermana, liczby z `check_block_filter.py`, źródło BKS (do potwierdzenia z promotorem), rysunek RPD od liczby iteracji, fig_distances/fig_cost.
17. **Najtańsze wzmocnienie tezy bez nowych długich obliczeń:** rysunek RPD względem liczby iteracji (oś logarytmiczna, krzywa dla każdego k, np. 100×5, 100×10, 100×20 i 500×20) z istniejących logów ruchów oraz mała tabela czasu ruchu w ms dla k i m. Do zrobienia po końcu biegów.

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

## 9a. Zmiany szkicu v2 → v3 (14.11.2026)

| Miejsce | Było | Jest |
|---|---|---|
| Streszczenie | TODO o wyniku | główny wynik: małe k, zależność od liczby maszyn, filtr dokładny |
| Protokół | filtr granic bloków jako główny, ablacja dla k ∈ {3, 8} | oba filtry dla każdego k, filtr dokładny główny, biegi 60 s, liczby walidacji |
| Wyniki | tabela pilotażowa, szkielet, „Planned analyses” | trzy tabele i akapity Reach, Cost of reach, Machines, Long budgets, Filter, Insert, Move anatomy |
| Wnioski | TODO | wersja robocza |
| Teoria | — | kontrprzykład (3, 4, 1, 2), wyjaśnienie optymizmu sumy delt, zdanie, że udział ruchów Dynasearch nie przewiduje jakości |
| Wersja polska | braki wobec angielskiej | uzupełnione braki wobec angielskiej |
| po przeglądach 14.11 (wieczór) | teza „zasięg powinien być krótki”; filtr „obniża najlepsze k”; założenie stwierdzenia o rozcięciu ruchu z trzema maksimami na jednej maszynie; rekurencja DP bez górnej granicy p−1; koszt O(mn + m²k³) | teza przeformułowana na „zasięg na miarę budżetu”, porównanie przy równej liczbie iteracji, osłabione założenie w stwierdzeniu o rozcięciu ruchu, poprawiony zakres w rekurencji DP, koszt O(mn + min(mnk², m²k³)), opis tego, czego filtr nie zachowuje |

Zmiany przeszły pięć rund przeglądu 14.11 i Rundy do popołudnia są w `ebe886e6`; zmiany wieczorne trafiają do następnego commita.

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
