# Plan badań i artykułu: przejście z flow shopu (PFSP) na job shop (J||C_max) — rodzina otoczeń liczonych ciągami na ścieżce krytycznej

Wersja 3, 11.11.2026 (po czwartej rundzie przeglądu, A.7; wersja 2 z 01.11.2026 miała trzy rundy, A.6). Gałąź `jobshop-plan-v2`, worktree `~/Desktop/PhD-jobshop`.
Format jak w `articles/kbonacci/PLAN.md`. **Stan i poprawki z rundy 4 są w A.7; tam, gdzie A.7 różni się od reszty dokumentu, obowiązuje A.7.**
Powiązane:
- `articles/kbonacci/PLAN.md`: oś k, czyli zasięg zamiany;
- `articles/krzyzowania/PLAN.md`: oś c (Motzkin → ≤ c krzyżowań → inwolucje); w przygotowaniu w sesji „involution”, jeszcze nie wypchnięty;
- `articles/katalog_sasiedztw.md`;
- `ROADMAP.md`, front B.

Prototyp i pilot z tej sesji: `scripts/jobshop_proto.py`, wyniki w `articles/jobshop/pilot/`.

Terminologia: „otoczenie” (Smutnicki) i „sąsiedztwo” (dotychczasowe teksty w repo) to to samo. W planie używam „sąsiedztwo”, jak w katalogu.

> **Streszczenie dla promotora (1 minuta)**
> - **Co:** przeniesienie naszej rodziny sąsiedztw z PFSP na job shop. Ścieżka krytyczna pełni rolę permutacji.
>   Proponowana konstrukcja to **rodzina ruchów granicznych** (typ × zasięg × złożenie, §2.4). Zawiera N5, N6, N7, Fib, k-bonacci i Dyn jako punkty jednej przestrzeni.
> - **Teoria:** T2 to dokładny warunek wykonalności dowolnego przestawienia wewnątrz bloków. Wnioski:
>   - rodziny bez krzyżowań nie wymagają testów kombinacji;
>   - krzyżowania dają ograniczenia dwuciałowe;
>   - flow shop jest wolny od ograniczeń;
>   - pod filtrem rodziny zamian spłaszczają się do ≤ 2 ruchów na blok (T10).
> - **Pilot (chmura, Python):**
>   - T2 bez wyjątku na ~184 tys. przypadków;
>   - ILS 1 s: ruchy złożone (k = 3, c6, c7, bm3) 7,4–7,5% RPD vs N5 10,6% i N6 8,4%;
>   - w dobrych rozwiązaniach 45–67% dalekich zamian (k = ∞) jest niewykonalnych (klasy n/m ≥ 2);
>   - przewaga ruchów złożonych zależy od nieporządku marszrut ρ: brak przy flow shopie, największa przy pośrednim ρ.
> - **Koszt:** ~6,5 dnia pracy Claude’a w chmurze + ~34 h Maca (opcja OPT). Bez kolizji z ICAPS/ANNSIM (harmonogram z bramkami, §10).
> - **Potrzebne decyzje:** D1 (czy JSSP w doktoracie); czy prace o „ruchach złożonych” (Bożejko–Wodecki) lub Gröflin–Klinkert nie zawierają T2; D15 (czasopismo czy GECCO).

---

## A. PROPOZYCJE USTALEŃ I PLAN WYKONAWCZY (01.11.2026)

> Zasada nadrzędna jak w k-bonaccim: **badamy sąsiedztwa, nie metaheurystyki.** ILS i SA mają te same stałe co w pracach o PFSP. Nowe są problem
> i sposób budowania sąsiedztw. Druga zasada dla JSSP: **jeden kod tam, gdzie się da** (DP, QUBO, analiza statystyczna). Wtedy różnice w wynikach
> wynikają z problemu, a nie z implementacji.

### A.0 Co już zrobiono w tej sesji (prototyp + pilot, chmura, ~5 h CPU łącznie z rundami krytyki)

| Element | Wynik |
|---|---|
| `scripts/jobshop_proto.py` | samodzielny prototyp (bez importów z `src/`): graf dysjunktywny, głowy i ogony, ścieżka krytyczna, bloki, rodziny N1/N5/Fib/k-bonacci/Dyn (+ testy Motzkina i inwolucji), estymata blokowa, test wykonalności z twierdzenia, DP z PFSP, ILS z adapterem JSSP |
| Benchmarki | JSPLIB: 162 instancje (abz, ft, la, orb, swv, ta01–80, yn). `data/` jest w `.gitignore`, więc pobieranie skryptem (P0) |
| Generator Taillarda dla JSSP | odtworzony; ta01 z ziaren 840612802 / 398197754 daje **identyczne** czasy i marszruty jak w JSPLIB. Pozwala zweryfikować ta01–ta80 jak `verify_taillard.py` dla PFSP |
| **Twierdzenie o przestawieniach bloków** (§4, T2) | sformułowane i udowodnione (szkic). Brute force: **0 rozbieżności** na 62 600 tasowaniach fragmentów bloków, 51 181 ruchach-inwolucjach i 70 680 pełnych tasowaniach bloków (wersja ścisła, po rundzie krytyki 1) |
| Lematy L1–L5 (§4) | 0 naruszeń: 84 203 zamiany sąsiednie, 169 118 ruchów złożonych z rozłącznych przedziałów, 2,55 mln par zagnieżdżonych, 40 702 zamiany poprawiające (filtr), 6 400 testów DP |
| Anatomia na Taillardzie (8 klas) | w optimach N5 (§14.2) i w dobrych rozwiązaniach po ILS (§14.2b): długości ścieżek i bloków, kandydaci, odsetek niewykonalnych |
| Kontinuum flow shop → job shop (ρ) | strukturalne (§14.3) i wydajnościowe: ILS w każdym ρ (§14.3b) |
| Zejście i ILS (1 s) | §14.4–14.5 |
| Rodzina ruchów granicznych (zamiany + wstawienia N6/N7, pojedyncze i złożone) | zaimplementowana; T2 dla rotacji: 0 rozbieżności / 24 967; pilot ILS §14.5b |

### A.1 Decyzje proponowane (do potwierdzenia z promotorem)

| # | Sprawa | Propozycja | Uzasadnienie |
|---|---|---|---|
| D1 | Miejsce w doktoracie | JSSP jako **rozdział uogólniający** rodzinę sąsiedztw („metoda działa poza permutacją”), a nie osobny front. Elastyczny job shop i hybrydowy flow shop na później | ROADMAP traktuje JSSP jako front B poza projektem. Pilot pokazuje, że przeniesienie wymaga głównie nowej warstwy (graf, ścieżka, wykonalność), a DP, QUBO i protokół zostają |
| D2 | Benchmark główny | **Taillard ta01–ta80** (8 klas × 10). To odpowiednik 12 plików PFSP. Instancje ft/la/orb/abz/swv/yn tylko do walidacji i porównań z literaturą. DMU opcjonalnie | ta to standard. Mamy generator, więc dane da się zweryfikować jak w PFSP |
| D3 | Miara | RPD do **aktualnego BKS** (plik z datą i źródłem) jako główna + RPD do LB dla ciągłości z PFSP | JSPLIB ma przestarzałe ograniczenia: część ta11–ta50 domknięto później (van Hoorn 2018; Coupvent des Graviers i in. 2025 domknęli ta33) |
| D4 | Start | **odpowiednik permutacji identycznościowej**: zadania rozdzielane po kolei (round-robin po indeksie), dekodowanie półaktywne. Deterministyczny, bez reguły priorytetowej | zgodność z D2 k-bonacciego („start z identyczności”). MWKR/INSA tylko jako ablacja |
| D5 | ILS / SA | **logika bez zmian**, przez adapter problemu (refaktoryzacja z testem regresji bit w bit dla PFSP). Te same stałe: τ = 10, pula 10, T0 = 1000, α = 0,99, Tmin = 100, stagnacja 100 ms z resetem T i kickiem | porównywalność; „badamy sąsiedztwa” |
| D6 | Kick | double bridge na **liście operacji** (porządek topologiczny bieżącego rozwiązania), potem dekodowanie półaktywne. Zawsze wykonalne | bezpośredni odpowiednik double bridge na π |
| D7 | Tabu | identyfikator ruchu = zbiór zamienionych par operacji (a nie pozycji) | pozycje na ścieżce zmieniają się co iterację |
| D8 | Budżety | 0,1 / 0,5 / 1 / 2 / 5 / 10 s, 5 ziaren; jeden przebieg 10 s i ekstrakcja budżetów z `move_log` | protokół k-bonacciego (A.2 tam) |
| D9 | Budowa sąsiedztwa | §2: jedna ścieżka krytyczna (remis rozstrzygany na rzecz poprzednika maszynowego), bloki, wirtualna permutacja, dokładny filtr granic, wykonalność z T2, DP z PFSP, dokładne C_max wybranego ruchu | §2, §4 |
| D10 | Ocena kandydatów | **estymata blokowa** (Taillard 1994 / Balas–Vazacopoulos 1998) dla wszystkich kandydatów + dokładne C_max ruchu wybranego. „Dokładne delty” (jedna pełna ocena na kandydata) tylko jako ablacja | przy 100×20 dokładna delta kosztuje ~3 ms na kandydata, a kandydatów jest do ~190 |
| D11 | Warianty (główne) | rodzina ruchów granicznych (§2.4): N5, Fib, k ∈ {3, 4}, Dyn (zamiany); N6, N7, c6, c7 (wstawienia pojedyncze i złożone); bm3 (wszystkie typy, zasięg 2). Do tego N1 jako najsłabszy punkt odniesienia. W MAX: k ∈ {6, 8}, bm4, bm∞, Inw oraz Motzkin i krzyżowania bez filtra. Motzkin pod filtrem ≡ Dyn (T10), więc go nie liczymy | §2.3, §2.4; pilot §14.5b |
| D12 | Kod | nowy pakiet `src/jobshop/`. Wspólne `solve_dp` i asemblery QUBO. **Zero zmian w ścieżkach PFSP**, poza adapterem w ILS/SA z testem regresji | bezpieczeństwo trwających kampanii |
| D13 | Liczenie | Mac M4 Pro, sekwencyjnie, **po** kampanii k-bonacciego. Chmura tylko do developmentu i testów | jedna tabela = jedna maszyna (§7) |
| D14 | Kwanty | po klasyce. QUBO z kandydatów ścieżki krytycznej jest małe i **blokowo-diagonalne** (T7), więc nie trzeba okien | §A.5 |
| D15 | Publikacja | podstawowo: **czasopismo** (C&OR / J. Scheduling / EJOR), Q1–Q2 2027. Opcjonalnie GECCO 2027 (ECOM), jeśli na bramce G3 (~20.12) H1, H6 i E4 są rozstrzygnięte i nie koliduje to z IJCAI dla k-bonacciego. Plan B: notatka teoretyczna (ORL) | §8, §10 |

### A.2 Co przenosi się z PFSP 1:1, co się zmienia, co jest nowe

| Element | PFSP (dotąd) | JSSP (propozycja) | Ocena |
|---|---|---|---|
| Rozwiązanie | permutacja π (n pozycji) | m permutacji maszynowych (n·m operacji) = selekcja w grafie dysjunktywnym | nowe |
| C_max | macierz Head O(mn) | najdłuższa ścieżka w DAG (sortowanie topologiczne) O(nm) | nowe, proste |
| Głowy i ogony | Head/Tail (m × n) | r(o), q(o) na operacjach | analogiczne |
| **„Pozycje” dla DP** | pozycje permutacji 1..n | **pozycje na ścieżce krytycznej 1..L (wirtualna permutacja)** | klucz przeniesienia |
| Bloki | odcinki między kolumnami zmiany maszyny U (≤ m − 1 granic) | maksymalne serie operacji jednej maszyny na ścieżce | analogiczne |
| Filtr bezstratny | T5 k-bonacciego: przedział musi zawierać kolumnę z U | zamiana musi zmienić pierwszą lub ostatnią operację bloku (w pierwszym bloku tylko ostatnią, w ostatnim tylko pierwszą) | znane (N5), tu dla ruchów złożonych (L3, L5) |
| Ruch elementarny | zamiana końców [i, j] **na wszystkich maszynach naraz** | zamiana dwóch operacji **jednego bloku** (jedna maszyna) | zmiana |
| Wykonalność | zawsze | **nie zawsze**; decyduje twierdzenie T2 | nowe, centralne |
| Delta kandydata | dokładna z Head/Tail, O(md) | estymata blokowa O(d) (dokładne dolne ograniczenie dla d = 1) albo dokładna O(nm) | zmiana |
| DP | `solve_dp`, O(n + \|K\|) | **ten sam kod** na pozycjach ścieżki; rozkłada się na bloki | 1:1 |
| Liczność | a_n (k-bonacci), M_n, I_n | **iloczyn po blokach** Π_b a(L_b) | nowe, proste |
| QUBO | zmienne = kandydaci, kara = konflikt | to samo, blokowo-diagonalne; dla krzyżowań dodatkowe kary wykonalności (dokładnie kwadratowe, T3d) | 1:1 + nowe |
| Addytywność | zawodzi (k-bonacci H5a) | zawodzi silniej: realny zysk to ~36–43% sumy delt (pilot, §14.6) | do zbadania |
| Start | permutacja identycznościowa | round-robin po indeksie zadania | analogiczne |
| Kick | double bridge na π | double bridge na liście operacji + dekodowanie | analogiczne |
| Benchmark | Taillard PFSP, 120 instancji | Taillard JSSP, 80 instancji (+ 82 klasycznych) | nowe |
| Weryfikacja danych | generator Taillarda PFSP | generator Taillarda JSSP (sprawdzony na ta01) | nowe |

### A.3 Plan wykonawczy: co i dlaczego, po kolei

**Etap 0: przygotowanie w chmurze (Claude, gałąź `claude/jobshop-plan` → `jobshop`, ~5–6 dni pracy).** Nie zajmuje Maca, więc może iść równolegle z kampanią k-bonacciego.

| # | Co | Dlaczego |
|---|---|---|
| 0.1 | `scripts/fetch_jsplib.py` (pobranie do `data/JSPLIB/`) + `scripts/verify_taillard_jssp.py` (generator; ziarna z Taillarda 1993) | dane jak w PFSP: zweryfikowane i odtwarzalne |
| 0.2 | `data/bks_jssp.json`: BKS z datą i źródłem dla ta01–80 i zestawów klasycznych | D3 |
| 0.3 | `src/jobshop/`: instancja, ocena (r, q, C_max, porządek topologiczny), ścieżka krytyczna, bloki | rdzeń |
| 0.4 | `src/jobshop/feasibility.py`: test z T2 (certyfikat głów + DFS) + testy brute force (L1–L5, T2) przeniesione z prototypu do `tests/` | poprawność |
| 0.5 | `src/jobshop/estimate.py`: estymata blokowa + test „dolne ograniczenie dla d = 1” | D10 |
| 0.6 | sąsiedztwa jako **jedna parametryzowana rodzina ruchów granicznych** (typ × zasięg × złożenie, §2.4). Tryby z prefiksem `js_`: `js_n1`, `js_n5`, `js_kbonacci_k<K>` (k = 2 to Fib, k = n to Dyn), `js_n6`, `js_n7`, `js_c6`, `js_c7`, `js_bm<K>`; w P2 `js_inw` i warianty bez filtra | jeden parser trybów, jeden kod (§7.1) |
| 0.7 | **Adapter problemu** w ILS/SA (start, C_max, sąsiad, kick, id ruchu) + **test regresji**: PFSP przed i po refaktoryzacji daje identyczne wyniki przy tym samym ziarnie i deterministycznym zegarze | D5; bezpieczeństwo starych wyników |
| 0.8 | runner: `RunConfig.problem ∈ {pfsp, jssp}`, ścieżka instancji JSPLIB, te same katalogi wyników i wznawianie | jedna infrastruktura |
| 0.9 | `scripts/run_jobshop_campaign.py` (etapy validate → main → continuum → ablation, dry-run z estymatą czasu, metadane jak `campaign_meta.jsonl`) | jak w k-bonaccim |
| 0.10 | `scripts/analyze_jobshop.py` (na bazie `analyze_kbonacci.py`): RPD do BKS/LB, Wilcoxon + Holm, Friedman + CD, liczba ruchów, rysunki | wyniki od razu do artykułu |
| 0.11 | `articles/jobshop/RUN_LOCAL.md`: instrukcja dla lokalnego Claude’a | przejęcie uruchamiania |
| 0.12 | Gantt dla JSSP (`src/visualization.py`) | rysunek 1 artykułu, kontrola wzrokowa |

**Etap 1: walidacja + kampania klasyczna (Mac).** Kolejność klas: 15×15 → 20×15 → 20×20 → 30×15 → 30×20 → 50×15 → 50×20 → 100×20.
**Etap 2: analiza i pisanie (chmura).** §2–§5 artykułu (konstrukcja i teoria) można pisać od razu, bo nie zależą od wyników.
**Etap 3: kwanty (po klasyce, po kredytach).**

### A.4 Trzy opcje: minimum / optimum / maksimum

Założenia jak w k-bonaccim: Mac sekwencyjnie, przebiegi 10 s z ekstrakcją budżetów, +10% narzutu.
Jedna para (wariant × algorytm) na pełnym Taillardzie to 80 instancji × 5 ziaren × 10 s, czyli **1,22 h**.

| | **MINIMUM** | **OPTIMUM (rekomendacja)** | **MAKSIMUM** |
|---|---|---|---|
| Warianty | N5, Fib, k = 3, Dyn, N6, c6 | N1, N5, Fib, k ∈ {3, 4}, Dyn, N6, N7, c6, c7, bm3 (11) | OPTIMUM + k ∈ {6, 8}, bm4, bm∞, Inw, Motzkin i ≤ c krzyżowań **bez filtra** z oceną top-t + TS-N5/TS-c6 (TSAB-light) + ablacje (remis na ścieżce, dokładne delty, bez filtra) |
| Algorytmy | ILS | ILS + SA | ILS + SA |
| Kampania główna (Mac) | 6 × 1,22 h ≈ **7,3 h** | 11 × 2 × 1,22 h ≈ **26,8 h** | 26,8 + 12,2 (5 wariantów × ILS + SA) + 2,4 (TS) + 2,4 (ablacje, 4 małe klasy) ≈ **43,8 h** |
| Kontinuum ρ (E4) | — | 4 rozmiary × 5 ρ × 10 inst. × 3 ziarna × 5 wariantów × 5 s ≈ **4,6 h** | to samo + SA ≈ **9,2 h** |
| Anatomia (E1) + zejście (E2) | ~1 h | ~2 h | ~3 h |
| Walidacja ekstrakcji | ~1 h | ~1 h | ~1 h |
| Zestawy klasyczne (82 instancje) | — | — | 6 wariantów × ILS ≈ **7,5 h** |
| **Razem (Mac)** | **~9 h** (jedna noc) | **~34 h** (~1,5 doby) | **~65 h** (~2,7 doby) |
| Kwanty | — | D-Wave MIN (§A.5) | D-Wave + IQM (§A.5) |
| Co daje artykułowi | przeniesienie działa; zamiany vs wstawienia, pojedyncze vs złożone (po jednym punkcie) | pełna mapa rodziny ruchów granicznych (typ × zasięg × złożenie), SA, kontinuum flow → job | + dalsze zasięgi, osie Motzkina i c bez filtra (krzyżowania z prawdziwymi ograniczeniami wykonalności), TS jako punkt odniesienia, ablacje, porównanie z literaturą na klasyce |

**Rekomendacja: OPTIMUM.** ~1,5 doby Maca. Daje hipotezy H1–H6 i H9 (mapa rodziny ruchów granicznych) oraz kontinuum, czyli najmocniejszy, nowy element artykułu.

### A.5 Kwanty na JSSP: struktura QUBO i koszty

Najważniejsze ustalenie pilota dla części kwantowej: **QUBO JSSP jest małe i rozkłada się na bloki.**
- Zamiany nie przekraczają granic bloków, więc konflikty występują tylko wewnątrz bloku.
- QUBO jest więc blokowo-diagonalne.
- Z filtrem granic blok ma ≤ 2(k − 1) zmiennych.
- Liczba zmiennych dla całego ruchu w optimach N5 (§14.2): k = 2 daje 7–22, k = 3 daje 11–40, k = 4 daje 14–55.
  W dobrych rozwiązaniach (po ILS) jest jeszcze mniej: k = 2 daje 4–8, k = 3 daje 8–14, k = 4 daje 12–19.
- Przy k = ∞ (Dyn) jest 18–186 zmiennych, z czego 19–67% odpada jako niewykonalne.

| | MIN | OPT | MAX |
|---|---|---|---|
| **D-Wave** (1 wywołanie na ruch, ~30 ms QPU, embedding trywialny) | ta01–10, k ∈ {2, 3}, ILS, 5 ziaren, reżim jednego ruchu: 100 wywołań, **~3 s QPU** | wszystkie klasy, k ∈ {2, 3, 4}, ILS + SA, 5 ziaren, 1–3 ruchy na przebieg: 2 400–7 200 wywołań, **72–216 s** (0,2–0,5 mies. quoty 400 s) | + Dyn i inwolucje bez filtra z karami wykonalności (gęstsze QUBO) |
| **IQM** (jeden obwód na ruch, całe QUBO ≤ 20 kubitów) | ta01–10, k = 2, ILS, ziarno 0: **10 obwodów ≈ 5–8 kredytów** (darmowa pula) | ta01–20 z k = 2 + ta01–10 z k = 3, 1 ziarno: ~30 obwodów ≈ **14–23 kredyty** | to samo dla 5 ziaren: ~150 obwodów ≈ **70–115 kredytów (~40–70 USD)** |
| **Odra 5 / IQM Spark (5 kubitów)** | bloki osobno: przy k ≤ 3 blok ma ≤ 4 zmienne, więc **mieści się bez okien** | — | — |

Koszt IQM (poprawiony 11.11.2026): zmierzona stawka to 0,445 kredytu za obwód przy p = 1 (4096 shotów), a najwyższy pomiar (7.10.2026) dał 0,75.
Wersja z 01.11 liczyła 1,2 kredytu za obwód, czyli zawyżała koszt mniej więcej dwukrotnie. Czas D-Wave bez zmian: mediana dostępu do QPU
z 24 317 wywołań (`results/qpu_timing.jsonl`) to 28,6 ms.
Sirius ma 16 kubitów. Całe ruchy mieszczą się dla k ≤ 3 (8–14 zmiennych w dobrych rozwiązaniach), a dla k = 4 bywa 12–19, więc część ruchów wymaga Garnet (20 kubitów).

Uczciwie: QUBO bloku ma ≤ 2(k − 1) zmiennych, więc DP rozwiązuje je dokładnie w mikrosekundach. Wartość części kwantowej to taki sam pomost
metodyczny jak w PFSP (protokół, reżim jednego ruchu), nie szybkość. Wyjątek: rodziny z krzyżowaniami. Tam kary wykonalności (T3d) tworzą
niebanalne QUBO wewnątrz bloku i to jest ciekawy przypadek dla QAOA.

### A.6 Przegląd krytyczny: trzy rundy (01.11.2026)

Plan v1 przeszedł trzy rundy „wewnętrznego krytyka”. Każdy zarzut sprawdziłem obliczeniem albo poprawką w tekście.

**Runda 1: poprawność i metodologia**

| # | Zarzut | Co zrobiono | Skutek |
|---|---|---|---|
| 1 | Anatomia i kontinuum liczone na słabych optimach N5 (RPD 15–160%), więc wnioski zależą od jakości stanów | powtórzone na stanach po 2 s ILS (§14.2b, §14.3) i dodany test wydajnościowy kontinuum (§14.3b) | „poprawiające dalekie zamiany” to artefakt. Niewykonalność dalekich ruchów rośnie z jakością (do 67%). H3 i H5 przeformułowane |
| 2 | Po T10 rodziny zamian zapadają się do ≤ 2 ruchów granicznych na blok, a plan pomijał wstawienia (N6/N7), naturalne w JSSP | rodzina ruchów granicznych (typ × zasięg × złożenie) zaimplementowana i przetestowana (T2 dla rotacji 0 / 24 967), pilot ILS (§14.5b) | nowa propozycja główna (§2.4), H9, warianty OPT |
| 3 | Test T2 w prototypie dopuszczał ścieżki przez inne operacje bloku. Czy to dokładne dla dowolnych przestawień? | pełne tasowania bloków (70 680): wersja luźna 1 fałszywy alarm, wersja ścisła 0 | T2 sformułowane z E-ścieżką omijającą blok; kod poprawiony. Dla zamian, wstawień i rodzin laminarnych obie wersje są równoważne |
| 4 | Wniosek „najlepsze k rośnie z długością bloków” oparty na jednej instancji (ta51) | złagodzony do „przesłanki” | H6 |
| 5 | Wybór ruchu rozkłada się na bloki, więc DP jest trywialne; recenzent to zauważy | powiedziane wprost (§0 p. 5); wkład przeniesiony na typ/zasięg ruchu, T2, kontinuum | §0, §11 |
| 6 | Nowość T2 niepewna | dodane źródła (Gröflin–Klinkert, Dauzère-Pérès–Paulli, Mati i in., Brucker–Knust) i ryzyko. Pracy Bożejki–Wodeckiego nie dało się pobrać, bo domena jest poza siecią sesji | §11, §13 |

**Runda 2: realizm (czas, zasoby, priorytety)**

| # | Zarzut | Poprawka |
|---|---|---|
| 1 | Harmonogram kalendarzowy ignorował zajętość Maca i kolizje z ICAISC/ICAPS/ANNSIM | harmonogram z bramkami G0–G3; Mac najwcześniej ~23.11, najpóźniej ~20.12; pisanie od 19.12 (§10) |
| 2 | Brak planu B, gdyby JSSP wypadł z zakresu albo wyniki były słabe | plan B: notatka teoretyczna (ORL) z T2, T3, T8, T10 i eksperymentem MIN (§10) |
| 3 | Kwanty JSSP konkurują o quotę D-Wave i kredyty IQM z k-bonaccim | kwanty JSSP w Q1 2027, po k-bonaccim (§10) |
| 4 | Społeczność JSSP oczekuje TS, a nie ILS z kickiem | TS-N5/TS-c6 (TSAB-light) w MAX (E6) |
| 5 | Więcej wariantów (rodzina ruchów granicznych) wydłuża kampanię | OPT 11 wariantów: ~34 h (nadal ~1,5 doby); MAX ~65 h |

**Runda 3: forma i spójność**

| # | Zarzut | Poprawka |
|---|---|---|
| 1 | Promotor nie przeczyta 60 KB | streszczenie na początku (1 minuta) |
| 2 | Niespójne odsyłacze i listy wariantów po zmianach | ujednolicone A.1, A.4, §6, §9, §15; wersja dokumentu |
| 3 | Pilot nieodtwarzalny bez danych | komenda pobrania JSPLIB w §14 |

### A.7 Runda 4: przegląd po przeniesieniu na Maca (11.11.2026)

Przegląd lokalny: plan przeczytany w całości, liczby z §14 przeliczone z `pilot/*.json`, kod z `src/jobshop` uruchomiony (testy), dowód T2 sprawdzony krok po kroku.

**Co się potwierdziło**
- Tabele §14.3b i §14.5b zgadzają się z surowymi przebiegami (przeliczone od zera). Suma przypadków T2: 62 600 + 51 181 + 70 680 = 184 461.
- Dowód T2 jest poprawny. Brakowało w nim jednego kroku, teraz dopisanego w szkicu artykułu (lemat: obejście z x do y nie zawiera żadnej innej operacji ścieżki krytycznej, więc nie używa relacji zmienionych w żadnym bloku; bez tego kierunek (⇐) nie działa, gdy ruch przestawia kilka bloków naraz).
- Kontrprzykład dla wersji luźnej da się wskazać wprost (ścieżka x ⇝ z → z′ ⇝ y przez łuk maszynowy z → z′ wewnątrz bloku, nowa kolejność z′, y, x, z), więc wersja ścisła nie jest artefaktem testu.
- Arytmetyka czasu kampanii (1,22 h na parę, 26,8 h, 4,6 h) jest poprawna.

**Stan kodu (różnice względem §9 i A.3)**

| Plan (v2) | Stan faktyczny |
|---|---|
| `core.py`, `estimate.py`, `data/bks_jssp.json` | `instance.py`, `schedule.py`, `feasibility.py`, `neighborhoods.py` (estymata w środku), `algorithms.py`, `runner.py`; ziarna i BKS w `src/jobshop/data/taillard_jssp.json` |
| 0.7 / D5 / D12: adapter problemu w ILS/SA + test regresji PFSP bit w bit | **zrobiono lustrzaną kopię** ILS/SA w `src/jobshop/algorithms.py`; kod PFSP nietknięty, więc test regresji jest zbędny. Koszt: zgodność stałych pilnuje tylko test porównujący `SA_DEFAULTS`/`ILS_DEFAULTS` z `config.yaml` (do dopisania) |
| etapy validate → main → continuum → ablation | validate → min → opt → continuum → bridge → ablation → classic (`run_jobshop_campaign.py`) |
| OPT ≈ 34 h | dry-run ILS: validate 1,5 h, min 6,8 h, opt 5,6 h, continuum 4,3 h, bridge 1,3 h, razem **~19,4 h**; SA („later, on request”) podwaja min + opt, czyli **~32 h** |
| 0.12 Gantt, `RUN_LOCAL.md`, szkic artykułu | Gantt niezrobiony; `RUN_LOCAL.md` niezrobiony; szkic: `main.tex`, `wersja_pl.tex` (11.11) |

**Usterki znalezione w rundzie 4**

| # | Usterka | Waga | Co zrobić |
|---|---|---|---|
| 1 | **`src/jobshop/data/taillard_jssp.json` nie trafił do repo**: reguła `data/` w `.gitignore` (linia 49) łapie też ten katalog. Bez pliku nie działa `verify_taillard_jssp.py`, `bks()` ani jeden z 25 testów (24 przechodzą) | blokuje kampanię | w sesji chmurowej `git add -f src/jobshop/data/taillard_jssp.json` i push; dodać do `.gitignore` wyjątek `!src/jobshop/data/`. Jeśli kontener przepadł: ziarna są w plikach Taillarda (`tai15_15.txt` itd.), BKS trzeba zebrać ponownie ze źródłami |
| 2 | **„Filtr bezstratny” (T4, C3) jest bezstratny tylko względem modelu addytywnego.** Zamiana wewnętrzna ma δ ≥ 0 osobno, więc DP jej nie wybierze, ale w połączeniu z ruchem granicznym zmienia harmonogram i może dać niższe C_max. Przy addytywności na poziomie 36–43% (§14.6) to nie jest drobiazg | teza | T4 formułować dwustopniowo: (a) twierdzenie blokowe: ruch bez zmiany granicy nie poprawia; (b) wniosek: optimum **addytywnego** DP się nie zmienia. Prawdziwe sąsiedztwo złożone jest większe; ablacja „bez filtra” (MAX) mierzy różnicę. To samo dotyczy „Motzkin ≡ Dyn” (już opisane jako „przy addytywnym DP”) |
| 3 | **Tytuł, §0 i C1 nadal prowadzą „rodziną liczoną ciągami”**, chociaż T10 i runda 1 przeniosły ciężar na rodzinę ruchów granicznych. Po filtrze ciągi (Fibonacci, k-bonacci, Motzkin) nie opisują już efektywnego sąsiedztwa bloku | spójność | tytuł roboczy: *Compound border moves for the job shop: feasibility of block reorderings and the flow-shop-to-job-shop continuum*. Liczności ciągowe zostają jako jedno stwierdzenie (sąsiedztwo formalne, iloczyn po blokach) |
| 4 | **H9 w pilocie jest słabsze, niż mówi §14.5b.** Test znaków na 11 instancjach: k = 3 vs N5 11:0 (p ≈ 0,001); c7 vs N7 10:1 (p ≈ 0,01); **c6 vs N6 7:3 i remis (p ≈ 0,34)**; k = 3 vs N6 9:2 (p ≈ 0,07). „Złożenie pomaga wstawieniom” jest w pilocie pokazane tylko dla N7 | wniosek | w artykule pilot opisywać jako kierunek; H9 rozstrzyga etap opt |
| 5 | **ILS w Pythonie przy RPD 5–13% nie przekona recenzentów czasopisma JSSP.** TS-N5/TS-c6 jest w MAX, a to właśnie on odpowiada na pytanie „czy ranking sąsiedztw przenosi się na metodę, której wszyscy używają” | ryzyko recenzji | **przenieść TS-N5 i TS-c6 (TSAB-light) z MAX do OPT** (~2,4 h Maca, ~1 dzień kodu). Bez tego celować w plan B albo w konferencję |
| 6 | **Budżety 0,1–10 s są stałe**, a ocena przy 100×20 kosztuje ~3 ms: 0,1 s to kilkanaście iteracji. Najmniejsze budżety dla dużych klas mierzą start, nie sąsiedztwo | protokół | zostawić (zgodność z PFSP), ale w tabeli głównej podawać też liczbę iteracji i dodać kolumnę przy budżecie proporcjonalnym do n·m (ekstrakcja z tego samego przebiegu, bez nowych obliczeń) |
| 7 | **H2: Spearman na 8 klasach** ma moc bliską zera (próg ρ_S ≥ 0,5 przy n = 8 nie jest istotny) | statystyka | liczyć na poziomie instancji (80 punktów: L̄ instancji vs najlepsze k), klasy tylko na rysunku |
| 8 | **Kontinuum ma konfundującą zmienną: liczbę iteracji.** Przy ρ = 0 N5 robi 2–4× więcej iteracji niż c6/Dyn, więc „N5 wygrywa we flow shopie” może być efektem kosztu, nie struktury | H5 | w E4 raportować wynik także przy równej liczbie iteracji (z `move_log`); ρ = 1 przy 20×5 i 50×10 jest zdegenerowane (same zera), więc te dwa punkty nie niosą informacji |
| 9 | `pilot/ils_border_b.json`, `ils_estimate_1s.json`, `ils_exact_1s_large.json`: **ta21 liczone do LB = 1217, nie do BKS = 1642** (stąd 51–56% w plikach). Tabele w §14.5 i §14.5b są przeliczone do BKS poprawnie, ale pole `summary` w plikach nie | odtwarzalność | pamiętać przy generowaniu tabel artykułu z JSON; w `src/jobshop` BKS pochodzi już z pliku danych |
| 10 | Nowość T2 nadal niesprawdzona (bramka G0 minęła 8.11 bez lektury) | ryzyko | bez zmian: Gröflin–Klinkert 2007, Dauzère-Pérès–Paulli 1997, Mati i in. 2011, prace Bożejki–Wodeckiego o ruchach złożonych; PDF-y od promotora |

**Harmonogram po aktualizacji (zastępuje daty w §10; bramki i warunki bez zmian)**

Stan na 11.11: ICAISC wysłany 9.11; kampania k-bonacciego jeszcze nie ruszyła, więc Mac nie zwolni się ~5–8.11, jak zakładał §10.
Przed JSSP stoją: GCMM (16.11), rewizja BPASTS (3.12), ICAPS k-bonacci (7.12 abstrakt, 14.12 artykuł), ANNSIM (18.12).

| Kiedy | Co | Warunek |
|---|---|---|
| teraz, bez Maca | odzyskać plik danych (usterka 1); `RUN_LOCAL.md`; test zgodności stałych z `config.yaml`; TS-light, jeśli decyzja z usterki 5 jest na tak | 25/25 testów, `verify_taillard_jssp.py` 80/80 |
| do ~30.11 | **G0**: rozmowa z promotorem (D1, D15, nowość T2) | JSSP w zakresie; T2 nieopisane w tej postaci |
| po kampanii k-bonacciego, najwcześniej po 14.12 | etap validate (1,5 h), potem **G1** | Spearman ≥ 0,7 dla estymaty d ≥ 2; test wykonalności < 10% iteracji |
| ~15–31.12 | min → opt → continuum → bridge (~18 h ILS), potem SA | — |
| styczeń 2027 | **G2**, analiza, decyzja G3 (czasopismo; konferencja tylko jeśli termin pasuje) | jak w §10 |
| luty–marzec 2027 | pisanie i zgłoszenie; kwanty JSSP bez zmian (Q1 2027) | — |

GECCO 2027 przy takim układzie jest nierealne (kampania kończy się na przełomie roku, a termin w 2026 r. wypadał 26.01). Zostaje czasopismo albo plan B.

---

## 0. Skrót

- **Pytanie główne** (tytuł roboczy EN od wersji 3: *Compound border moves for the job shop: feasibility of block reorderings and the flow-shop-to-job-shop continuum*;
  poprzedni: *From flow shop to job shop: sequence-counted compound neighborhoods on the critical path*):
  czy rodzina sąsiedztw liczonych ciągami (Fibonacci → k-bonacci → Dynasearch → Motzkin → inwolucje) przenosi się z permutacyjnego flow shopu na job shop?
  Jeśli tak, to co ją ogranicza i co ją wzmacnia?
- **Teza (do potwierdzenia):**
  1. Przeniesienie jest naturalne. **Ścieżka krytyczna pełni rolę permutacji**, a bloki rolę odcinków między granicami.
     DP, liczności, filtr bezstratny i QUBO przechodzą prawie bez zmian (liczność to iloczyn po blokach).
  2. Jedyną nową przeszkodą jest **wykonalność**. Opisuje ją jedno twierdzenie (T2): przestawienie wewnątrz bloków jest niewykonalne
     ⟺ istnieje odwrócona para operacji połączona ścieżką, która wychodzi i wchodzi łukami zadaniowymi i omija resztę bloku.
     Wnioski:
     - zamiany sąsiednie są zawsze wykonalne (van Laarhoven i in. 1992);
     - **rodziny laminarne** (k-bonacci, Dynasearch, Motzkin) wymagają tylko testu pojedynczych zamian;
     - **krzyżowania** wprowadzają prawdziwe ograniczenia dwuciałowe (dokładnie kwadratowe w QUBO).
  3. **Flow shop to granica bez przeszkód:** przy wspólnych marszrutach (ρ = 0) każde przestawienie bloku jest wykonalne.
     Kontinuum ρ ∈ [0, 1] to oś, na której widać „przejście flow → job”.
     Pilot: przewaga ruchów złożonych nad N5 jest niemonotoniczna. Przy ρ = 0 jej nie ma, przy pośrednim ρ jest duża (N5 traci 4–22 pp), przy ρ = 1 zależy od n/m.
  4. Korzyść z zasięgu k zależy od długości bloków. Te zależą od n/m i od ρ: przy n/m ≈ 1 hierarchia się spłaszcza, przy n/m ≥ 2 się rozwiera.
     **Uwaga:** w dobrych rozwiązaniach (po ILS) 45–57% dalekich zamian (k = ∞) przy 30×15 i 50×15 jest niewykonalnych.
     Zysk z zasięgu jest więc ograniczony przez wykonalność, nie tylko przez koszt.
  5. **Uczciwie o algorytmie:** wybór ruchu rozkłada się na bloki (T7), a filtr spłaszcza rodziny zamian do ≤ 2 ruchów granicznych na blok (T10).
     DP jest tu trywialne (suma optimów bloków). Wkładem nie jest więc algorytm wyboru, tylko:
     - (a) typ i zasięg ruchu granicznego (§2.4);
     - (b) teoria wykonalności (T2 i wnioski);
     - (c) kontinuum flow → job.
- **Co już mamy (pilot, §14):**
  - twierdzenie T2 z 0 rozbieżności na ~184 tys. przypadków brute force (wersja ścisła);
  - zejście: k ≥ 3 daje lepsze optima lokalne niż N5 (RPD 28,4–28,6% vs 30,8%; 35–39 wygranych na 60 par);
  - ILS 1 s: Fib i k-bonacci są lepsze od N5. Z dokładnymi deltami RPD wynosi 7,5–8,3% vs 9,0% (N1 11,3%).
    Z estymatą blokową: 5,1–6,2% vs 7,3%, przy 3–4× większej liczbie iteracji.
  - Addytywność zawodzi mocno: realny zysk to ~36–43% sumy delt.

---

## 1. Kontekst i luka badawcza

**Co wiadomo (do potwierdzenia lekturą, §13):**
1. **Sąsiedztwa JSSP na ścieżce krytycznej:**
   - N1: wszystkie zamiany sąsiednich operacji krytycznych; zawsze wykonalne, spójne (van Laarhoven, Aarts, Lenstra 1992);
   - N4 (Dell’Amico, Trubian 1993);
   - **N5**: pierwsza i ostatnia para bloku (Nowicki, Smutnicki 1996, TSAB);
   - **N6**: przeniesienie operacji przed lub za blok, z warunkami wykonalności (Balas, Vazacopoulos 1998);
   - N7: rozszerzenie N6 (Zhang i in. 2007).
2. **Ocena ruchu bez pełnego przeliczenia:** estymata Taillarda (1994) i jej odpowiedniki dla wstawień (Balas, Vazacopoulos 1998).
3. **Metaheurystyki SOTA:** i-TSAB (Nowicki, Smutnicki 2005), TS/PR (Peng, Lü, Cheng 2015), GLS z wyrzucaniem operacji (Nagata, Ono 2018),
   algorytm memetyczny z jawnym zarządzaniem różnorodnością (Constantino, Segura 2022).
   Porównania sąsiedztw: Kuhpfahl i Bierwirth (2016, kryterium TWT).
4. **Wykładnicze sąsiedztwa (Dynasearch, VLSN)** dotyczą głównie jednej maszyny i maszyn równoległych (Congram i in. 2002; Ahuja i in. 2002).
   W szybkim wyszukiwaniu nie znalazłem Dynasearch dla JSSP z analizą wykonalności ruchów złożonych.
   Bożejko i Wodecki mają jednak prace o blokach i „technice ruchów złożonych” (compound moves) w TS. **Trzeba je przeczytać przed decyzją o nowości** (§11, §12).
5. **Nasza rodzina dla PFSP** (Adjacent, Fib, k-bonacci, Dyn, Motzkin; plan krzyżowań). Tam wykonalność jest darmowa.

**Luka:**
- Nikt nie przeniósł systematycznie rodziny sąsiedztw liczonych ciągami na JSSP. Znane są pojedyncze punkty (N1, N5) i pojedyncze ruchy wstawiające.
- Brak **teorii wykonalności ruchów złożonych** w blokach. Znane wyniki dotyczą pojedynczego ruchu (N1, warunki N6).
- Nie wiadomo, **jak wartość ruchów złożonych zależy od struktury marszrut**, czyli od przejścia flow shop → job shop.
- Nie zmierzono **addytywności** ruchów złożonych w JSSP.

---

## 2. Proponowany sposób budowania sąsiedztw (otoczeń) dla JSSP

### 2.1 Model i oznaczenia
- Zadanie j ma marszrutę (μ_{j,1}, …, μ_{j,m}) i czasy p_{j,t}. Operacja o = (j, t) ma poprzednika zadaniowego JP(o) i następnika JS(o).
- Rozwiązanie to m permutacji maszynowych σ_1, …, σ_m. Daje ono poprzednika i następnika maszynowego MP(o) i MS(o).
  Graf dysjunktywny G(σ): łuki zadaniowe i maszynowe. Rozwiązanie jest wykonalne ⟺ G(σ) jest acykliczny.
- Głowa r(o) to najdłuższa ścieżka do o (bez p_o), ogon q(o) to najdłuższa ścieżka od o (bez p_o). C_max = max_o r(o) + p_o + q(o).
  Liczenie przez sortowanie topologiczne kosztuje O(nm).

### 2.2 Ścieżka krytyczna jako wirtualna permutacja
1. **Ścieżka P** = (o_1, …, o_L), odtwarzana wstecz od operacji kończącej się w C_max.
   Przy remisie (oba poprzedniki krytyczne) wybieramy **poprzednika maszynowego**, co daje dłuższe bloki. Ablacja: zadaniowy lub losowy.
2. **Bloki** B_1, …, B_b to maksymalne serie kolejnych operacji P połączonych łukami maszynowymi. Każdy blok to spójny fragment jednej σ_k.
3. **Wirtualna permutacja** to pozycje 1..L ścieżki. Kandydat to zamiana końców (a, b) operacji jednego bloku, czyli przedział [s + a, s + b] pozycji ścieżki.
   Przedziały z różnych bloków są automatycznie rozłączne.
4. Porównanie z PFSP:
   - w PFSP wirtualną permutacją jest sama π, a ścieżka krytyczna idzie przez siatkę (maszyna × pozycja);
   - jej poziome odcinki to bloki, pionowe kroki to łuki zadaniowe;
   - ruch PFSP zmienia kolejność **na wszystkich maszynach naraz**, ruch JSSP tylko na jednej.

### 2.3 Rodziny na wirtualnej permutacji

L oznacza długość bloku. „Liczność” to liczba ruchów złożonych w bloku (z ruchem pustym); całe sąsiedztwo to **iloczyn po blokach**.

| Sąsiedztwo | Ruchy elementarne w bloku | Ruch złożony | Liczność w bloku | Kandydaci po filtrze (blok) | Wykonalność | Wybór |
|---|---|---|---|---|---|---|
| N1 | zamiany sąsiednie | jeden ruch | L − 1 | ≤ 2 | zawsze (T3a) | min δ |
| N5 | pierwsza i ostatnia para | jeden ruch | ≤ 2 | ≤ 2 | zawsze | min δ |
| **Fib (k = 2)** | zamiany sąsiednie | rozłączne | F(L + 1) | ≤ 2 (pary N5) | zawsze | DP |
| **k-bonacci** | zamiany końców, d ≤ k − 1 | rozłączne przedziały | a_k(L) | ≤ 2(k − 1) | test pojedynczych (T3b) | DP |
| **Dyn (k = ∞)** | dowolne zamiany końców | rozłączne przedziały | 2^{L−1} | ≤ 2L − 3 | test pojedynczych (T3b) | DP |
| **Motzkin** | dowolne zamiany | bez krzyżowań, zagnieżdżanie wolno | M_L | ≤ 2L − 3 | test pojedynczych (T3b) | z filtrem ≡ Dyn (T10); bez filtra DP przedziałowe O(L³) w bloku |
| ≤ c krzyżowań | dowolne zamiany | ≤ c przecięć łuków | jak w katalogu, w bloku | ≤ 2L − 3 | pary (T3c, T3d) | z filtrem c ≤ 1 na blok (T10); bez filtra DP z krzyżowaniami / MILP |
| Inwolucje (**Inw**) | dowolne zamiany | rozłączne końce | I_L | ≤ 2L − 3 | pary (T3c, T3d) | z filtrem: wyliczenie O(L²) w bloku; bez filtra skojarzenie z zakazami / QUBO |
| N6 (wstawienie) | przeniesienie operacji przed lub za blok | jeden ruch | ≤ 2(L − 1) | ≤ 2(L − 1) | T2 (para „przeskakiwana”) | min δ |
| Okno de Montmorta | rotacje i deranżacje w oknie | rozłączne okna | jak w katalogu | — | T2 | DP po oknach |

**Filtr granic (bezstratny, L3):** kandydat musi zmienić pierwszą lub ostatnią operację bloku. W pierwszym bloku liczy się tylko ostatnia,
w ostatnim bloku tylko pierwsza. Przy jednym bloku C_max to obciążenie maszyny, więc rozwiązanie jest optymalne.
Dla Fib filtr redukuje kandydatów dokładnie do par N5, więc **Fib = „złożone N5”**.

**Ważne (T10): filtr spłaszcza rodziny w JSSP.** Po filtrze każda zamiana w bloku ma koniec na granicy: (0, b) albo (a, L − 1). Stąd:
- **W bloku są najwyżej dwie zamiany:** jedna przy początku i jedna przy końcu. W pierwszym i ostatnim bloku najwyżej jedna.
- **Zagnieżdżenie jest niemożliwe**, bo zamiana wewnętrzna nie dotyka granicy, więc filtr ją usuwa. Zatem przy addytywnym DP **Motzkin ≡ Dyn**.
- **Krzyżowanie jest możliwe tylko jako para** (0, b), (a, L − 1) z a < b. Oś c z planu krzyżowań zawęża się w JSSP do **c ∈ {0, 1} na blok**,
  a wybór w bloku to wyliczenie O(L²) zamiast skojarzenia.
- Liczba ruchów złożonych w bloku wewnętrznym po filtrze jest wielomianowa:
  - Dyn: 2L − 2 + C(L − 2, 2);
  - z krzyżowaniem: 2L − 2 + 2·C(L − 2, 2);
  - Fib: 4 dla L ≥ 4.

  Całe sąsiedztwo (iloczyn po blokach) pozostaje wykładnicze w liczbie bloków.
- Wniosek dla planu:
  - w kampanii (OPT) liczymy rodzinę ruchów granicznych (§2.4). Wariant **Inw** (Dyn + dopuszczona krzyżująca się para graniczna, z testem pary z T3c) idzie do MAX;
  - pełne osie Motzkina i krzyżowań mają sens w JSSP tylko **bez filtra**, przy ocenie nieaddytywnej (ablacja w MAX, E5).

Pełne (niefiltrowane) liczności z kolumny „Liczność” opisują sąsiedztwo formalnie, np. do porównania z PFSP (§14.2).

### 2.4 Rodzina ruchów granicznych: propozycja główna dla JSSP (dodane po rundzie krytyki 1)

T10 pokazuje, że po filtrze wszystkie rodziny zamian sprowadzają się do najwyżej dwóch ruchów na blok: jednego przy początku i jednego przy końcu.
Naturalnym uogólnieniem w JSSP nie jest więc geometria łuków (zagnieżdżenia, krzyżowania), tylko **typ ruchu granicznego**.
Proponuję rodzinę z trzema osiami:

| Oś | Wartości |
|---|---|
| **typ** ruchu, który wymienia operację graniczną | zamiana końców (S); wstawienie operacji wewnętrznej przed pierwszą lub za ostatnią (N6, Balas–Vazacopoulos); przesunięcie pierwszej lub ostatniej operacji w głąb bloku (N7, Zhang i in.) |
| **zasięg** k | odległość ruchu ≤ k − 1 (k = 2: tylko sąsiednia zamiana, wtedy wszystkie typy są tym samym ruchem) |
| **złożenie** | pojedynczy ruch (najlepszy) albo ruch złożony: ≤ 1 ruch na każdy koniec bloku, przedziały rozłączne, wybór DP (`solve_dp` bez zmian) |

Znane sąsiedztwa to punkty tej przestrzeni:

| | pojedynczy | złożony |
|---|---|---|
| S, k = 2 | **N5** | **Fib** |
| S, k | — | **k-bonacci** |
| S, k = ∞ | — | **Dyn** |
| N6, k = ∞ | **N6** | **c6** (nowe) |
| N6 + N7, k = ∞ | **N7** | **c7** (nowe) |
| S + N6 + N7, k | — | **bm_k** (nowe) |

Wykonalność wstawień też rozstrzyga T2. Rotacja jest przestawieniem bloku.
- Przeniesienie u w przód (za przeskakiwane x) odwraca pary (u, x). Ruch jest niewykonalny ⟺ istnieje ścieżka JS(u) ⇝ JP(x) dla pewnego x.
- Przeniesienie u wstecz odwraca pary (x, u). Ruch jest niewykonalny ⟺ istnieje ścieżka JS(x) ⇝ JP(u).
- Warunek Balasa i Vazacopoulosa (L(v, n) ≥ L(JS(u), n)) **implikuje** brak takiej ścieżki. Jest więc wystarczający, a T2 daje warunek dokładny.
  Do sprawdzenia, czy dokładna wersja jest znana (Dauzère-Pérès i Paulli 1997; Mati i in. 2011; §13).
- Ruchy złożone z rozłącznych przedziałów są wykonalne ⟺ wykonalne są składowe (T3b, bo rotacje w rozłącznych przedziałach też dają sumę zbiorów odwróconych par).
- Walidacja w prototypie: **0 rozbieżności T2 na 24 967 ruchach granicznych** (775 niewykonalnych) i **0 niewykonalnych na 6 291 złożeniach** wykonalnych składowych.
  Estymata dla d = 1 pozostaje dolnym ograniczeniem (0 naruszeń).

Okno de Montmorta z katalogu (#12) to dalsze uogólnienie: dowolne rotacje i deranżacje w oknach. W JSSP, przez filtr, sensowne są tylko okna dotykające granicy bloku.

Wyniki pilota ILS dla tej rodziny: §14.5b.

### 2.5 Ocena kandydatów i wybór ruchu
1. **Estymata blokowa** (D10): przelicza głowy (od lewej) i ogony (od prawej) operacji przedziału w nowej kolejności.
   Głowy JP i maszynowego poprzednika przedziału oraz ogony JS i maszynowego następnika bierze z bieżącego rozwiązania. Koszt O(d).
   - Dla d = 1 to **dokładne dolne ograniczenie** nowego C_max, równe mu, gdy ruch nie poprawia (pilot: 0 naruszeń na 84 203).
   - Dla d ≥ 2 to przybliżenie; jakość do zmierzenia w E0.
2. **Wybór:** DP z PFSP (`solve_dp`) na kandydatach z deltami δ̂ = Ĉ − C. Rozkłada się na niezależne optima bloków.
3. **Ruch pusty:** jeśli DP nic nie wybiera, stosujemy najlepszą pojedynczą zamianę, jak w PFSP (`force_move_if_none`).
4. **Dokładna ocena** wybranego ruchu złożonego: jedna pełna ocena O(nm). Wymagana, bo addytywność zawodzi (§14.6).
5. **Identyfikator tabu:** zbiór zamienionych par operacji (D7).

### 2.6 Wykonalność w praktyce
- d = 1 to zawsze wykonalne, bez testu.
- d ≥ 2: odwrócone pary zamiany (i, j) to (σ_i, σ_y) dla y ∈ (i, j] i (σ_x, σ_j) dla x ∈ [i, j).
- Para (x, y) jest „zakazana” ⟺ istnieje E-ścieżka: JS(x) ⇝ JP(y) (albo JS(x) = JP(y)) bez przechodzenia przez inne operacje bloku.
  Dla pojedynczej zamiany i wstawienia wystarcza zwykła osiągalność (równoważne, T2).
- **Certyfikat głów:** ścieżka a ⇝ b wymaga r(b) ≥ r(a) + p(a). Jeśli r(JP(y)) ≤ r(JS(x)), to ścieżki nie ma.
  Pilot: certyfikuje **89,8%** wykonalnych zamian d ≥ 2 i ani razu się nie myli (0 / 118 075).
- **DFS** z JS(x) przycinany głowami (tylko węzły z r ≤ r(JP(y))) rozstrzyga pozostałe przypadki dokładnie.
- Dla rodzin laminarnych to wszystko: ruch złożony jest wykonalny ⟺ każda zamiana jest wykonalna (T3b).
- Dla krzyżowań sprawdzamy dodatkowo pary krzyżujących się zamian. Kary wchodzą do QUBO, a do DP i MILP jako ograniczenia (T3c, T3d).

### 2.7 Jedna iteracja (pseudokod)

```
ocena(σ) → r, q, C_max, porządek topologiczny                # O(nm)
P ← ścieżka_krytyczna(r, remis = maszynowy)                  # O(L)
B ← bloki(P)
K ← ∅
dla bloku (s, L) z B:
    dla ruchu granicznego (a, b, typ) zgodnego z rodziną (typ ∈ {S, N6, N7}, zasięg b − a ≤ k − 1):
        jeśli nie granica_ok(a, b, L, indeks bloku): pomiń     # filtr L3, bezstratny
        jeśli b − a ≥ 2 i nie wykonalny(a, b, typ):   pomiń     # T2: certyfikat głów, potem DFS
        K ← K ∪ {(s+a, s+b, typ, Ĉ(a, b, typ) − C_max)}        # estymata O(b − a)
wybór ← DP(K) (rodziny laminarne) | skojarzenie/MILP/QUBO z karami par (krzyżowania)
jeśli wybór = ∅: wybór ← {argmin δ̂}                           # jak w PFSP
σ' ← zastosuj(σ, wybór); C' ← ocena(σ')                        # dokładnie, O(nm)
zwróć σ', C', id = {(P[i], P[j]) : (i, j) ∈ wybór}
```

Koszt iteracji:
- O(nm) dwie oceny;
- O(L) ścieżka i bloki;
- O(b·k²) estymaty dla k-bonacciego z filtrem;
- testy wykonalności: zwykle O(1) dzięki certyfikatowi, DFS rzadko;
- O(L + |K|) DP.

### 2.8 Wiele ścieżek krytycznych (opcja, poza rdzeniem)
JSSP ma płaskie krajobrazy: często kilka ścieżek krytycznych, więc wiele delt równa się 0.
- Warunek konieczny poprawy: ruch musi „trafić” każdą ścieżkę krytyczną.
- Rozszerzenie: sąsiedztwo na sumie bloków kilku ścieżek (podgraf krytyczny). DP na DAG-u zamiast na ciągu.
- Do zrobienia tylko, jeśli E1 pokaże, że plateau dominuje.

### 2.9 Flow shop jako przypadek szczególny (ρ = 0)
- Gdy wszystkie marszruty to 1 → 2 → … → m, maszyny wzdłuż każdej ścieżki są niemalejące.
  Ścieżka JS(x) ⇝ JP(y) między operacjami jednej maszyny nie istnieje, bo musiałaby się cofnąć z maszyny M + 1 do M − 1.
  Zatem (z T2) **każde przestawienie bloku jest wykonalne**. Pilot: 0 niewykonalnych przy ρ = 0.
- Sąsiedztwa JSSP na flow shopie dają harmonogramy **niepermutacyjne** (F||C_max). Rozwiązania PFSP są ich podzbiorem.
  Daje to most do poprzednich prac: na tai20_5 itd. możemy zmierzyć „zysk z niepermutacyjności” tym samym kodem (E4).

---

## 3. Wkład artykułu (C1–C6) i dowody

| # | Wkład | Czym udowodniony |
|---|---|---|
| C1 | **Konstrukcja:** rodzina sąsiedztw liczonych ciągami na wirtualnej permutacji ścieżki krytycznej; liczność jako iloczyn po blokach; spłaszczenie pod filtrem (T10). Na tej podstawie **rodzina ruchów granicznych** (typ × zasięg × złożenie), w której N5, N6, N7, Fib, k-bonacci i Dyn są punktami jednej przestrzeni | T1, T10, §2.3–2.4 |
| C2 | **Twierdzenie o przestawieniach bloków** i wnioski: N1, rozkład dla rodzin laminarnych, krzyżowania jako ograniczenia dwuciałowe (dokładnie kwadratowe), wykonalność wstawień, flow shop bez ograniczeń | T2, T3, T8 + brute force (§14.1) |
| C3 | **Filtr granic dla ruchów złożonych** (bezstratny w modelu addytywnym, A.7) + niezmienniczość DP; estymata dokładna dla d = 1 | T4, T5 |
| C4 | **Anatomia i kontinuum flow → job:** długości bloków, kandydaci, niewykonalność i korzyść z zasięgu w funkcji n/m i ρ | H2, H5 (E1, E4) |
| C5 | **Studium empiryczne:** ILS/SA z każdą rodziną na Taillardzie JSSP, porównanie z N1/N5/N6/N7, addytywność | H1, H3, H4, H6 (E2, E3, E5, E6) |
| C6 (opc.) | **Kwanty:** blokowo-diagonalne, małe QUBO JSSP; IQM bez okien; kary wykonalności dla krzyżowań | H7 (E7) |

---

## 4. Część teoretyczna: twierdzenia do udowodnienia

| # | Treść | Status |
|---|---|---|
| T1 | **Liczność:** \|N_F(P)\| = Π_b a_F(L_b) dla F ∈ {Fib, k-bonacci, Dyn, Motzkin, inwolucje}; wersja z filtrem | trywialne; liczby w §14.2 |
| **T2** | **Twierdzenie o przestawieniach bloków:** niech σ' zmienia kolejność operacji tylko wewnątrz bloków jednej ścieżki krytycznej rozwiązania wykonalnego σ (dowolna permutacja zbioru pozycji każdego bloku). σ' jest niewykonalne ⟺ istnieje blok B i operacje x, y ∈ B takie, że x poprzedza y w σ, y poprzedza x w σ', i w G(σ) istnieje **E-ścieżka** z x do y, czyli ścieżka x → JS(x) ⇝ JP(y) → y, która **nie przechodzi przez inne operacje bloku B** (dopuszczalne JS(x) = JP(y)) | **dowód (szkic poniżej)**; brute force: 0 / 113 781 (tasowania fragmentów, inwolucje) i **0 / 70 680 pełnych tasowań bloków** (wersja ścisła) ✅ |
| T3a | Zamiana sąsiednich operacji krytycznych jest wykonalna (van Laarhoven i in. 1992) | wniosek z T2 (blok bez przestoju); 0 / 84 203 ✅ |
| T3b | **Rodziny laminarne** (przedziały rozłączne lub zagnieżdżone): zbiór odwróconych par ruchu złożonego to suma zbiorów jego zamian. Zatem ruch jest wykonalny ⟺ każda zamiana jest wykonalna osobno. Obejmuje k-bonacci, Dyn i Motzkina | wniosek z T2; 0 / 169 118 (rozłączne), 0 / 2,55 mln par i 0 / 67 746 zbiorów (zagnieżdżone) ✅ |
| T3c | **Krzyżowania** zmieniają zbiór odwróconych par: pojawiają się nowe, a niektóre znikają. Wykonalne składowe mogą dać niewykonalny ruch (0,71% par krzyżujących się, 9,5% zbiorów z krzyżowaniami), a niewykonalna składowa może zostać „uratowana” (174 przypadki) | wniosek z T2; brute force ✅ |
| T3d | **Wykonalność jako kara kwadratowa:** każda zakazana para (x, y) zależy od ≤ 2 zamian (tych, które ruszają x i y). Wskaźnik naruszenia jest więc wielomianem stopnia 2 w zmiennych QUBO (przy ograniczeniu „każda operacja w ≤ 1 zamianie”). Dokładne kodowanie inwolucji z wykonalnością w QUBO | do spisania (krótkie) |
| T4 | **Filtr granic**: ruch, który nie zmienia żadnej (właściwej) granicy bloku, zostawia ścieżkę o tej samej długości, więc nie poprawia (twierdzenie blokowe). Wniosek: **addytywne** DP na kandydatach przefiltrowanych ma tę samą wartość optymalną. Uwaga (A.7, usterka 2): filtr nie jest bezstratny dla prawdziwego C_max ruchu złożonego | dowód krótki; 0 / 40 702 i 0 / 6 400 ✅ |
| T5 | **Estymata:** dla d = 1 to dokładne dolne ograniczenie, równe nowemu C_max, gdy ruch nie poprawia (Taillard 1994). Dla d ≥ 2 przybliżenie | d = 1 ✅ (0 / 84 203); d ≥ 2: E0 |
| T6 | **Koszt iteracji** jak w §2.7; liczba kandydatów ≤ Σ_b 2(k − 1) | do spisania; zmierzyć w E0 |
| T7 | **Rozkład na bloki:** problem wyboru (DP / QUBO) to suma niezależnych problemów bloków; QUBO blokowo-diagonalne | trywialne |
| T8 | **Flow shop:** przy identycznych marszrutach każde przestawienie bloku jest wykonalne | wniosek z T2; pilot: 0 niewykonalnych przy ρ = 0 ✅ |
| T9 (opc.) | **Addytywność w JSSP:** ujęcie max-plus jak T8 k-bonacciego; delty zamian z różnych bloków oddziałują przez ścieżki omijające bloki; warunek wystarczający addytywności | do zrobienia, jeśli E5 da ciekawy obraz |
| T10 | **Spłaszczenie rodzin pod filtrem:** po filtrze granic blok ma ≤ 2 zamiany (początek / koniec). Zagnieżdżenie jest niemożliwe (Motzkin ≡ Dyn przy addytywnym DP), a krzyżowanie tylko jako jedna para graniczna (c ≤ 1). Liczność w bloku wielomianowa (§2.3), w całym sąsiedztwie wykładnicza w liczbie bloków | dowód trywialny (z T4); ważny dla interpretacji i doboru wariantów |

**Uwaga (runda krytyki 1): wersja ścisła jest konieczna.**
- Wersja „luźna” (dowolna ścieżka JS(x) ⇝ JP(y), także przez inne operacje B łukami maszynowymi) dała **1 fałszywy alarm na 70 680** pełnych tasowań bloków. Wersja ścisła: 0.
- Dla pojedynczej zamiany, pojedynczego wstawienia i rodzin laminarnych obie wersje są równoważne:
  pierwszy fragment ścieżki od przenoszonej operacji jest zawsze „czysty”.
  Pilot (§14) i test dla pojedynczych zamian (§2.6) są więc poprawne.
- Dla krzyżowań i dowolnych przestawień implementacja musi używać wersji ścisłej (DFS z zakazem wejścia w operacje bloku).

**Szkic dowodu T2.**
- (⇐) Ścieżka x → JS(x) ⇝ JP(y) → y nie używa relacji zmienionych przez σ', więc istnieje w G(σ'). Razem z relacją maszynową y → x daje cykl.
- (⇒) Cykl w G(σ') musi zawierać zmienioną relację, bo reszta grafu jest jak w G(σ). Dzielimy go na kroki „N” i „E”:
  - kroki N to relacje maszynowe między operacjami bloków w nowej kolejności;
  - kroki E to fragmenty przez operacje spoza bloków; istnieją w G(σ).
- Krok E nie może iść wstecz po P (z blokiem P dałby cykl w G(σ)), a krok N nie wychodzi poza blok. Zatem cykl leży w jednym bloku.
- Scalamy kolejne kroki N (porządek na maszynie jest przechodni): a_1 ≺' b_1 ⇝_E a_2 ≺' b_2 ⇝_E … ⇝_E a_1.
- Gdyby b_i ≺' a_{i+1} dla każdego i, porządek ≺' zawierałby cykl. Zatem dla pewnego i: a_{i+1} ≺' b_i.
  Wtedy E-ścieżka b_i ⇝ a_{i+1} to szukana odwrócona para. ∎
- E-ścieżka między operacjami jednej maszyny wychodzi przez JS i wchodzi przez JP. Łuki maszynowe prowadzą tylko do operacji z tej samej maszyny,
  a przed lub za blokiem dałyby cykl w G(σ).

---

## 5. Hipotezy badawcze

### 5.1 Zestawienie

| ID | Hipoteza (skrót) | Eksperyment | Test | Kryterium potwierdzenia | Dane wstępne (pilot) |
|---|---|---|---|---|---|
| H0 | Implementacja poprawna (T2–T5, liczności, DP, estymata) | testy, E0 | brute force | 100% zgodności | ✅ prototyp |
| H1 | Przy zejściu do optimum lokalnego k ≥ 3 daje lepsze optima niż N5; złożony Fib jest „zbyt zachłanny” | E2 | Wilcoxon + Holm, Page (k) | k ≥ 3 istotnie lepsze od N5 w ≥ 6/8 klas | ✅ 28,4–28,6% vs 30,8%; Fib 32,8% |
| H2 | Korzyść z zasięgu k rośnie z długością bloków (n/m): przy 15×15 i 20×20 hierarchia się spłaszcza, przy 50×15 i 100×20 się rozwiera | E1, E3 | Spearman (k_sat, L̄), regresja k* ~ n/m | ρ_S ≥ 0,5; k* rośnie z n/m | bloki: 2,4 (15×15) → 25 (100×20) |
| H3 | Niewykonalność rośnie z odległością ruchu d, z n/m i z jakością rozwiązania; test z T2 kosztuje < 10% czasu iteracji | E0, E1, E4 | Page (d), pomiar czasu | trend istotny; udział czasu < 10% | ✅ losowe: 0,2% (d = 2) → 4,9% (d ≥ 6). Po ILS (k = ∞): 19% (15×15), 45% (30×15), 57% (50×15), 67% (100×20) kandydatów |
| H4 | Addytywność zawodzi silniej niż w PFSP; realny zysk ruchu złożonego to mniej niż połowa sumy delt | E5 | Wilcoxon (realny/przewidywany) | mediana < 0,5 | ✅ 36–43% (dokładne delty), 14–19% (estymata) |
| H5 | **Kontinuum (przeformułowane):** przewaga ruchów złożonych nad N5 jest **niemonotoniczna w ρ**: brak przy ρ = 0 (flow shop), największa przy pośrednim ρ (0,1–0,5), a przy ρ = 1 zależna od n/m (przy n/m ≥ 4 instancje są łatwe: ścieżka = 1 blok, C_max = LB) | E4 | Friedman w każdym ρ, interakcja ρ × wariant (model mieszany) | istotna interakcja; przewaga maksymalna przy pośrednim ρ | ✅ pilot §14.3b: przy ρ = 0 N5 najlepsze we wszystkich 4 rozmiarach; przy ρ = 0,1–0,5 N5 traci 4–22 pp, a złożone 0,4–7,5 pp |
| H6 | W ILS (równy czas) Fib i k-bonacci są lepsze od N5 i N1; najlepsze k jest umiarkowane (3–4) | E3 | Friedman + Nemenyi (CD), Wilcoxon + Holm | k ∈ {3, 4} istotnie lepsze od N5 w ≥ 5/8 klas | ✅ 1 s: k = 3 7,5%, Fib 8,3%, N5 9,0%, N1 11,3% |
| H7 | **Pozycja względem odniesień:** najlepsza rodzina w ILS nie jest gorsza od ILS-N6. Do i-TSAB / TS/PR raportujemy uczciwy dystans | E6 | Wilcoxon + Holm | jak w hipotezie | — (ryzyko: N6 bywa silniejsze) |
| H8 | **Kwanty:** QUBO JSSP mieści się bez okien (≤ 20 zmiennych dla k = 2 do 30×20), a jakość ruchu z QPU jest bliska DP | E7 | odsetek ruchów optymalnych | ≥ 90% optymalnych dla k = 2 na IQM | rozmiary ✅ (§14.2) |
| H9 | **Złożenie ważniejsze niż typ:** w rodzinie ruchów granicznych wersje złożone są lepsze od pojedynczych dla każdego typu (S, N6, N7). Przy złożeniu różnice między typami są małe | E3 | Friedman + Nemenyi; Wilcoxon + Holm (c6 vs N6, c7 vs N7, k = 3 vs N5) | złożone istotnie lepsze od pojedynczych w ≥ 5/8 klas | ✅ pilot: c6 7,5 vs N6 8,4; c7 7,5 vs N7 8,9; k = 3 7,4 vs N5 10,6 (§14.5b) |

### 5.2 Karty hipotez

**H1. Zasięg pomaga już przy zejściu; złożony Fib jest zbyt zachłanny**
- *Treść:* z tych samych startów zejście best-improvement z k ≥ 3 kończy w lepszych optimach niż N5. Fib (złożone N5) kończy w gorszych.
- *Uzasadnienie:* przy k ≥ 3 dochodzą zamiany (0, b) i (a, L − 1), czyli przeniesienia pierwszej lub ostatniej operacji bloku głębiej.
  Ruchy N5 są zawarte (L3).
  Fib stosuje naraz kilka poprawiających zamian N5, a addytywność zawodzi (H4). Dochodzi więc szybciej (6,6 vs 16,9 kroku), ale do gorszego atraktora.
- *Pilot:* 15 instancji × 4 starty. N1 = N5 (30,8%; zgodnie z L3 te same optima). Fib 32,8%, k = 3 28,5%, k = 4 28,4%, k = 8 i Dyn 28,6%.
  k = 4 jest lepsze od N5 w 39 parach, gorsze w 17 (na 60).
- *Jeśli odrzucona:* przekaz „ruchy złożone pomagają tylko w ILS” (H6). Wtedy pytanie, dlaczego, prowadzi do H4.

**H2. Skala bloku wyznacza opłacalny zasięg**
- *Treść:* k* (najlepsze k przy budżecie) i k_sat (nasycenie zysku) rosną ze średnią długością bloku L̄, a ta z n/m.
- *Uzasadnienie:* filtr zostawia ≤ 2(k − 1) kandydatów na blok, a dalekie zamiany istnieją tylko w długich blokach.
  Pilot (optima N5): L̄ = 2,4 (15×15), 2,9 (20×20), 3,8 (30×20), 7,7 (30×15), 7,9 (50×20), 12,4 (50×15), 25,2 (100×20).
- *Pomiar:* E1 (stany z trajektorii ILS), E3 (k* w klasach).
- *Teza w artykule (EN):* “The useful reach of a compound move is set by the block length, which grows with n/m.”

**H3. Wykonalność kosztuje mało, ale rośnie z zasięgiem**
- *Treść:* odsetek niewykonalnych zamian rośnie z d. Test z T2 (certyfikat + DFS) zajmuje < 10% czasu iteracji.
  Rodziny laminarne nie płacą za kombinacje (T3b).
- *Pilot:* d = 2: 0,21%; d = 3: 1,35%; d = 4: 2,8%; d = 5: 3,9%; d ≥ 6: 4,9% (losowe instancje).
  Na Taillardzie przy k = ∞: 5–29% kandydatów w optimach N5.
  W **dobrych rozwiązaniach (po 2 s ILS)** 19% (15×15), 45% (30×15), 57% (50×15) i **67% (100×20)**. Przy k ≤ 4 to tylko 0–5%.
- *Konsekwencja:* im lepsze rozwiązanie i dłuższe bloki, tym więcej dalekich ruchów jest niewykonalnych.
  To drugi, obok kosztu oceny, argument za umiarkowanym k. Wyjaśnia też, dlaczego zysk z zasięgu może się nasycać wcześniej niż w PFSP.

**H4. Addytywność w JSSP**
- *Treść:* przewidywany zysk (suma delt) przecenia realny bardziej niż w PFSP.
  Estymata dodatkowo zawyża zysk pojedynczego ruchu (jest dolnym ograniczeniem C_max).
- *Pilot (stany od losowych do bliskich optimum, ~100 ruchów na rodzinę):*
  - średnio ~10 zamian w ruchu;
  - realny zysk to 36–43% przewidywanego (dokładne delty) i 14–19% (estymata);
  - ruch prawie zawsze poprawia (2–4% nie poprawia).
- *Wariant zaradczy (E5):* top-t ruchów z DP oceniane dokładnie, wybór najlepszego. Albo „DP z budżetem zamian” (≤ s zamian w ruchu).

**H5. Kontinuum flow shop → job shop (centralna nowa oś; przeformułowana po rundzie krytyki 1)**
- *Generator:* Taillard z parametrem ρ, czyli prawdopodobieństwem przestawienia kroku marszruty (ρ = 0: wspólna marszruta, F||C_max; ρ = 1: Taillard JSSP).
  Czasy p_ij są wspólne dla wszystkich ρ (te same ziarna), więc porównanie jest sparowane.
- *Treść (wersja 2):*
  - (a) niewykonalność dalekich ruchów wynosi 0 przy ρ = 0 (T8) i jest największa przy pośrednim ρ;
  - (b) przewaga ruchów złożonych nad N5 jest **niemonotoniczna**: brak przy ρ = 0, maksymalna przy ρ ≈ 0,1–0,5, przy ρ = 1 zależna od n/m.
- *Wersja 1 odrzucona:* „liczba poprawiających dalekich zamian maleje z ρ” była artefaktem słabych stanów (A.6, runda 1).
- *Pilot wydajnościowy (§14.3b; ILS 1 s, estymata, 3 instancje × 2 ziarna; RPD do najlepszego wyniku dowolnego wariantu na instancji):*
  - ρ = 0: N5 najlepsze we wszystkich 4 rozmiarach (0,6–2,2%); warianty złożone tracą 0,1–1,8 pp;
  - ρ = 0,1–0,5: N5 4,4–22,1%, a najlepszy wariant złożony 0,4–4,4% (30×15: N5 14–19% vs c6 2,2–4,1%);
  - ρ = 1: przy 20×5 i 50×10 wszystkie warianty (prawie) zawsze trafiają LB (ścieżka to jeden blok); przy 20×10 i 30×15 N5 5,7–5,8% vs c6 1,6%.
- *Możliwe wyjaśnienie (do sprawdzenia w E4/E1):*
  - przy ρ = 0 bloki są krótkie i liczne (3–8 operacji, 5–15 bloków), wszystko jest wykonalne, a N5 robi 2–4× więcej iteracji niż c6/Dyn;
  - przy pośrednim ρ marszruty są „prawie przepływowe”, ale z zaburzeniami: bloki się wydłużają (do 20–24), a ruchy dalsze niż N5 są potrzebne do ich przebudowy.
- *Most do PFSP:* przy ρ = 0 te same instancje (np. tai20_5) liczymy algorytmem PFSP i algorytmem JSSP. Różnica to zysk z niepermutacyjności.
  Do tego sprawdzenie, czy ranking N5 > złożone przy ρ = 0 nie jest artefaktem kodowania flow shopu jako JSSP.
- *Teza w artykule (EN):* “Compound border moves pay off in between: not in the pure flow shop, most under mild route disorder, and less as instances become easy.”

**H6. ILS przy równym czasie**
- *Pilot (1 s, 8 instancji × 3 ziarna, dokładne delty):* RPD do BKS: N1 11,3; N5 9,0; Fib 8,3; k = 3 7,5; k = 4 7,6; Dyn 7,7.
  Iteracje: 737 / 1249 / 1312 / 882 / 828 / 642.
- *Ze estymatą (D10):* N5 7,3; Fib 6,2; k = 3 5,1; k = 4 5,2; Dyn 5,4. Na trzech większych instancjach: N5 18,8; Fib 16,2; k = 3 13,3; Dyn 12,3.
  Na ta51 (50×15, długie bloki) wygrywa Dyn. To zgodne z H2, ale to jedna instancja, więc tylko przesłanka.
- *Ryzyko:* 1 s i 3 ziarna to za mało na wnioski. Pełny test w E3 (10 s, 5 ziaren, ekstrakcja).

**H7. Uczciwa pozycja**
- i-TSAB i TS/PR to dopracowane TS z pamięcią długoterminową. Nasze ILS z kickiem nie będzie z nimi konkurować.
- Raportujemy dystans do BKS i porównanie z ILS-N6 przy tym samym szkielecie: **wkładem jest konstrukcja sąsiedztw, nie metaheurystyka**.

**H8. Kwanty** opisane w §A.5 i E7.

**H9. Złożenie ważniejsze niż typ ruchu (dodane po rundzie krytyki 1)**
- *Treść:* w każdym typie ruchu granicznego (zamiana, N6, N7) wersja złożona (≤ 1 ruch na koniec bloku, wszystkie bloki naraz) jest lepsza od pojedynczej.
  Przy złożeniu wybór typu ma drugorzędne znaczenie.
- *Uzasadnienie:* ruch złożony skraca naraz kilka bloków tej samej ścieżki krytycznej, a pojedynczy tylko jeden.
  T3b gwarantuje, że złożenie nie kosztuje dodatkowych testów wykonalności.
- *Pilot (§14.5b):* c6 vs N6: 7,53 vs 8,41; c7 vs N7: 7,50 vs 8,90; k = 3 vs N5: 7,40 vs 10,62. Typy w czołówce: k = 3 7,40, bm3 7,42, c7 7,50, c6 7,53.
- *Jeśli odrzucona* (np. N6 > c6 w dłuższych budżetach): wartość złożenia zależy od budżetu. Wtedy wykres w funkcji budżetu (ekstrakcja z 10 s) staje się centralny.
- *Teza w artykule (EN):* “Compounding matters more than the move type: every border move gains from being applied in all blocks at once.”

---

## 6. Eksperymenty

K, czyli badane warianty (OPT): N1, N5, Fib, k ∈ {3, 4}, Dyn, N6, N7, c6, c7, bm3.
W MAX dochodzą: k ∈ {6, 8}, bm4, bm∞, Inw, Motzkin i ≤ c krzyżowań bez filtra (c ∈ {1, 2, ∞}), TS-N5/TS-c6.
Budżety 0,1–10 s (ekstrakcja z 10 s), 5 ziaren, ta01–ta80.

### E0. Walidacja i model kosztu (H0, H3; T2–T6), ~1 h CPU, chmura
- Testy jednostkowe z prototypu: L1–L5, T2 (dowolne przestawienia), T3c, T8. Brute force dla n ≤ 10, m ≤ 5.
- Liczności T1 brute force dla bloków L ≤ 9.
- Jakość estymaty dla d ≥ 2:
  - korelacja Spearmana δ̂ z δ;
  - odsetek fałszywie poprawiających;
  - wpływ na wybór DP (ten sam ruch?).
- Mikro-benchmark czasu iteracji: ocena, ścieżka, estymaty, wykonalność, DP, ocena końcowa; 8 klas × warianty × 30 stanów.
- **Do artykułu:** tabela kosztu iteracji, liczby z T2.

### E1. Anatomia na stanach z trajektorii ILS (H2, H3), ~2 h CPU
- Stany: 8 klas × 10 instancji × 20 stanów (losowe momenty ILS z k = 4) + optima lokalne N5.
- Na każdym stanie dla każdego wariantu:
  - długość ścieżki, liczba i długości bloków;
  - kandydaci, niewykonalni (według d), poprawiający, zerowe delty (plateau);
  - liczba ścieżek krytycznych (próbka).
- **Do artykułu:** rozkład długości bloków w klasach; kandydaci i niewykonalność w funkcji k.

### E2. Zejście do optimum lokalnego (H1), ~1 h CPU
- Start z round-robin + 10 losowych startów na instancję; best-improvement do braku poprawy.
- Metryki: RPD, liczba kroków, czas. Test Page’a dla k i Wilcoxon względem N5.

### E3. Kampania główna (H2, H6), opcje z A.4
- ILS (+ SA w OPT), warianty K, protokół D4–D8. Ekstrakcja budżetów, walidacja ekstrakcji (jak k-bonacci: 2 klasy, k ∈ {2, 4}, osobne przebiegi).
- Analiza:
  - Friedman + Nemenyi (CD) w każdej klasie i globalnie;
  - Wilcoxon + Holm dla par kluczowych: N5 vs Fib vs k = 3 vs najlepsze k; N6 vs c6, N7 vs c7 (H9); najlepszy złożony vs N6;
  - k* w funkcji klasy (n/m);
  - liczba ruchów.
- **Do artykułu:** tabela główna (klasa × wariant, RPD do BKS), diagram CD, wykres k*(n/m).

### E4. Kontinuum flow shop → job shop (H5), ~4,6 h
- Generator Taillarda z ρ ∈ {0, 0,1, 0,25, 0,5, 1}, rozmiary 20×5, 20×10, 30×15, 50×10, 10 instancji (ziarna z jednego generatora), 3 ziarna.
- Warianty N5, k = 3, Dyn, N6, c6; ILS; 5 s.
- Miara: RPD do najlepszego wyniku dowolnego wariantu na instancji (LB przy małym ρ jest słabe) i, pomocniczo, RPD do LB.
- Analiza:
  - interakcja ρ × wariant (Friedman w każdym ρ, model mieszany);
  - anatomia (E1) w każdym ρ, na stanach po ILS (nie na optimach N5, A.6).
- Kontrola: przy ρ = 0 te same instancje także w kodzie PFSP, żeby wykluczyć artefakt kodowania flow shopu jako JSSP.
- **Most do PFSP:** przy ρ = 0 dodatkowo tai20_5, tai20_10 i tai50_10 (oryginalne instancje PFSP), algorytm PFSP vs JSSP (zysk z niepermutacyjności).
- **Do artykułu:** rysunek „ρ na osi x”: RPD wariantów, niewykonalność, poprawiające dalekie zamiany.

### E5. Addytywność i wybór świadomy addytywności (H4), ~3 h
- Pomiar realny/przewidywany na stanach z E1 dla każdego wariantu (dokładne delty vs estymata).
- Warianty zaradcze w ILS:
  - (a) top-t (t ∈ {3, 5}) dokładnie oceniane;
  - (b) limit zamian w ruchu s ∈ {1, 2, 4};
  - (c) wybór tylko jednego bloku (najlepszy blok).
- Klasy 15×15, 30×15, 50×15.

### E6. Punkty odniesienia (H7); N6/N7 są już w E3, w MAX dochodzą TS i zestawy klasyczne
- N6 i N7 z dokładną wykonalnością z T2. Porównać z warunkami Balasa–Vazacopoulosa: ile ruchów N6 odrzucają niepotrzebnie (to tani, dodatkowy wynik dla T2).
- TS-N5 i TS-c6 (TSAB-light: tabu na odwróconych łukach, najlepszy ruch nie-tabu, aspiracja): pokazuje, czy ranking sąsiedztw z ILS przenosi się na TS.
- Zestawy klasyczne (ft, la, orb, abz, swv, yn): tabela vs BKS i krótkie porównanie z wartościami z literatury (bez powtarzania cudzych metaheurystyk).

### E7. Kwanty (H8), po klasyce
- D-Wave: QUBO całego ruchu (blokowo-diagonalne). Reżim jednego ruchu jak w `windowed_qubo`; ILS + SA; k ∈ {2, 3, 4}.
- IQM: QAOA p = 1, 4096 shotów; ta01–10, k = 2 (i 3).
- Odra 5 (5 kubitów): bloki osobno, k ≤ 3.
- Inwolucje z karami wykonalności (T3d) na symulatorze. To pierwsze QUBO z prawdziwymi ograniczeniami dwuciałowymi w naszym programie.

### Mapa: hipoteza → eksperyment → materiał w artykule

| H | E | Materiał | Sekcja |
|---|---|---|---|
| H0 | E0 | liczby brute force, tabela kosztu | 4, 6.1 |
| H1 | E2 | tabela zejścia | 6.2 |
| H2 | E1, E3 | rozkład bloków, k*(n/m) | 6.3 |
| H3 | E0, E1 | niewykonalność vs d i k | 6.3 |
| H4 | E5 | realny/przewidywany, warianty zaradcze | 6.5 |
| H5 | E4 | rysunek kontinuum | 6.4 |
| H6 | E3 | tabela główna, CD | 6.6 |
| H7 | E6 | N6/N7, klasyka | 6.6 |
| H8 | E7 | rozmiary QUBO, jakość ruchu | 7 lub osobny artykuł |
| H9 | E3 | mapa rodziny ruchów granicznych (typ × zasięg × złożenie): tabela i diagram CD | 6.6 |

---

## 7. Protokół i trafność wyników

**7.1 Uczciwość porównań**
- Wszystkie warianty idą tą samą ścieżką kodu. Fib to `js_kbonacci_k2`, Dyn to `js_kbonacci_kn`.
  Ten sam start, ten sam kick, to samo tabu, ten sam filtr i ta sama estymata.
- N5 to nie osobna implementacja, tylko k = 2 z wyborem pojedynczego ruchu. N1 to ten sam kod bez filtra.
- Remis na ścieżce krytycznej jest rozstrzygany jedną regułą dla wszystkich (D9). Ablacja w MAX.

**7.2 Pomiar czasu** jak w k-bonaccim §7.2: jedna maszyna, losowa kolejność przebiegów, liczba iteracji zawsze zapisana.

**7.3 Dane i wartości odniesienia**
- ta01–ta80 zweryfikowane generatorem (0.1).
- BKS z datą i źródłem (0.2). JSPLIB jest nieaktualny: np. ta11 ma tam przedział 1323–1361, a w literaturze są nowsze domknięcia.
- RPD do LB (max obciążenia maszyny / długości zadania) dodatkowo, dla ciągłości z PFSP.
  Dla ta51–ta80 LB często równa się optimum (pilot: ta51 LB = 2760 = optimum).

**7.4 Statystyka** jak w k-bonaccim §7.4:
- α = 0,05, korekta Holma; bloki = instancje, wartości uśrednione po ziarnach;
- Friedman + Nemenyi (CD); test Page’a dla trendów w k i ρ.

**7.5 Reprodukowalność**
- Metadane każdego startu etapu (commit, maszyna, Python, stałe), jak w `campaign_meta.jsonl`.
- Skrypty E0–E7 w repo.
- Anonimowe repozytorium dla recenzentów, jeśli konferencja.

**7.6 Zagrożenia trafności**
- *Wewnętrzne:*
  - Python zaniża liczbę iteracji, szczególnie przy 100×20 (ocena ~3 ms);
  - raportujemy liczbę iteracji i koszt iteracji (E0);
  - numba opcjonalnie (P2).
- *Zewnętrzne:*
  - tylko C_max i tylko Taillard w tabeli głównej;
  - klasyka w MAX;
  - ΣwT (Kuhpfahl, Bierwirth) jako dyskusja.
- *Konstrukcyjne:* RPD zależy od BKS, więc podajemy źródło i datę.

---

## 8. Struktura artykułu (czasopismo ~20 stron; wersja konferencyjna GECCO: 8 stron)

| § | Sekcja | Zawartość | Rysunki/tabele | Zależy od |
|---|---|---|---|---|
| 1 | Introduction | luka „rodzina sąsiedztw poza permutacją”, pytanie o przejście flow → job, wkład C1–C6 | — | całość |
| 2 | Background | JSSP, graf dysjunktywny, N1–N7, estymaty; rodzina PFSP (Fib, k-bonacci, Dyn, Motzkin) | — | §13 |
| 3 | Building neighborhoods on the critical path | wirtualna permutacja, liczność (T1), filtr (T4), spłaszczenie (T10), **rodzina ruchów granicznych** (§2.4), algorytm 1 (§2.7) | rys. 1 (ścieżka, bloki, ruch złożony na Gantcie), tab. 1 (mapa: typ × zasięg × złożenie, z N5/N6/N7/Fib/Dyn jako punktami) | teoria |
| 4 | Feasibility | T2 z dowodem; T3a–d; wstawienia; flow shop (T8) | rys. 2 (cykl przez E-ścieżkę) | teoria |
| 5 | Evaluation and additivity | estymata (T5), addytywność, warianty zaradcze | — | E0, E5 |
| 6 | Experiments | 6.1 setup; 6.2 zejście; 6.3 anatomia i k; 6.4 kontinuum ρ; 6.5 addytywność; 6.6 ILS/SA + odniesienia | rys. 3–5, tab. 2–4, CD | E1–E6 |
| 7 | Quantum note (opc.) | blokowo-diagonalne QUBO, IQM bez okien, kary wykonalności | tab. 5 | E7 |
| 8 | Conclusions | rekomendacja k, reguła z n/m, ρ | — | — |

---

## 9. Prace implementacyjne

| P | Zadanie | Dni | Uwagi |
|---|---|---|---|
| P0 | `fetch_jsplib.py`, `verify_taillard_jssp.py`, parser JSPLIB i formatu Taillarda | 0,5 | ziarna ta01–ta80 z Taillarda (1993) |
| P0 | `bks_jssp.json` (BKS + LB, źródła, data) | 0,5 | van Hoorn 2018 + nowsze (Coupvent des Graviers i in. 2025) |
| P0 | `src/jobshop/core.py`: instancja, ocena, r/q, ścieżka, bloki | 1 | z prototypu |
| P0 | `feasibility.py` + `estimate.py` + testy (L1–L5, T2, T8) | 1 | z prototypu |
| P0 | rodzina ruchów granicznych: `js_n1`, `js_n5`, `js_kbonacci_k<K>`, `js_kbonacci_kn`, `js_n6`, `js_n7`, `js_c6`, `js_c7`, `js_bm<K>` | 1,5 | `solve_dp` wspólne; jeden generator kandydatów (typ × zasięg), jedna estymata (dowolna nowa kolejność przedziału), jeden test wykonalności (T2) |
| P0 | adapter problemu w ILS/SA + **test regresji PFSP bit w bit** | 1 | najważniejszy test etapu 0 |
| P0 | runner + `run_jobshop_campaign.py` + `analyze_jobshop.py` + `RUN_LOCAL.md` | 1 | jak k-bonacci |
| P1 | `js_inw` (krzyżująca się para graniczna, test pary T3c) + TS-light dla E6 (MAX) | 1 | Inw i TS tylko w MAX |
| P1 | generator z ρ + skrypt E4 + most PFSP (ρ = 0 na tai*) | 0,5 | H5 |
| P1 | warianty zaradcze addytywności (top-t, limit zamian) | 0,5 | E5 |
| P2 | Motzkin i ≤ c krzyżowań **bez filtra** (ablacja: DP przedziałowe, skojarzenie z zakazami / MILP, kary T3d, ocena top-t) | 1,5 | w porozumieniu z planem krzyżowań |
| P2 | QUBO JSSP (asemblery z `common_qubo`) + D-Wave / IQM / Odra 5 | 1 | E7 |
| P2 | numba dla oceny i estymat | 1–2 | go/no-go po E0 |
| P2 | wiele ścieżek krytycznych (§2.8) | 1 | tylko gdy E1 pokaże dominację plateau |

**Łącznie: P0 ~6,5 dnia, P1 ~2 dni, P2 ~4,5–5,5 dnia.**

---

## 10. Harmonogram i punkty kontrolne

> **Daty w tym rozdziale są nieaktualne od 11.11.2026.** Obowiązuje tabela z A.7. Bramki G0–G3 i ich warunki zostają.

Kontekst terminów:
- ICAISC 15.11 (Hamming);
- ICAPS 7.12 (abstrakt) i 14.12 (artykuł k-bonacci);
- ANNSIM 18.12 (QAOA p);
- IJCAI ~11–15.01 (zapas k-bonacciego);
- CPAIOR 2027 ~1–7.12.2026: za wcześnie dla JSSP;
- GECCO 2027: termin jeszcze nieogłoszony; w 2026 było to 19.01 (abstrakt) i 26.01 (artykuł).

Zasoby, które się zderzają:
- **Mac:** kampania k-bonacciego zajmuje go od ~2.11. ILS to ~22 h, z SA do ~44 h, więc Mac jest wolny najwcześniej ~5–8.11.
  Dodatkowe przebiegi pod ICAPS mają pierwszeństwo do ~1.12.
- **Uwaga użytkownika:** ICAISC 15.11, ICAPS 7.12 i 14.12, ANNSIM 18.12. JSSP nie może zabierać czasu pisania przed 19.12.
  Etap 0 i kampania idą bez udziału użytkownika (Claude w chmurze i lokalny Claude z `RUN_LOCAL.md`).
- **Quota D-Wave i kredyty IQM:** najpierw kwanty k-bonacciego (~225 s QPU, ~200 kredytów IQM). Kwanty JSSP dopiero w kolejnych miesiącach (Q1 2027).

**Harmonogram oparty na bramkach (zamiast sztywnych dat):**

| Bramka / okres | Zadania | Warunek przejścia dalej | Jeśli warunek niespełniony |
|---|---|---|---|
| **G0** (do ~8.11) | rozmowa z promotorem: D1 (JSSP w doktoracie?), D15; lektura nowości (Bożejko–Wodecki, Gröflin–Klinkert, Dauzère-Pérès–Paulli) | JSSP w zakresie; T2 i rodzina ruchów granicznych nieopisane w tej postaci | poza zakresem: tylko **plan B** (notatka teoretyczna, niżej). T2 znane: przesunąć akcent na T3/T10, rodzinę ruchów granicznych i kontinuum |
| Etap 0 (2–15.11, chmura) | P0 (§9) | testy L1–L5, T2, T8 zielone; **regresja PFSP bit w bit** | naprawić przed czymkolwiek na Macu |
| **G1** (~16–22.11, chmura) | E0, E1, E2 (tanie) | estymata dla d ≥ 2 porządkuje kandydatów (Spearman ≥ 0,7); koszt testu wykonalności < 10% iteracji | zmienić D10 (dokładne delty dla k ≤ 3) i przeliczyć czas kampanii |
| Kampania (najwcześniej ~23.11, najpóźniej ~20.12; Mac) | E3 + E4 (OPT ~34 h), bez udziału użytkownika | — | — |
| **G2** (po kampanii) | analiza | co najmniej jeden wariant złożony istotnie lepszy od N5 **i** nie gorszy od N6 w ≥ 5/8 klas | **plan B** + uczciwy wynik negatywny w rozprawie |
| **G3** (~20.12) | GECCO czy czasopismo? | H1, H6 i E4 rozstrzygnięte; ICAPS złożony bez opóźnień | czasopismo (bez terminu) |
| od 19.12 | pisanie §3–§6 (teorię §3–5 można szkicować wcześniej, jeśli starczy czasu) | — | — |
| ~26.01 (jeśli GECCO) albo luty–marzec (czasopismo) | zgłoszenie | — | — |
| Q1 2027 | E7 (kwanty), po kwantach k-bonacciego | wolna quota / kredyty | przesunąć |

**Plan B (minimalna jednostka publikacyjna):** krótka notatka teoretyczna (np. Operations Research Letters / Optimization Letters, 4–6 stron).
- Zawartość: T2 z dowodem, wnioski T3 (rodziny laminarne, krzyżowania, wstawienia), T8 (flow shop), T10 (spłaszczenie) i mały eksperyment MIN (~9 h Maca).
- Koszt: ~1 tydzień pisania, niezależnie od kampanii OPT.

---

## 11. Ryzyka i plan B

| Ryzyko | P-stwo | Skutek | Środek zaradczy |
|---|---|---|---|
| Promotor (Bożejko–Wodecki) już ma „ruchy złożone” w blokach JSSP | średnie | utrata części nowości | przeczytać w tygodniu 1. Nowość przesunąć na T2/T3 (wykonalność rodzin, krzyżowania jako kary dwuciałowe), kontinuum ρ i rodzinę liczoną ciągami |
| „To tylko N5 z DP” / „DP jest tu trywialne” | wysokie | słaba recenzja | przyznajemy to wprost (T7, T10). Wkład: rodzina ruchów granicznych (typ × zasięg × złożenie, §2.4), T2 z wnioskami (rodziny laminarne, krzyżowania, wstawienia, flow shop) i kontinuum |
| **T2 (lub jego istotna część) jest już znane** | średnie | słabsza teoria | sprawdzić w tygodniu 1: Gröflin i Klinkert (short cycle property, wykonalne wstawienia), Dauzère-Pérès i Paulli 1997, Mati i in. 2011, Brucker i Knust („Complex Scheduling”). Jeśli znane, cytujemy i akcentujemy wnioski: rodziny laminarne, krzyżowania jako kary dwuciałowe, flow shop, rotacje |
| Brak dostępu do części literatury z sesji (np. yadda.icm.edu.pl poza dozwoloną siecią) | pewne | opóźnienie oceny nowości | PDF-y od promotora albo dopuszczenie domeny w ustawieniach sieci sesji |
| ILS z kickiem słaby wobec SOTA | wysokie | słabsza pozycja | H7: uczciwa pozycja, porównanie z N6/N7 w tym samym szkielecie |
| Python: mało iteracji przy 100×20 | wysokie | zaszumione wnioski dla dużych klas | estymata zamiast dokładnych delt (D10), liczba iteracji w tabelach, numba (P2) |
| Addytywność psuje duże k | średnie | H6 słabsze | E5 (top-t, limit zamian); to też wynik |
| Plateau (wiele ścieżek krytycznych) | średnie | dużo ruchów z δ = 0 | §2.8, pomiar w E1 |
| Kolizja z ICAPS / ANNSIM / IJCAI | wysokie | brak czasu | etap 0 robi Claude w chmurze; Mac dopiero po k-bonaccim; czasopismo zamiast GECCO |
| BKS nieaktualne | średnie | błędne RPD | plik BKS z datą; RPD do LB obok |
| Front B poza projektem doktorskim | — | decyzja promotora | D1: przeformułować jako rozdział uogólniający |

---

## 12. Decyzje do podjęcia (z promotorem)

| # | Decyzja | Rekomendacja |
|---|---|---|
| D1 | JSSP w doktoracie? | tak, jako rozdział uogólniający (C1–C2 są ogólne dla każdego problemu z grafem dysjunktywnym) |
| D2 | Benchmark główny | ta01–ta80; klasyka w MAX |
| D3 | Miara | RPD do BKS + RPD do LB |
| D4 | Start | round-robin (odpowiednik identyczności); MWKR jako ablacja |
| D5 | Metaheurystyki | ILS + SA bez zmian (porównywalność z PFSP); TS-N5 (TSAB-light) tylko jako punkt odniesienia w MAX? |
| D9 | Reguła remisu na ścieżce | poprzednik maszynowy |
| D10 | Estymata vs dokładne delty | estymata + dokładna ocena ruchu |
| D11 | Zakres wariantów | OPTIMUM |
| D15 | Publikacja | czasopismo; GECCO tylko po punkcie kontrolnym |
| — | Relacja do prac promotora o ruchach złożonych | do ustalenia po lekturze (§13) |
| — | Konstrukcja główna | rodzina ruchów granicznych (§2.4) zamiast „samej” osi k; Motzkin i krzyżowania w JSSP tylko jako ablacja bez filtra (T10) |
| — | Kolejność względem planu krzyżowań | krzyżowania najpierw na PFSP. W JSSP oś c pod filtrem zawęża się do c ≤ 1 na blok (T10), więc w OPT jest tylko Inw; pełna oś bez filtra w MAX |

---

## 13. Literatura do przeczytania lub sprawdzenia

- **Pilne:**
  - Bożejko, Wodecki: tabu search z blokami i „compound moves technique”. Czy obejmuje JSSP i wykonalność ruchów złożonych?
  - Bożejko, Uchroński, Wodecki (2012): bloki ścieżki krytycznej, eliminacja.
  - Grabowski, Wodecki (2004, C&OR): bardzo szybki TS dla flow shopu.
  - **Nowość T2:**
    - Gröflin, Klinkert (2007, EJOR: wykonalne wstawienia, short cycle property; 2009, DAM: blokujący job shop);
    - Dauzère-Pérès, Paulli (1997, Ann. OR): klasyfikacja ruchów;
    - Mati, Dauzère-Pérès, Lahlou (2011, EJOR);
    - Brucker, Knust, „Complex Scheduling” (rozdział o sąsiedztwach i wykonalności).
- **Sąsiedztwa JSSP:**
  - van Laarhoven, Aarts, Lenstra 1992 (Oper. Res. 40(1));
  - Dell’Amico, Trubian 1993 (Ann. OR 41);
  - Nowicki, Smutnicki 1996 (Manag. Sci. 42(6)) i 2005 (J. Sched. 8);
  - Balas, Vazacopoulos 1998 (Manag. Sci. 44(2));
  - Zhang, Li, Guan, Rao 2007 (C&OR 34);
  - Kuhpfahl, Bierwirth 2016 (C&OR 66);
  - Mati, Dauzère-Pérès, Lahlou 2011 (EJOR): wykonalność ruchów dla kryteriów regularnych.
- **Ocena ruchu:** Taillard 1994 (ORSA J. Comput. 6(2)).
- **SOTA:** Peng, Lü, Cheng 2015 (C&OR 53); Nagata, Ono 2018 (C&OR 90); Constantino, Segura 2022.
- **Benchmarki i ograniczenia:**
  - instancje: Taillard 1993 (EJOR 64), Fisher i Thompson 1963, Lawrence 1984, Applegate i Cook 1991, Adams, Balas, Zawack 1988,
    Storer, Wu, Vaccari 1992, Yamada i Nakano 1992;
  - ograniczenia: van Hoorn 2018 (J. Sched. 21), Coupvent des Graviers i in. 2025 (arXiv 2504.16106);
  - repozytorium JSPLIB.
- **VLSN:** Congram, Potts, van de Velde 2002; Ahuja i in. 2002; Grosso, Della Croce, Tadei 2004.
- **Kwanty JSSP** (już cytowane w naszych pracach): Venturelli i in. 2015 (arXiv:1506.08479); studium VQA dla JSSP (cytowane w `qaoa_gcmm`).
- **Statystyka:** jak w k-bonaccim (Demšar 2006, Page 1963, Vargha i Delaney 2000).

---

## 14. Wyniki pilota (01.11.2026, chmura, Python 3, `scripts/jobshop_proto.py`)

Instancje pobrane z JSPLIB do `data/JSPLIB/instances` (poza gitem).

Pobranie danych (jednorazowo; P0 zastąpi to skryptem `fetch_jsplib.py`):

```bash
mkdir -p data/JSPLIB/instances && python3 -c "import json,urllib.request as u; b='https://raw.githubusercontent.com/tamy0612/JSPLIB/master/'; [u.urlretrieve(b+e['path'], 'data/JSPLIB/instances/'+e['name']) for e in json.load(u.urlopen(b+'instances.json'))]"
```

Odtworzenie pilota:

```bash
python scripts/jobshop_proto.py lemmas --reps 400 --seed 7
python scripts/jobshop_proto.py theorem --reps 160 --seed 21
python scripts/jobshop_proto.py anatomy --classes 15x15,20x15 --instances 5 --states 4
python scripts/jobshop_proto.py continuum --instances 4 --states 2
python scripts/jobshop_proto.py descent --files ta01,ta02,ft10 --starts 4
python scripts/jobshop_proto.py ils --files ta01,ft10 --seeds 3 --time-ms 1000
python scripts/jobshop_proto.py additivity --files ta01,ft10 --states 12
python scripts/jobshop_proto.py anatomy --classes 15x15,50x15 --instances 3 --states 2 --ils-ms 2000
python scripts/jobshop_proto.py continuum --instances 4 --states 2 --ils-ms 2000
python scripts/jobshop_proto.py continuum_ils --modes n5,k3,dyn,c6 --instances 3 --seeds 2 --time-ms 1000
python scripts/jobshop_proto.py ils --files ta01,ft10 --modes n5,k3,dyn,n6,n7,c6,c7,bm3,bminf --seeds 3 --time-ms 1000 --estimate
```

### 14.1 Lematy i twierdzenie (losowe instancje n ≤ 20, m ≤ 15, ρ ∈ {0; 0,5; 1}, stany losowe i optima lokalne N1)

| Własność | Przypadki | Naruszenia |
|---|---|---|
| L1 = T3a: zamiana sąsiednia krytyczna wykonalna | 84 203 | 0 |
| L2 = T3b: rozłączne przedziały, składowe wykonalne ⇒ ruch wykonalny | 169 118 ruchów | 0 |
| T3b: pary zagnieżdżone / zbiory typu Motzkin z zagnieżdżeniem | 2 547 971 / 67 746 | 0 / 0 |
| L3 = T4: zamiana poprawiająca dotyka właściwej granicy bloku | 40 702 | 0 |
| L5: DP z filtrem = DP bez filtra (dokładne delty) | 6 400 | 0 |
| T5: estymata d = 1 to dolne ograniczenie i jest dokładna, gdy ruch nie poprawia | 84 203 | 0 |
| **T2: dowolne przestawienia w 1–3 blokach** | 62 600 (977 niewykonalnych) | **0** |
| **T2: inwolucje (≥ 3 rozłączne zamiany, w tym 36 782 z krzyżowaniami)** | 51 181 (10 052 niewykonalnych) | **0** |
| **T2, wersja ścisła: pełne tasowania 1–2 bloków (≥ 4 operacje)** | 70 680 (17 036 niewykonalnych) | **0** (wersja luźna: 1 fałszywy alarm) |
| T2 dla ruchów granicznych (zamiany + wstawienia N6/N7) / złożenia rozłącznych | 24 967 (775 niewyk.) / 6 291 | 0 / 0 |
| T3c: pary krzyżujące się, obie wykonalne, razem niewykonalne | 18 196 / 2 578 173 | (0,71%) |
| T3c: zbiory z krzyżowaniami niewykonalne mimo wykonalnych składowych | 6 258 / 66 082 | (9,5%) |
| T3c: „ratowanie” (składowa niewykonalna, ruch wykonalny), tylko przy krzyżowaniach | 174 | — |
| Certyfikat głów (§2.6): błędne certyfikaty / pokrycie wykonalnych | 0 / 118 075; pokrycie 89,8% | 0 |

Niewykonalność pojedynczej zamiany w zależności od odległości d:

| d | 2 | 3 | 4 | 5 | ≥ 6 |
|---|---|---|---|---|---|
| % | 0,21 | 1,35 | 2,82 | 3,92 | 4,92 |

### 14.2 Anatomia na Taillardzie (optima lokalne N5 ze startów round-robin i losowych; 3–5 instancji × 2–4 stany na klasę)

| Klasa | L (ścieżka) | bloki | L̄ bloku | max blok | kand. k=2 | k=3 | k=4 | k=8 | k=∞ | niewyk. k=∞ | popr. k=∞ | log₁₀\|Fib\| | log₁₀\|Dyn\| | log₁₀\|Inw\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15×15 | 25 | 11,4 | 2,4 | 6,8 | 6,9 | 11,3 | 14,2 | 18,1 | 18,5 | 0,95 | 0,30 | 3,3 | 4,2 | 5,0 |
| 20×15 | 31 | 10,4 | 3,4 | 8,6 | 9,2 | 15,3 | 19,4 | 29,0 | 30,3 | 3,4 | 0,25 | 4,8 | 6,3 | 7,9 |
| 20×20 | 37 | 14,6 | 2,9 | 8,2 | 11,8 | 19,6 | 24,2 | 31,4 | 34,2 | 3,4 | 0,30 | 5,2 | 6,7 | 8,3 |
| 30×15 | 44 | 8,7 | 7,7 | 14,4 | 9,2 | 16,1 | 21,8 | 38,0 | 50,9 | 11,0 | 0,20 | 7,9 | 10,7 | 16,0 |
| 30×20 | 48 | 16,3 | 3,8 | 11,8 | 14,5 | 23,2 | 29,0 | 40,2 | 47,8 | 6,3 | 1,33 | 7,3 | 9,5 | 12,6 |
| 50×15 | 68 | 9,2 | 12,4 | 23,2 | 11,2 | 20,8 | 29,5 | 52,5 | 88,8 | 22,2 | 1,17 | 12,7 | 17,7 | 29,3 |
| 50×20 | 74 | 13,2 | 7,9 | 17,3 | 18,2 | 31,8 | 42,2 | 69,2 | 98,3 | 12,5 | 0,17 | 13,5 | 18,3 | 27,1 |
| 100×20 | 137 | 14,8 | 25,2 | 39,8 | 21,8 | 39,5 | 54,7 | 97,3 | 185,5 | 53,8 | 0,67 | 26,5 | 36,7 | 68,6 |

Wnioski:
1. Bloki wydłużają się z n/m: 2,4–3,8 przy n/m ≤ 1,5 i 7,7–25 przy n/m ≥ 2.
2. Kandydatów po filtrze jest **mało** (k = 2: 7–22), więc QUBO jest małe (§A.5).
3. Przy k = ∞ niewykonalnych jest 5–29% kandydatów.
4. W optimach N5 poprawiające dalekie zamiany są rzadkie (0,2–1,3 na stan). Zysk z zasięgu objawia się raczej w trakcie przeszukiwania (H6) niż w ucieczce z optimum.

### 14.2b Anatomia w dobrych rozwiązaniach (dodane po rundzie krytyki 1)

Stany: 2 s ILS (k = 4, estymata) ze startu round-robin lub losowego, potem domknięcie N5. 3 instancje × 2 stany na klasę.
RPD: do BKS tam, gdzie jest w prototypie (ta01–05), w pozostałych do LB. Dla 20×20 LB jest słabe, więc RPD jest zawyżone.

| Klasa | RPD [%] | L (ścieżka) | bloki | L̄ bloku | max blok | kand. k=2 | k=3 | k=4 | k=8 | k=∞ | niewyk. k=4 | niewyk. k=8 | niewyk. k=∞ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15×15 | 5,8 | 21 | 7,7 | 3,1 | 8,0 | 6,0 | 9,7 | 12,2 | 17,5 | 18,3 | 0 | 2,7 | 3,5 (19%) |
| 20×20 | (46 do LB) | 30 | 13,0 | 2,4 | 8,5 | 7,8 | 13,8 | 18,7 | 25,5 | 28,2 | 0,7 | 3,2 | 5,7 (20%) |
| 30×15 | 15,7 | 35 | 4,7 | 9,1 | 17,0 | 5,3 | 10,0 | 14,0 | 26,2 | 41,8 | 0,3 | 4,7 | 19,0 (45%) |
| 50×15 | 11,3 | 56 | 3,7 | 20,1 | 31,7 | 4,0 | 7,8 | 11,7 | 23,7 | 63,7 | 0 | 2,0 | 36,2 (57%) |
| 100×20 | 10,3 | 113 | 3,7 | 32,8 | 60,8 | 4,5 | 8,5 | 12,3 | 26,0 | 116,7 | 0 | 2,2 | 77,7 (67%) |

Wnioski z porównania z §14.2:
1. W dobrych rozwiązaniach przy n/m ≥ 2 ścieżka ma **mniej, ale dłuższych bloków** (np. 50×15: 3,7 bloku o średniej długości 20).
2. Kandydatów przy k ≤ 4 jest bardzo mało (4–19), więc QUBO jest jeszcze mniejsze.
3. **Niewykonalność dalekich zamian rośnie z jakością i z n/m:** do 67% przy k = ∞. Przy k ≤ 4 praktycznie jej nie ma.
4. Liczba „poprawiających dalekich zamian” z §14.2–14.3 (do 9 na stan) to **artefakt słabych stanów**.
   W dobrych rozwiązaniach (lokalnych optimach silniejszego sąsiedztwa) z definicji jest 0. Dlatego H5 mierzymy wydajnościowo (§14.3b).

### 14.3 Kontinuum flow shop → job shop (generator Taillarda z ρ; optima N5; 4 instancje × 2 starty)

> **Uwaga (runda krytyki 1):** tabela pochodzi z optimów N5, czyli stanów słabej jakości. Kolumny „popr.” to artefakt (w dobrych rozwiązaniach takich zamian nie ma).
> Wnioski o kontinuum opierają się na §14.3b (wydajność) i na niewykonalności w stanach po ILS (§14.3b, akapit „Struktura”). Tabelę zostawiam dla przejrzystości historii planu.

| Rozmiar | ρ | L (ścieżka) | L̄ bloku | kand. k=∞ | niewyk. k=∞ | popr. k=3 | popr. k=8 | popr. k=∞ | log₁₀\|Dyn\| |
|---|---|---|---|---|---|---|---|---|---|
| 20×5 | 0 | 32 | 6,4 | 35 | **0** | 0,12 | 1,1 | 1,8 | 8,2 |
| 20×5 | 0,25 | 31 | 9,3 | 34 | 4,4 | 0,62 | 1,8 | 2,1 | 8,2 |
| 20×5 | 1 | 21 | 12,2 | 13 | 3,4 | 0,12 | 0,5 | 0,5 | 5,3 |
| 30×15 | 0 | 79 | 5,3 | 100 | **0** | 1,1 | 3,4 | 4,5 | 19,2 |
| 30×15 | 0,1 | 101 | 9,9 | 137 | 9,8 | 1,0 | 3,4 | 5,3 | 27,1 |
| 30×15 | 0,25 | 87 | 11,0 | 123 | 14,0 | 0,88 | 4,3 | 5,6 | 23,7 |
| 30×15 | 0,5 | 64 | 7,4 | 82 | 10,1 | 0,50 | 1,6 | 1,6 | 16,3 |
| 30×15 | 1 | 44 | 5,8 | 50 | 4,5 | 0,38 | 0,62 | 0,62 | 10,0 |
| 50×10 | 0 | 108 | 10,8 | 156 | **0** | 0,75 | 4,4 | 9,0 | 29,4 |
| 50×10 | 0,1 | 141 | 21,4 | 200 | 13,9 | 0,75 | 3,6 | 9,9 | 40,4 |
| 50×10 | 0,25 | 134 | 22,4 | 192 | 21,8 | 0,38 | 2,1 | 6,8 | 38,5 |
| 50×10 | 0,5 | 98 | 15,6 | 137 | 18,1 | 0,12 | 1,5 | 4,5 | 27,5 |
| 50×10 | 1 | 60 | 20,5 | 66 | 19,9 | 0,38 | 1,25 | 1,4 | 16,4 |

Wnioski:
1. **ρ = 0 daje zero niewykonalnych** (T8).
2. ~~Poprawiające dalekie zamiany są najczęstsze przy ρ ≤ 0,25 i spadają 3–9× do ρ = 1.~~ Artefakt słabych stanów; zastąpione przez §14.3b.
3. Ścieżki i sąsiedztwa są największe przy małym, ale niezerowym ρ.

Uwaga: RPD do LB przy małym ρ jest duże (LB obciążenia maszyn jest słaby dla flow shopu), więc w E4 porównujemy warianty względem siebie, nie do LB.

### 14.3b Kontinuum wydajnościowo (dodane po rundzie krytyki 1)

Ustawienia: ILS 1 s, estymata, 3 instancje × 2 ziarna na (rozmiar, ρ). Wartości to RPD [%] do najlepszego wyniku dowolnego wariantu na tej instancji,
bo LB jest słabe przy małym ρ.

| Rozmiar | ρ | N5 | k = 3 | Dyn | c6 |
|---|---|---|---|---|---|
| 20×5 | 0 | **2,23** | 2,42 | 2,90 | 2,30 |
| 20×5 | 0,1 | 6,97 | 4,51 | **2,18** | 2,39 |
| 20×5 | 0,25 | 6,06 | 3,99 | 3,76 | **2,27** |
| 20×5 | 0,5 | 6,51 | 3,73 | 1,63 | **1,28** |
| 20×5 | 1 | 0 | 0 | 0 | 0 |
| 20×10 | 0 | **0,60** | 1,56 | 1,43 | 0,99 |
| 20×10 | 0,1 | 4,37 | 2,03 | **1,54** | 1,98 |
| 20×10 | 0,25 | 5,76 | **0,44** | 2,97 | 1,87 |
| 20×10 | 0,5 | 8,72 | 4,35 | **2,00** | 3,96 |
| 20×10 | 1 | 5,66 | 2,54 | 1,66 | **1,60** |
| 30×15 | 0 | **0,68** | 2,46 | 2,27 | 2,47 |
| 30×15 | 0,1 | 18,74 | **3,39** | 7,52 | 4,13 |
| 30×15 | 0,25 | 16,62 | 4,02 | 4,26 | **2,20** |
| 30×15 | 0,5 | 14,18 | 3,85 | 4,28 | **2,61** |
| 30×15 | 1 | 5,78 | 2,81 | 1,84 | **1,59** |
| 50×10 | 0 | **0,64** | 1,41 | 1,65 | 1,11 |
| 50×10 | 0,1 | 22,14 | 7,54 | **4,37** | 5,63 |
| 50×10 | 0,25 | 17,75 | 4,58 | **2,69** | 4,39 |
| 50×10 | 0,5 | 17,35 | 5,72 | **1,75** | 2,63 |
| 50×10 | 1 | 1,59 | 0,48 | **0** | **0** |

Struktura w tych samych instancjach (stany po 2 s ILS): przy ρ = 0 niewykonalnych jest 0, przy ρ = 0,1–0,5 do 45 z 99 kandydatów k = ∞ (50×10).
Przy ρ = 1 i n/m ≥ 4 ścieżka krytyczna to 1–1,25 bloku, czyli C_max = LB.

Wnioski:
1. **ρ = 0 (flow shop): N5 wygrywa we wszystkich czterech rozmiarach.** Pozostałe warianty są blisko.
2. **ρ = 0,1–0,5: N5 jest 3–8× gorsze od najlepszego wariantu złożonego.**
3. **ρ = 1:** przewaga złożonych jest mniejsza, a przy n/m ≥ 4 znika, bo instancje są łatwe.

Kandydat na główny wynik artykułu (H5). Wymaga E4 z większą próbą i dłuższym budżetem.

### 14.4 Zejście best-improvement (15 instancji: ta01–05, ft10, la16–20, abz5–6, orb01–02; 4 starty; dokładne delty)

| | N1 | N5 | Fib | k = 3 | k = 4 | k = 8 | Dyn |
|---|---|---|---|---|---|---|---|
| RPD do BKS [%] (średnio) | 30,8 | 30,8 | 32,8 | 28,5 | **28,4** | 28,6 | 28,6 |
| start round-robin / losowe | 19,3 / 34,6 | 19,3 / 34,6 | 20,6 / 36,9 | 18,6 / 31,9 | **17,5** / 32,0 | 19,1 / **31,8** | 19,1 / 31,8 |
| kroki | 16,9 | 16,9 | 6,6 | 8,3 | 8,4 | 8,4 | 8,4 |
| lepsze / gorsze od N5 (na 60) | — | — | 26 / 29 | 35 / 20 | **39 / 17** | 37 / 19 | 37 / 19 |

N1 = N5, zgodnie z L3: te same optima, ale N5 jest szybsze o ~37%.

### 14.5 ILS, 1 s (8 instancji: ta01–03, ft10, la16, la19, abz5, orb01; 3 ziarna; dokładne delty)

| | N1 | N5 | Fib | k = 3 | k = 4 | Dyn |
|---|---|---|---|---|---|---|
| RPD do BKS [%] | 11,31 | 8,99 | 8,29 | **7,54** | 7,61 | 7,66 |
| iteracje / s | 737 | 1 249 | 1 312 | 882 | 828 | 642 |

**Ta sama próba ze estymatą blokową (D10) zamiast dokładnych delt** oraz trzy większe instancje (ta21 20×20, ta31 30×15, ta51 50×15;
RPD do BKS 1642 / 1764 / 2760):

| | N5 | Fib | k = 3 | k = 4 | Dyn |
|---|---|---|---|---|---|
| 8 małych, estymata: RPD [%] | 7,34 | 6,16 | **5,13** | 5,19 | 5,39 |
| 8 małych, estymata: iteracje / s | 3 820 | 3 761 | 3 613 | 3 279 | 3 028 |
| 3 większe, dokładne delty: RPD [%] | 21,28 | 19,78 | 18,21 | **17,52** | 20,07 |
| 3 większe, dokładne delty: iteracje / s | 240 | 258 | 179 | 131 | 68 |
| 3 większe, estymata: RPD [%] | 18,84 | 16,15 | 13,31 | 13,97 | **12,32** |
| 3 większe, estymata: iteracje / s | 1 009 | 1 056 | 983 | 831 | 735 |

Wnioski z pilota ILS:
1. Ranking jest stabilny: N1 < N5 < Fib < k ∈ {3, 4} ≈ Dyn (od najgorszego).
2. **Estymata daje 3–10× więcej iteracji i lepsze wyniki** niż dokładne delty. To potwierdza D10.
3. Przy dokładnych deltach Dyn traci na dużych instancjach przez koszt (68 iteracji/s). Przy estymacie wygrywa na ta51 (50×15, długie bloki), zgodnie z H2.

To tylko 1 s i 3 ziarna, wyłącznie do kierunku. Wnioski rozstrzyga E3.

### 14.5b ILS z rodziną ruchów granicznych (dodane po rundzie krytyki 1)

Ustawienia: 1 s, estymata, 11 instancji (8 małych + ta21, ta31, ta51), 3 ziarna, RPD do BKS.
Ranga to średnia ranga na instancjach (1 = najlepszy z 9).

| | N5 | k = 3 | Dyn | N6 | N7 | c6 | c7 | bm3 | bm∞ |
|---|---|---|---|---|---|---|---|---|---|
| RPD [%], 8 małych | 7,54 | **5,26** | 5,52 | 5,68 | 6,53 | 5,36 | 5,48 | 5,42 | 5,70 |
| RPD [%], 3 większe | 18,84 | 13,12 | **12,69** | 15,69 | 15,21 | 13,31 | 12,89 | 12,76 | 13,02 |
| RPD [%], wszystkie 11 | 10,62 | **7,40** | 7,47 | 8,41 | 8,90 | 7,53 | 7,50 | 7,42 | 7,70 |
| średnia ranga | 8,77 | **3,36** | 4,45 | 5,73 | 7,05 | 3,68 | 4,32 | 3,68 | 3,95 |
| iteracje / s (11 instancji) | 2 868 | 2 659 | 2 095 | 2 304 | 1 970 | 2 109 | 1 974 | 2 403 | 1 592 |

Wnioski:
1. **Złożenie pomaga także wstawieniom:** c6 jest lepsze od N6 (7,53 vs 8,41), a c7 od N7 (7,50 vs 8,90).
   Zastrzeżenie (A.7, usterka 4): na poziomie instancji c7 wygrywa z N7 10:1, ale c6 z N6 tylko 7:3 przy jednym remisie, co nie jest istotne.
2. N6 jest lepsze od N5 (8,41 vs 10,62), zgodnie z literaturą.
3. Ale **złożone zamiany o umiarkowanym zasięgu (k = 3) są co najmniej tak dobre jak pojedyncze wstawienia N6/N7**.
4. Czołówka (k = 3, bm3, c6, c7, Dyn) jest w pilocie nierozróżnialna (1 s, 3 ziarna). Rozstrzyga E3.
5. Dodanie typów wstawień do zamian (bm3 vs k = 3) w pilocie nic nie daje. Hipoteza robocza H9: o jakości decyduje bardziej **złożenie** niż **typ** ruchu.

### 14.6 Addytywność (9 instancji, 12 stanów każda: od losowych do bliskich optimum N5; ruchy z ≥ 2 zamianami)

| | Fib | k = 3 | k = 4 | k = 8 | Dyn |
|---|---|---|---|---|---|
| ruchy / średnio zamian | 100 / 9,8 | 103 / 9,8 | 103 / 9,9 | 103 / 9,8 | 103 / 9,8 |
| realny zysk / suma delt (dokładne delty) | 42,6% | 37,2% | 36,4% | 36,4% | 35,9% |
| realny zysk / suma estymat | 18,7% | 14,8% | 14,5% | 14,2% | 14,2% |
| trafienie dokładne | 0–1% | 1% | 1% | 1% | 1% |
| ruch nie poprawia | 2,0% | 1,9% | 2,9% | 2,9% | 3,9% |

Wniosek: addytywność w JSSP zawodzi **mocniej** niż w PFSP (w pilocie k-bonacciego trafienie dokładne było 10–42%).
Mimo to ruch złożony prawie zawsze poprawia. Estymata jest optymistyczna zgodnie z T5 (dolne ograniczenie C_max).

---

## 15. Estymata zasobów (podsumowanie)

| Zasób | MIN | OPT | MAX |
|---|---|---|---|
| Praca w chmurze (Claude) | P0 ~6,5 dnia | P0 + P1 ~8,5 dnia | P0–P2 ~13–14 dni |
| Mac M4 Pro (sekwencyjnie) | ~9 h | ~34 h | ~65 h |
| D-Wave (QPU) | — | ~3 s (poziom MIN z §A.5) | 72–216 s (poziom OPT z §A.5) |
| IQM (kredyty) | — | ~5–8 (darmowa pula) | ~14–115 (do ~70 USD) |
| Pamięć / dysk | pomijalne (wyniki JSON jak w PFSP) | | |
