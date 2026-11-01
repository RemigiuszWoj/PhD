# Plan artykułu i badań: sąsiedztwa z krzyżowaniami (Motzkin → pojedyncze krzyżowania → inwolucje) ze studium parametru c

Wersja 1, 01.11.2026, gałąź `claude/tender-thompson-ql1r4e`. Układ jak w planie k-bonacciego (`articles/kbonacci/PLAN.md` na gałęzi `kbonacci-icaps`).
Powiązane: `articles/katalog_sasiedztw.md` (pozycje #7–#9), `ROADMAP.md` (fronty C i E).
Skrypty i surowe wyniki pilotów: `articles/crossings/pilot/` (tabele z tego planu: `python analyze_pilots.py all`).

---

## A. PROPOZYCJA USTALEŃ WIĄŻĄCYCH I PLAN WYKONAWCZY (do zatwierdzenia)

> Zasada nadrzędna jak w k-bonaccim: **badamy sąsiedztwa, nie metaheurystyki.** Protokół i wszystkie stałe badawcze są jak w poprzednich
> artykułach i w kampanii k-bonacciego, żeby nowe wyniki dało się zestawić ze starymi. Nowa jest klasa sąsiedztw i jeden czynnik:
> **bezpiecznik** (ruch złożony albo najlepsza pojedyncza zamiana, zależnie od tego, który daje mniejsze rzeczywiste C_max; §14).

### A.0 Co pokazał pilot (01.11.2026) i co z tego wynika dla planu

| Ustalenie | Liczby z pilota | Konsekwencja |
|---|---|---|
| Oś c **prawie nic nie kosztuje obliczeniowo** | kandydaci i delty są te same dla każdego c (wszystkie pary, O(mn³)). Dokładny wybór na stanach z trajektorii: c₀ ≤ 0,6 ms, c₁ i P₁ ≤ 4,3 ms, inwolucje ≤ 1,3 ms, przy deltach 14–221 ms (n = 50–100). Drożej jest tylko na gęstych stanach startowych (100×10: c₁ 0,5 s), stąd limit top-K (E0) | inaczej niż przy k nie ma kompromisu koszt–zysk; pytanie dotyczy tylko zysku |
| Oś c **prawie nie zwiększa liczności** | \|N_c\| ≈ M_n·(n/3)^c/c! (sprawdzone do n = 2000), wzrost zostaje 3ⁿ; dopiero inwolucje rosną nadwykładniczo | „pojedyncze krzyżowania” to wielomianowy dodatek do Motzkina, a nie nowa skala |
| Najlepsza inwolucja (wg sumy delt) **ma dużo krzyżowań** | mediana krzyżowań na stanach z trajektorii: 0–3 (n = 20), 6–23 (n = 50), 76 (100×10); na starcie 7–16 / ~100 / ~440 | c ≤ 2 nie odtwarza optimum inwolucyjnego, ale ma już 83–98% jego przewidywanego zysku |
| **Addytywność** psuje się, gdy zamiany na siebie nachodzą | pary zamian poprawiających (trajektoria): zysk realny/przewidywany 0,52–0,66 dla rozłącznych, 0,22–0,34 dla zagnieżdżonych, 0,12–0,37 dla krzyżujących się; para razem nie poprawia w 7–12% / 25–32% / 27–34% | o addytywności decyduje geometria pary, a nie samo c |
| Przy wyborze addytywnym **więcej krzyżowań = gorzej** | ILS 1 s: inwolucje gorsze od Motzkina o 3,1 / 7,6 / 6,9 pp (tai20_5 / tai20_10 / 50×10); 5 s: o 1,3 / 4,6 / 9,1 pp; 72/72 par, p < 0,001 | warianty *plain* zostają jako punkt odniesienia (tak liczyliśmy dotąd) |
| **Bezpiecznik** (1 dodatkowa ocena C_max) | 50×10: Motzkin 17,3% → 5,9% (1 s) i 13,7% → 5,2% (5 s), 24/24; lepiej niż Dynasearch (7,7%); przy n = 20 dla Motzkina bez zmian, dla inwolucji duża poprawa | nowy czynnik w kampanii. Tłumaczy słabe wyniki Motzkina przy n ≥ 50 w `windowed_qubo` |
| **Z bezpiecznikiem poziom c prawie przestaje mieć znaczenie** | c₁ i inwolucje wobec c₀: różnice ≤ 0,2 pp (1 s) i ≤ 0,44 pp (5 s), p ≥ 0,08. P₁ daje sygnał w obie strony: −0,31 pp na tai20_10, +0,71 pp przy 50×10 (p ≈ 0,03 przed korektą Holma) | H5 (równoważność) jako pytanie otwarte; oba wyniki da się opublikować |
| Kwanty: **największe sąsiedztwo ma najrzadsze QUBO** | QUBO inwolucji ma tylko konflikty „wspólny koniec”: 1,4–9× mniej sprzężeń, 15–58% mniej kubitów fizycznych na Pegasusie, 30–57% mniej bramek CZ w QAOA | czysty i tani eksperyment kwantowy: QPU jako wyrocznia na zapisanych stanach |
| QUBO **z poprawkami par** (DP tego nie wyrazi) | stany z trajektorii: ruchy niepoprawiające 31–62% → 0–4%, zysk 1,3–1,4× zysku najlepszej pojedynczej zamiany; ale QUBO jest gęste, a z MILP w ILS wychodzi 4–6× mniej iteracji | wariant MAX i pomost do prac kwantowych |

### A.1 Propozycje decyzji (domyślnie jak w k-bonaccim, A.0–A.1 tam)

| # | Decyzja | Propozycja |
|---|---|---|
| D1 | Konferencja | GECCO 2027: CFP jeszcze nieopublikowane, wcześniejsze terminy 29.01.2025 i 1.02.2024. Zapas: SOCO 2027 albo czasopismo. Punkt kontrolny 10.01.2027 |
| D2 | Start | permutacja identycznościowa, jak dotąd |
| D3 | Budżety | 0,1 / 0,5 / 1 / 2 / 5 / 10 s, 5 ziaren (0–4); jeden przebieg 10 s + ekstrakcja (`scripts/extract_budgets.py` z `kbonacci-icaps`) |
| D4 | ILS i SA | bez zmian (tabu 10, mushroom 10; SA: T0 = 1000, α = 0,99, podłoga 100, reheat 1,5, stagnacja 100 ms). Najpierw tylko ILS |
| D5 | Miara | RPD do LB Taillarda, Wilcoxon + Holm, α = 0,05; do tego test trendu Page’a po łańcuchu c₀ ⊂ c₁ ⊂ P₁ ⊂ inw. i TOST dla H5 (to ta sama kampania, inna analiza) |
| D6 | Filtr blokowy | bez zmian: test ostatniej maszyny, jak w Fib/Dyn/Motzkinie/k-bonaccim; dokładny filtr ścieżki krytycznej tylko w ablacji |
| D7 | Warianty | c ∈ {0, 1, ∞} i P₁ („każdy łuk krzyżowany co najwyżej raz”), każdy w wersji *plain* i *safe*, plus Swap (najlepsza pojedyncza zamiana) jako kontrola. c = 0 liczymy na nowo (jeden kod dla całej krzywej, jak k = 2 i k = n w k-bonaccim) |
| D8 | Maszyna | Mac (M4 Pro), sekwencyjnie, **po** kampanii k-bonacciego |
| D9 | Kolejność | klasyka → anatomia → kwanty (najpierw wyrocznia na stanach, pętla z QPU tylko w MAX) |
| D10 | Gałąź | Etap 0 na bazie `kbonacci-icaps` (albo po jej zmergowaniu do main), bo korzysta z jej narzędzi (kampania, ekstrakcja, analiza, filtr ścieżki krytycznej) |

### A.2 Jeden przebieg 10 s zamiast sześciu

Bez zmian względem k-bonacciego (A.2 tam): limit czasu jest wyłącznie warunkiem stopu, budżety 0,1–5 s wyciągamy z `move_log`.
Walidację ekstrakcji robi kampania k-bonacciego (etap `validate`), więc tu jej nie powtarzamy. Nowe tryby idą przez `get_neighbor`,
a ILS zapisuje `move_log` sam.

### A.3 Plan wykonawczy: co i dlaczego, po kolei

**Etap 0: przygotowanie w chmurze (Claude, ~3 dni)**

| # | Co | Dlaczego |
|---|---|---|
| 0.1 | `src/neighborhoods/classical/crossing.py`: kandydaci jak w `kbonacci.compute_candidates` z k = n (ten sam kod delt i filtr); wybór `c0` (rzadkie DP przedziałowe), `c1` i `pa1` (DP z rdzeniami H), `inv` (skojarzenie o maksymalnej wadze), `c2` (MAX), `swap`; flaga `safe`; statystyki: zysk przewidywany i realny, liczba zamian, krzyżowania, czy zadziałał bezpiecznik | jeden kod dla całej osi; prototyp `pilot/crossing_select.py` zgadza się z brute force w 2000/2000 przypadków |
| 0.2 | Tryby `cross_c0`, `cross_c1`, `cross_pa1`, `cross_inv`, `cross_c2`, sufiks `_safe`, `swap` w `get_neighbor` | ta sama ścieżka co stare sąsiedztwa |
| 0.3 | Testy: liczności (J-ułamek vs brute force); wybór vs brute force; `cross_c0` daje tę samą wartość DP co stary `motzkin` przy tych samych kandydatach; bezpiecznik nigdy nie jest gorszy od najlepszej pojedynczej zamiany; limit top-K na gęstych stanach | żeby na Macu nic nie wybuchło; równoważność z dotychczasowym Motzkinem |
| 0.4 | `scripts/run_crossing_campaign.py` (wzór: `run_kbonacci_campaign.py`): etapy main → anatomy → ablation, wznawianie, dry-run; po każdym etapie tabela liczby ruchów i udziału bezpiecznika | uruchomienie jedną komendą |
| 0.5 | `scripts/crossing_anatomy.py` (z `pilot/pilot_anatomy.py`): stany z trajektorii ILS (nie z zejścia), wszystkie c, pary wg geometrii, struktura QUBO | mechanizm do artykułu (E1) |
| 0.6 | Analiza: uogólnienie `analyze_kbonacci.py` (RPD, Wilcoxon + Holm, Page, TOST, iteracje, udział bezpiecznika) | wyniki od razu w formie tabel |
| 0.7 | `articles/crossings/RUN_LOCAL.md` | przejęcie uruchamiania przez lokalnego Claude’a |
| 0.8 | `networkx` w requirements (blossom, czysty Python) | jedyna nowa zależność |

**Etap 1: kampania klasyczna (Mac).** main (ILS) → anatomy → ablation; potem decyzja o SA (jak w k-bonaccim A.0).
**Etap 2: analiza i pisanie (chmura).** Teoria (§4) nie zależy od wyników, więc można ją pisać od razu.
**Etap 3: kwanty** (A.5).

### A.4 Trzy opcje: minimum / optimum / maksimum

Założenia: Mac sekwencyjnie, 120 instancji × 5 ziaren, przebieg 10 s + ekstrakcja, +10% narzutu. Wszystkie warianty liczą delty dla
wszystkich par, więc tai500_20 to reżim jednego ruchu (jedna ocena ~36 s, jak Dynasearch i k = n; +0,36 h na wariant).
Razem **~2,3 h na wariant i algorytm**.

| | **MINIMUM** | **OPTIMUM (rekomendacja)** | **MAKSIMUM** |
|---|---|---|---|
| Warianty | c₀, inw. (plain + safe) + Swap = 5 | c₀, c₁, P₁, inw. (plain + safe) + Swap = 9 | OPT + c₂ (plain + safe) + model drugiego rzędu (c₀ i inw., safe, top-K = 12) = 13 |
| ILS | ~11,5 h | ~21 h | ~30 h |
| Dodatki | — | anatomia E1 (~2 h), ablacja dokładnego filtra dla c₀-safe i inw.-safe (~4,6 h) | + ablacja limitu top-K (~2 h) |
| **Razem klasyka (ILS)** | **~12 h** | **~28 h** (~1,2 doby) | **~39 h** (~1,6 doby) |
| SA (po decyzji) | +11,5 h | +21 h | +30 h |
| Kwanty (A.5) | MIN | OPT | MAX |
| Co daje artykułowi | odpowiedź, czy krzyżowania pomagają i czy bezpiecznik ratuje sytuację | pełna krzywa c na obu łańcuchach, mechanizm (addytywność wg geometrii), porównanie z Dynasearch i k-bonaccim | + model drugiego rzędu (pomost kwantowy), pełne ablacje |

### A.5 Kwanty: wyrocznia na zapisanych stanach (eksperyment główny) i pętla (opcjonalnie)

**Pomysł.** QPU w pętli daje 1–4 ruchy na przebieg (2,1 s opóźnienia na wywołanie, reżim z `windowed_qubo`), więc mówi mało o samym
sąsiedztwie. Zamiast tego zapisujemy stany z klasycznych trajektorii ILS i dla każdego stanu wysyłamy na QPU trzy QUBO na **tych
samych** kandydatach:
- QUBO Motzkina: kara za wspólny koniec i za krzyżowanie;
- QUBO inwolucji: kara tylko za wspólny koniec;
- QUBO inwolucji z poprawkami par (Q_ab, §H6).

Porównujemy z dokładnym optimum (MILP): lukę energii, odsetek rozwiązań dopuszczalnych, rzeczywisty zysk C_max, liczbę kubitów
fizycznych i długość łańcuchów, zrywanie łańcuchów. Tak mierzymy wpływ gęstości QUBO na jakość sprzętu, bez szumu metaheurystyki.
W MAX dochodzi miękka kara λ za krzyżowanie (λ ∈ {0, P/4, P}): λ = 0 daje QUBO inwolucji, λ = P daje QUBO Motzkina.

| | MIN | OPT | MAX |
|---|---|---|---|
| D-Wave: stany | tai20_5: 30 stanów × 3 QUBO | tai20_5/10/20: 90 × 3 | + 50×10 (K ≤ 60): +90 × 3; + λ; + pętla tai20_5 (inw.-safe, Motzkin-safe; ILS + SA, protokół `windowed_qubo`) |
| D-Wave: wywołania / QPU | 90 / ~3 s | 270 / ~8 s | ~1 000 / ~30 s |
| IQM (K ≤ 16 na sirius, ≤ 20 na garnet; p = 1; 4096 strzałów) | 5 stanów × 2 QUBO = 10 obwodów, ~7–10 kr. (darmowa pula) | 10 × 3 = 30 obwodów, ~20–29 kr. | 30 × 3 = 90 obwodów, ~60–86 kr. |
| Bez kosztu | transpilacja QAOA na fałszywych backendach (IBM, IQM): CZ i głębokość dla trzech QUBO | jak MIN | jak MIN |

D-Wave łącznie ≤ ~40 s QPU (~10% quoty 400 s/mies.), IQM łącznie ≤ ~125 kredytów (~75 USD). Stawki: ~30 ms QPU na wywołanie przy
num_reads = 100 (`windowed_qubo`); IQM ~0,25 kr. na zadanie + 0,43–0,95 kr. na obwód (`gate_qaoa`, plan k-bonacciego).
Przy K ≤ 16 QUBO mieści się na IQM **bez okien**: na stanach z trajektorii przy n = 20 mediana \|C\| to 4–17.

### A.6 Czego świadomie nie robimy

| Element | Co to jest | Dlaczego odpada |
|---|---|---|
| NEH, budżety t·n·m/2, IG, adaptacyjny wybór c | jak w k-bonaccim A.6 | zmieniają protokół albo algorytm przeszukiwania |
| k-nieprzecinające | brak k wzajemnie krzyżujących się łuków (2-nieprzecinające = Motzkin), znane z RNA | dokładny wybór nie ma prostego DP; tylko akapit w teorii (wzrost ~5ⁿ dla k = 3) |
| Bez zagnieżdżeń („odwrócony Motzkin”) | krzyżowania wolno, zagnieżdżenia nie; ta sama liczność M_n (symetria krzyżowań i zagnieżdżeń) | ciekawostka na akapit; DP nie jest przedziałowe (kolejka zamiast stosu) |
| Klasyczny wybór z karą λ·cr | min Σδ + λ·cr | dokładny wybór wymaga MILP; tylko w części kwantowej |

---

## 0. Skrót

- **Pytanie główne** (tytuł roboczy EN: *Should swaps cross?*): czy zamiany w ruchu złożonym dla PFSP powinny móc się krzyżować?
  Motzkin dopuszcza zagnieżdżenia, ale nie krzyżowania; inwolucje dopuszczają wszystko. Gdzie na tej osi leży optimum i co je wyznacza?
- **Teza (z pilota, do potwierdzenia):**
  - Oś c prawie nic nie kosztuje obliczeniowo (te same delty, dokładny wybór za ułamek ich ceny) i prawie nie zwiększa liczności (\|N_c\| ≈ M_n(n/3)^c/c!).
  - Ograniczeniem nie jest ani koszt, ani rozmiar, tylko **nieaddytywność**. Rośnie ona, gdy zamiany na siebie nachodzą: pary rozłączne
    są najbardziej addytywne, zagnieżdżone i krzyżujące się wyraźnie mniej (przy n ≥ 50 krzyżujące się najmniej).
  - Przy wyborze addytywnym krzyżowania szkodzą. Jedna dodatkowa ocena C_max (bezpiecznik) usuwa większość strat, a wtedy poziom c przestaje mieć znaczenie.
  - Strona kwantowa: QUBO inwolucji jest najrzadsze, a tylko QUBO potrafi uwzględnić poprawki par (DP tego nie umie).
- **Cel:** GECCO 2027 (termin do potwierdzenia, spodziewany koniec stycznia lub początek lutego 2027); zapas: SOCO 2027 albo czasopismo.
  CPAIOR 2027 (abstrakt 1.12, artykuł 7.12.2026 wg beri.net) odpada, bo koliduje z ICAISC 15.11 i ICAPS 14.12.
- **Co już mamy (01.11):**
  - liczności (J-ułamek + brute force n ≤ 11), prawo Poissona sprawdzone do n = 2000, stała wzrostu P₁;
  - prototypy dokładnego wyboru c₀ / c₁ / P₁ / inw. / c₂ (MILP), 2000/2000 zgodności z brute force;
  - piloty: anatomia (192 stany), zejście (72 przebiegi), poprawki par (151 stanów), ILS 1 s i 5 s (3 klasy × 4 instancje × 3 ziarna),
    struktura QUBO (16 stanów).

---

## 1. Kontekst i luka badawcza

**Co wiadomo:**
1. Dynasearch (Congram, Potts, van de Velde 2002) wybiera niezależne, rozłączne zamiany dokładnym DP. Motzkin (prace własne; Bożejko i in.)
   dokłada zagnieżdżenia, DP przedziałowe O(n³).
2. VLSN (Ahuja, Ergun, Orlin, Punnen 2002) obejmuje sąsiedztwa przeszukiwane skojarzeniem lub przydziałem. Sarvanov i Doroshko (1981)
   oraz Gutin (1999) przeszukują nadwykładnicze sąsiedztwa TSP przydziałem w czasie wielomianowym; Deineko i Woeginger (2000) to przegląd
   wykładniczych sąsiedztw TSP i QAP.
3. **Analogia z RNA jest dokładna.** DP Nussinova (1978) to DP Motzkina, a pseudowęzły to krzyżowania: Rivas i Eddy (1999) O(n⁶),
   Akutsu (2000) proste pseudowęzły O(n⁴), Lyngsø i Pedersen (2000) NP-trudność w modelach energetycznych, Tabaska i in. (1998) dowolne
   pseudowęzły przez skojarzenie o maksymalnej wadze. Czyli: c = 0 ↔ Nussinov, c = 1 ↔ pseudowęzeł typu H, c = ∞ ↔ Tabaska.
4. Kombinatoryka krzyżowań: Touchard (1952), Riordan (1975), Flajolet (1980, ułamki łańcuchowe), Ismail, Stanton i Viennot (1987, q-Hermite),
   Kasraoui i Zeng (2006, symetria krzyżowań i zagnieżdżeń), Chen i in. (2007, k-nieprzecinające), Pilaud i Rué (2014, diagramy z k krzyżowaniami).

**Luka:**
- Osi między Motzkinem a inwolucjami nie badano w szeregowaniu; nikt nie sprawdzał, czy krzyżowania pomagają.
- Nikt nie zmierzył nieaddytywności w PFSP **w zależności od geometrii pary** (rozłączne, zagnieżdżone, krzyżujące się).
- Słabe wyniki Motzkina przy n ≥ 50 we wcześniejszych pracach nie mają wyjaśnienia. W tabeli skalowania `windowed_qubo`
  (ILS, 5 s, m = 10) Motzkin ma 13,49% wobec 5,39% Dynasearch przy n = 50 i 14,21% wobec 4,67% przy n = 100. Pilot wskazuje przyczynę (§14).
- W pracach kwantowych nikt nie zestawił gęstości QUBO z rozmiarem sąsiedztwa: największe sąsiedztwo (inwolucje) ma najrzadsze QUBO.

---

## 2. Definicje (propozycja)

**Ruch elementarny:** zamiana końców (i, j), i < j; zadania wewnątrz przedziału zostają na miejscu.
**Ruch złożony:** zbiór zamian o parami różnych końcach, czyli inwolucja σ; sąsiad to π∘σ.
**Geometria pary** (a, b), (c, d), a < c: *rozłączne*, gdy b < c; *zagnieżdżone*, gdy d < b; *krzyżujące się*, gdy c < b < d.
**cr(σ)** to liczba par krzyżujących się. **N_c** = {σ : cr(σ) ≤ c}. **P_r** = {σ : każdy łuk krzyżuje się z co najwyżej r innymi}.

„Pojedyncze krzyżowania” mają dwa naturalne znaczenia i oba są w planie:
- **N₁**: w całym ruchu co najwyżej jedno krzyżowanie;
- **P₁**: każdy łuk krzyżuje się co najwyżej raz, ale rozłącznych par krzyżujących się może być wiele.

Dwa łańcuchy zawierań (ściśle rosnące dla dostatecznie dużego n, np. N₁ ⊊ P₁ od n = 8):
- Motzkin = N₀ ⊊ N₁ ⊊ N₂ ⊊ … ⊊ N_C = inwolucje, gdzie C = C(⌊n/2⌋, 2) to największa możliwa liczba krzyżowań;
- Motzkin = P₀ ⊊ N₁ ⊊ P₁ ⊊ P₂ ⊊ … = inwolucje.

N₂ i P₁ są nieporównywalne: N₂ pozwala jednemu łukowi przeciąć dwa inne, a P₁ pozwala na wiele rozłącznych par krzyżujących się.

| Sąsiedztwo | Warunek | n = 10 | n = 20 | Wzrost |
|---|---|---|---|---|
| Dynasearch (dla porównania) | rozłączne przedziały | 512 | 5,2·10⁵ | 2ⁿ |
| **Motzkin** = N₀ | bez krzyżowań | 2 188 | 5,1·10⁷ | ~3ⁿ·n^(−3/2) |
| **N₁** | ≤ 1 krzyżowanie | 5 038 | 2,6·10⁸ | ~M_n·n/3 |
| N₂ | ≤ 2 krzyżowania | 7 108 | 7,6·10⁸ | ~M_n·(n/3)²/2 |
| N_c | ≤ c krzyżowań | — | — | **~M_n·(n/3)^c/c!** |
| **P₁** | każdy łuk krzyżowany ≤ 1 raz | 5 263 | 4,9·10⁸ | **~3,4626ⁿ** |
| 3-nieprzecinające | bez trzech wzajemnie krzyżujących się łuków | 7 990 | — | ~5ⁿ (literatura) |
| **Inwolucje** | dowolne rozłączne zamiany | 9 496 | 2,4·10¹⁰ | nadwykładniczy, ~(n/e)^(n/2)·e^(√n) |

Liczności zweryfikowane brute force dla n ≤ 11 (`pilot/counts.py`); zgodne z katalogiem dla N₁ i N₂ przy n = 5, 7, 9, 10.
Sąsiedztwo bez zagnieżdżeń ma dokładnie M_n ruchów.

| n | Motzkin | N₁ | N₂ | N₃ | P₁ | 3-nieprzec. | inwolucje |
|---|---|---|---|---|---|---|---|
| 5 | 21 | 26 | 26 | 26 | 26 | 26 | 26 |
| 7 | 127 | 204 | 225 | 232 | 204 | 225 | 232 |
| 8 | 323 | 589 | 701 | 749 | 593 | 715 | 764 |
| 9 | 835 | 1 717 | 2 221 | 2 485 | 1 753 | 2 347 | 2 620 |
| 10 | 2 188 | 5 038 | 7 108 | 8 413 | 5 263 | 7 990 | 9 496 |
| 11 | 5 798 | 14 840 | 22 826 | 28 733 | 15 995 | 27 908 | 35 696 |

Część wszystkich inwolucji, którą pokrywa N_c, czyli \|N_c\| / I(n):

| n | c = 0 | c = 1 | c = 2 | c = 4 |
|---|---|---|---|---|
| 10 | 0,23 | 0,53 | 0,75 | 0,95 |
| 20 | 2·10⁻³ | 0,011 | 0,032 | 0,12 |
| 50 | 10⁻¹³ | 1,5·10⁻¹² | 1,2·10⁻¹¹ | 2,8·10⁻¹⁰ |

Losowa inwolucja ma ~n/2 − √n/2 łuków, a każda para łuków krzyżuje się z prawdopodobieństwem 1/3. Krzyżowań jest więc ~n²/24
(pilot: 9,5 przy n = 20, 76 przy n = 50, 334 przy n = 100). N_c przy stałym c to „prawie nieprzecinający się” róg zbioru inwolucji.
Na osi k jest odwrotnie: k ≈ log₂n + 2 pokrywa ~80% Dynasearch.

**Fakt strukturalny, ważny dla interpretacji.** Każda z rodzin N_c i P_r zawiera wszystkie pojedyncze zamiany. Przy wyborze addytywnym
ma więc **te same optima lokalne co Swap**: jeśli żadna pojedyncza zamiana nie poprawia, to żaden ruch złożony nie poprawia
w przewidywaniu. Oś c zmienia tylko długość kroku poza optimami lokalnymi, a nie zbiór optimów. To odróżnia ją od osi k, gdzie małe k
ma własne optima lokalne.

---

## 3. Wkład artykułu (C1–C6) i dowody

| # | Wkład | Czym udowodniony |
|---|---|---|
| C1 | Rodziny N_c i P_r: dwa łańcuchy od Motzkina do inwolucji, liczność (J-ułamek), prawo Poissona, wzrost P₁ | T1–T4 |
| C2 | Dokładny wybór najlepszego ruchu: DP przedziałowe (c = 0), DP z rdzeniami H (c = 1 i P₁), skojarzenie (c = ∞), rdzenie z ≤ c + 1 łuków (c ≥ 2); koszt pomijalny wobec delt | T5–T6 + E0 |
| C3 | Anatomia: ile krzyżowań ma najlepszy ruch i ile przewidywanego zysku dają krzyżowania | H1, H8 (E1) |
| C4 | Nieaddytywność wg geometrii pary i jej skutek: przy wyborze addytywnym realny zysk maleje z c | H2, H3 (E1, E2) |
| C5 | Praktyka: bezpiecznik (1 ocena C_max) i model drugiego rzędu; wyjaśnienie wyników Motzkina przy n ≥ 50 | H4–H6 (E2, E3) |
| C6 | Kwanty: gęstość QUBO a rozmiar sąsiedztwa; QUBO z poprawkami par jako model, którego DP nie wyrazi | H6, H7 (E4) |

**Element podsumowujący:** tabela-taksonomia (Adjacent, Swap, Fib, k-bonacci, Dyn, Motzkin, N₁, P₁, inwolucje): liczność, wzrost,
algorytm wyboru i jego koszt, liczba sprzężeń QUBO. Spina artykuł z planem k-bonacciego (oś k) i z katalogiem.

---

## 4. Część teoretyczna: twierdzenia do udowodnienia

| # | Treść | Status |
|---|---|---|
| T1 | **J-ułamek:** Σ_n Σ_σ q^cr(σ) xⁿ = 1/(1 − x − [1]_q x²/(1 − x − [2]_q x²/(1 − x − …))), gdzie [h]_q = 1 + q + … + q^(h−1). Dowód: bijekcja Flajoleta (ścieżki Motzkina z historiami; zamknięcie łuku, po którym otwarto jeszcze j łuków, tworzy j krzyżowań). q = 0 daje liczby Motzkina, q = 1 inwolucje | zgodne z brute force dla n ≤ 11 ✅; dowód standardowy (Flajolet 1980) |
| T2 | **Prawo Poissona:** przy stałym c liczba inwolucji z dokładnie c krzyżowaniami to M_n·(n/3)^c/c!·(1 + O(1/n)). Stąd \|N_c\| ~ M_n(n/3)^c/c! i wzrost 3ⁿ dla każdego stałego c | numerycznie: stosunek do wzoru 0,82 / 0,91 / 0,95 / 0,98 / 0,995 przy n = 50 / 100 / 200 / 500 / 2000 (c = 1; dla c = 2–5 podobnie) ✅; dowód do spisania (analiza osobliwości pochodnych J-ułamka po q w q = 0; oczekiwana osobliwość (1 − 3x)^(1/2 − c)) |
| T3 | **Łańcuchy i maksimum:** N_c ⊊ N_{c+1} dla c < C(⌊n/2⌋, 2); N₁ ⊊ P₁ dla n ≥ 8; najwięcej krzyżowań ma skojarzenie (1, k+1), …, (k, 2k) | trywialne ✅ (brute force) |
| T4 | **P₁:** funkcja tworząca S = 1 + xS + x²S² + x⁴S⁴ (punkt stały / łuk / rdzeń H z trzema przerwami); wzrost 1/ρ = 3,4626, ρ = 0,288797 | zgodne z brute force dla n ≤ 11 ✅; dowód krótki |
| T5 | **Bezstratne przycinanie:** rodziny są dziedziczne (podzbiór dopuszczalnego ruchu jest dopuszczalny), więc dla przewidywanego optimum liczą się tylko łuki z δ < 0. Filtr ścieżki krytycznej (k-bonacci T5) jest bezstratny dla każdego łuku. Pozycje, które nie są końcem żadnego kandydata, można pominąć (kompresja do E ≤ 2\|C\| punktów) | dowód krótki; prototypy liczą na skompresowanych zbiorach ✅ |
| T6 | **Algorytmy wyboru:** c = 0: DP przedziałowe O(E² + E·\|C\|); c = 1 i P₁: tablica rdzeni K[i][d] z par krzyżujących się kandydatów, O(\|C\|²), plus DP O(E³); c = ∞: skojarzenie o maksymalnej wadze, O(n³) (Edmonds, Gabow); c ≥ 2: spójne składowe grafu krzyżowań mają ≤ c + 1 łuków, więc rdzenie O(\|C\|^(c+1)) i plecak po budżecie krzyżowań | c₀, c₁, P₁, inw. ✅ (brute force, 2000 przypadków); c₂ jako MILP ✅; DP dla c₂ do implementacji |
| T7 | **Wspólne optima lokalne:** każda rodzina zawierająca wszystkie pojedyncze zamiany ma przy wyborze addytywnym optima lokalne Swap; bezpiecznik gwarantuje, że ruch nie jest gorszy od najlepszej pojedynczej zamiany | trywialne |
| T8 | **Nieaddytywność wg geometrii:** w ujęciu ścieżek kratowych (k-bonacci T8) para rozłączna oddziałuje tylko przez ścieżkę krytyczną, a para zagnieżdżona lub krzyżująca się zmienia też kontekst wewnątrz przedziału. Człon drugiego rzędu: Q_ab = Δ_ab − δ_a − δ_b. Do pokazania: minimalne kontrprzykłady i warunek wystarczający na Q_ab = 0 | do spisania; kontrprzykłady brute force (E0) |

---

## 5. Hipotezy badawcze

### 5.1 Zestawienie

| ID | Hipoteza (skrót) | Eksperyment | Test | Kryterium potwierdzenia | Dane wstępne |
|---|---|---|---|---|---|
| H0 | Implementacja poprawna | testy, E0 | brute force | 100% zgodności | ✅ prototypy |
| H1 | Przewidywany zysk rośnie z c, ale się nasyca | E1 | Page po łańcuchu c₀ ⊂ c₁ ⊂ P₁ ⊂ inw. | c₀ ≥ 60% i c ≤ 2 ≥ 85% zysku inwolucji na stanach z trajektorii | c₀ 64–90%, c₁ 67–95%, P₁ 72–96%, c₂ 83–98% ✅ |
| H2 | Addytywność: rozłączne > nachodzące na siebie (zagnieżdżone ≈ krzyżujące się) | E1 | Wilcoxon na parach, Holm | rozłączne istotnie lepsze w ≥ 10/12 klas; zagnieżdżone vs krzyżujące się: pytanie otwarte | 0,52–0,66 / 0,22–0,34 / 0,12–0,37 ✅ |
| H3 | Przy wyborze addytywnym jakość ILS maleje z c | E2 | Page, Wilcoxon + Holm | inw. istotnie gorsze od c₀ w ≥ 10/12 klas | 3/3 klasy, 72/72 par (1 s i 5 s) ✅ |
| H4 | Bezpiecznik pomaga, tym bardziej, im większe n i c | E2 | Wilcoxon + Holm | safe istotnie lepsze od plain dla c₀ we wszystkich klasach n ≥ 50 | 50×10: −11,3 pp (1 s), −8,5 pp (5 s), 24/24 ✅ |
| H5 | Z bezpiecznikiem poziom c nie ma znaczenia | E2 | TOST ±0,3 pp, Friedman | równoważność w ≥ 10/12 klas | c₁, inw.: ≤ 0,2 pp (1 s), ≤ 0,44 pp (5 s), p ≥ 0,08; P₁: w obie strony, p ≈ 0,03 bez korekty ⚠️ |
| H6 | Model drugiego rzędu poprawia zysk na krok; w ILS potrzebuje szybkiego solvera | E1, E3 | Wilcoxon | < 5% ruchów niepoprawiających; w ILS zależnie od solvera | 31–62% → 0–4% ✅; ILS z MILP: brak zysku |
| H7 | QUBO inwolucji jest rzadsze i daje lepsze próbki z QPU niż QUBO Motzkina przy tych samych kandydatach | E4 | Wilcoxon na sparowanych stanach | mniej kubitów i CZ w ≥ 90% stanów; mniejsza luka energii | 15–58% mniej kubitów, 30–57% mniej CZ ✅ (16 stanów) |
| H8 | Optimum inwolucyjne ma tym więcej krzyżowań, im więcej kandydatów poprawiających | E1 | Spearman | ρ(cr, \|C\|) > 0,5 | mediana 0–3 (n = 20) … 76 (100×10) ✅ |

### 5.2 Karty hipotez

**H1. Przewidywany zysk rośnie z c, ale się nasyca**
- *Treść:* przewidywany (addytywny) zysk najlepszego ruchu rośnie wzdłuż łańcucha c₀ ⊂ c₁ ⊂ P₁ ⊂ inw. (z definicji),
  ale już c₀ ma większość zysku inwolucji.
- *Dane:* na stanach z trajektorii c₀ ma 64–90%, c₁ 67–95%, P₁ 72–96%, c₂ 83–98% zysku inwolucji; na stanach startowych c₀ ma 58–84%.
- *Pomiar (E1):* stosunek zysku dla każdego c do zysku inwolucji; test Page’a po łańcuchu.
- *Teza w artykule (EN):* “Crossings add predicted gain, but most of it is already in the noncrossing part.”

**H2. Addytywność zależy od geometrii pary**
- *Treść:* dla dwóch zamian poprawiających realny łączny zysk w stosunku do sumy delt jest wyraźnie większy dla par rozłącznych niż dla
  par, które na siebie nachodzą (zagnieżdżonych i krzyżujących się). Odwrotnie z odsetkiem par, które razem nie poprawiają.
- *Dane (stany z trajektorii):*

  | n | rozłączne | zagnieżdżone | krzyżujące się |
  |---|---|---|---|
  | 20 | 0,66 (7%) | 0,30 (32%) | 0,37 (27%) |
  | 50 | 0,61 (11%) | 0,34 (25%) | 0,23 (31%) |
  | 100 | 0,52 (12%) | 0,22 (27%) | 0,12 (34%) |

  (zysk realny/przewidywany; w nawiasie odsetek par, które razem nie poprawiają). Na stanach startowych różnice są mniejsze:
  0,91–0,96 / 0,85–0,93 / 0,82–0,90.
- *Uwaga:* kolejność zagnieżdżonych i krzyżujących się zależy od n: przy n = 20 gorzej wypadają zagnieżdżone, przy n ≥ 50 krzyżujące się.
  Dlatego hipoteza mówi o nakładaniu się, a różnica między tymi dwoma typami zostaje pytaniem otwartym (trend z n).
- *Pomiar (E1):* 300 par na stan, warstwowo wg geometrii; regresja błędu względem długości części wspólnej przedziałów.
- *Teza w artykule (EN):* “Overlap, not crossing as such, breaks additivity.”

**H3. Przy wyborze addytywnym krzyżowania szkodzą**
- *Treść:* w ILS (protokół jak dotąd) RPD rośnie wzdłuż łańcucha c₀ → c₁ → P₁ → inw.
- *Dane (ILS, RPD do LB [%], tai20_5 / tai20_10 / 50×10):*

  | | c₀ | c₁ | P₁ | inw. |
  |---|---|---|---|---|
  | 1 s | 3,38 / 9,36 / 17,29 | 3,75 / 9,79 / 17,81 | 3,75 / 10,17 / 19,27 | 6,53 / 16,95 / 24,16 |
  | 5 s | 3,10 / 8,78 / 13,69 | — | — | 4,41 / 13,33 / 22,82 |

- *Dlaczego:* H2, a do tego ILS (tabu) przyjmuje ruch pogarszający. Przy n = 50 ruchy złożone wybrane z sumy delt na stanach z trajektorii
  średnio **pogarszają** C_max dla każdego c (§14).

**H4. Bezpiecznik**
- *Treść:* z dwóch ruchów, złożonego i najlepszej pojedynczej zamiany, wybieramy ten o mniejszym rzeczywistym C_max. Koszt: jedna ocena C_max, O(mn).
- *Dane:* 50×10: c₀ 17,29 → 5,93 (1 s) i 13,69 → 5,16 (5 s); inw. 24,16 → 6,09 i 22,82 → 5,60; wszystko 12/12, p < 0,001.
  Przy n = 20 dla c₀ różnice są nieistotne, ale dla inwolucji duże (4,41 → 2,95 i 13,33 → 8,79 przy 5 s).
  Bezpiecznik działa w 27–60% iteracji, częściej przy większym n i przy inwolucjach. W zejściu do optimum lokalnego (72 przebiegi)
  ruch złożony nie poprawiał w 22–65% kroków dla Motzkina i w 40–74% dla inwolucji.
- *Konsekwencja:* wyjaśnia słabe wyniki Motzkina przy n ≥ 50 we wcześniejszych pracach (§14).

**H5. Z bezpiecznikiem poziom krzyżowań nie ma znaczenia**
- *Treść:* c₀-safe, c₁-safe, P₁-safe i inw.-safe są równoważne z dokładnością do ±0,3 pp.
- *Dane:* c₁-safe i inw.-safe wobec c₀-safe: 1 s: różnice ≤ 0,2 pp, p ≥ 0,19; 5 s: ≤ 0,44 pp, p ≥ 0,08 (przy 50×10 inw.-safe nieco
  gorsze: +0,44 pp, p = 0,08). P₁-safe (tylko 1 s): −0,01 / −0,31 / +0,71 pp (tai20_5 / tai20_10 / 50×10), dwa ostatnie z p ≈ 0,03 przed
  korektą Holma. Sygnał idzie w obie strony, więc pełna kampania musi to rozstrzygnąć.
- *Jeśli odrzucona,* czyli istnieje c* > 0 istotnie lepsze: przekaz pozytywny „krzyżowania pomagają, gdy pilnuje się addytywności”.
  Oba wyniki da się opublikować, bo mechanizm (H2) jest zmierzony.
- *Test:* TOST (dwa jednostronne testy Wilcoxona z marginesem ±0,3 pp), Friedman w każdej klasie.
- *Kontrola Swap:* c₀-safe jest lepsze od samej najlepszej zamiany przy 50×10 (−0,70 pp przy 1 s, −0,67 pp przy 5 s, p ≤ 0,03),
  a przy n = 20 tylko na tai20_5 przy 1 s (−0,43 pp, p = 0,02). Ruchy złożone mają więc sens głównie przy większym n.

**H6. Model drugiego rzędu**
- *Treść:* min Σ δ_a x_a + Σ Q_ab x_a x_b, gdzie Q_ab to rzeczywista zmiana dla pary minus δ_a minus δ_b, na K najlepszych kandydatach.
  Wybiera ruchy, które prawie zawsze poprawiają.
- *Dane (151 stanów z trajektorii, n = 20 i 50):* ruchy niepoprawiające: addytywnie 31–62%, z poprawkami 0–4%. Realny zysk to 1,3–1,4×
  zysku najlepszej pojedynczej zamiany, podczas gdy ruch addytywny nie jest od niej lepszy w 83–91% stanów.
  Koszt: K(K−1)/2 ocen C_max plus solver. W ILS 1 s z MILP wychodzi 4–6× mniej iteracji i nie ma zysku.
- *Rola:* klasycznie wariant MAX (szybki solver dla K ≤ 16: wyliczenie skojarzeń albo tabu na QUBO). Kwantowo to naturalny kandydat
  na QPU (A.5), bo DP nie umie uwzględnić Q_ab.

**H7. QUBO inwolucji na sprzęcie**
- *Treść:* przy tych samych kandydatach QUBO inwolucji ma mniej sprzężeń (tylko wspólne końce), mniejsze zanurzenie i płytszy obwód
  QAOA niż QUBO Motzkina, a próbki z QPU są bliżej optimum.
- *Dane (pilot):*

  | Stan | K | Sprzężenia M / I / PQ | Kubity na Pegasusie P16 (najdł. łańcuch) M / I / PQ | CZ w QAOA p = 1 na siatce 4×5, M / I / PQ |
  |---|---|---|---|---|
  | 20×10, trajektoria | 18 | 59 / 41 / 140 | 26 (3) / 22 (2) / 45 (3) | 209 / 146 / 463 |
  | 20×20, trajektoria | 11 | 27 / 13 / 55 | 14 (2) / 10 (1) / 20 (2) | 81 / 35 / 178 |
  | 20×20, start | 58 | 829 / 402 / 1 306 | 362 (10) / 214 (6) / 473 (12) | — |
  | 20×5, start | 65 | 1 258 / 539 / — | 502 (11) / 307 (8) / — | — |
  | 20×10, start | 108 | 2 917 / 1 179 / — | 1 332 (19) / 825 (14) / — | — |
  | 50×10, trajektoria | 105 | 2 117 / 634 / — | 1 001 (16) / 425 (8) / — | — |

  (M = Motzkin, I = inwolucje, PQ = inwolucje z poprawkami par). QUBO z poprawkami jest najgęstsze: to cena za lepszy model.
- *Pomiar:* E4 (wyrocznia na zapisanych stanach).

**H8. Ile krzyżowań ma optimum inwolucyjne**
- *Treść:* liczba krzyżowań w najlepszym (addytywnie) ruchu inwolucyjnym rośnie z liczbą kandydatów poprawiających; P₁ wystarcza tylko przy małym \|C\|.
- *Dane:* na trajektorii mediana 0 / 3 / 1 (20×5 / 20×10 / 20×20), 6 / 22 / 23 (50×5 / 50×10 / 50×20), 76 (100×10); na starcie
  7–16 / 93–103 / 438. Łuk krzyżowany więcej niż raz występuje w 27–100% stanów.

---

## 6. Eksperymenty

### E0. Walidacja i model kosztu (Etap 0 + Mac, ~1 h)
- Testy z Etapu 0; równoważność `cross_c0` ze starym `motzkin` (ta sama wartość DP przy tych samych kandydatach).
- Czas wyboru wobec czasu delt (12 klas × 30 stanów). Pilot w chmurze:
  - stany z trajektorii: c₀ ≤ 0,6 ms, c₁ i P₁ ≤ 4,3 ms, inw. ≤ 1,3 ms;
  - stany startowe 100×10: c₀ 11 ms, c₁ 0,5 s, inw. 60 ms;
  - delty dla wszystkich par: 14–29 ms (n = 50), 109–221 ms (n = 100), 823 ms (200×10).
- **Limit top-K dla gęstych stanów.** Na starcie przy n ≥ 100 kandydatów poprawiających jest ~2 000, a przy n = 500 zapewne
  dziesiątki tysięcy. Rdzenie c₁/P₁ i skojarzenie liczymy wtedy tylko na K najlepszych kandydatach (propozycja: K = 400 dla rdzeni,
  3 000 dla skojarzenia) i logujemy, jak często limit działa.
- Minimalne kontrprzykłady addytywności wg geometrii (n ≤ 6, m ≤ 3) do T8.

### E1. Anatomia na stanach z trajektorii (Mac, ~2 h, bez nowych przebiegów ILS/SA)
- Stany: 12 klas × 10 instancji × 20 stanów, losowane momenty przebiegów ILS c₀-safe i c₀-plain (10 s).
- Na każdym stanie:
  - zysk przewidywany i realny dla c₀, c₁, P₁, c₂ (gdy \|C\| ≤ 120) i inwolucji;
  - liczba zamian, krzyżowania, największa liczba krzyżowań na łuk;
  - 300 par wg geometrii;
  - sprzężenia QUBO trzech modeli; dla \|C\| ≤ 40 model drugiego rzędu (MILP).
- **Do artykułu:** rysunek „zysk przewidywany i realny w funkcji c”, rysunek addytywności wg geometrii, histogram krzyżowań optimum inwolucyjnego.

### E2. Kampania główna ILS (Mac, ~21 h w OPT)
- 9 wariantów (OPT): c₀, c₁, P₁, inw. × {plain, safe} + Swap; 12 plików × 10 instancji × 5 ziaren; 10 s z ekstrakcją 0,1–5 s.
- Odniesienia z wcześniejszych kampanii: stary Motzkin, stary Dynasearch, k-bonacci (k ∈ {2, 3, 4, 8, n}).
- Analiza:
  - tabele RPD (wariant × budżet × klasa);
  - Wilcoxon + Holm: plain vs safe, każde c vs c₀, safe vs Swap, safe vs k-bonacci k = n;
  - Page po łańcuchu (H3), TOST (H5);
  - liczba iteracji i udział bezpiecznika.
- SA tą samą komendą z `--algos sa`, po decyzji.

### E3. Ablacje (Mac)
- OPT: dokładny filtr ścieżki krytycznej dla c₀-safe i inw.-safe.
- MAX: c₂ (plain i safe); model drugiego rzędu (c₀ i inw., top-K = 12, szybki solver); limit top-K (K ∈ {200, 400, ∞} przy n ≤ 100).

### E4. Kwanty (po klasyce)
- Wyrocznia na zapisanych stanach (A.5): D-Wave i IQM, trzy QUBO na stan. Metryki: luka energii do optimum MILP, dopuszczalność,
  realny zysk C_max, kubity fizyczne, długość łańcuchów, zrywanie łańcuchów; dla QAOA liczba CZ i głębokość po transpilacji.
- MAX: pętla na D-Wave dla tai20_5 (inw.-safe, Motzkin-safe), protokół `windowed_qubo`.

### Mapa: hipoteza → eksperyment → materiał w artykule

| H | E | Materiał | Sekcja |
|---|---|---|---|
| H1, H8 | E1 | rysunek zysku w funkcji c, histogram krzyżowań | 5 |
| H2 | E1 | rysunek addytywności wg geometrii | 5 |
| H3–H5 | E2 | tabela główna, rysunek RPD w funkcji c (plain i safe) | 6 |
| H6 | E1, E3 | tabela modelu drugiego rzędu | 6 |
| H7 | E4 | tabela struktury QUBO i jakości próbek | 7 |

---

## 7. Protokół i trafność wyników

**7.1 Uczciwość porównań**
- Wszystkie c idą jedną ścieżką kodu (`crossing.py`). c₀ to Motzkin w nowym kodzie; stary `motzkin` jest tylko wierszem odniesienia.
- Wariant plain ma semantykę dotychczasowego Motzkina: ruch przyjmowany bez sprawdzenia, a gdy DP nic nie wybierze, najlepsza pojedyncza zamiana.
- Wariant safe różni się wyłącznie jedną oceną C_max i wyborem lepszego z dwóch ruchów.
- Swap (najlepsza pojedyncza zamiana) jest naturalną kontrolą: z T7 wynika, że pokazuje dokładnie to, co ruchy złożone dają ponad pojedynczą zamianę.

**7.2–7.5:** pomiar czasu, dane, statystyka i reprodukowalność jak w k-bonaccim (§7.2–7.5 tam): jedna maszyna, losowa kolejność
przebiegów, zapis liczby iteracji, metadane w każdym JSON, anonimowe repozytorium dla recenzentów. Statystyka: Wilcoxon + Holm (bloki =
instancje, średnia z ziaren), Page po łańcuchu, TOST dla H5, Friedman w klasie, A12 Varghi i Delaneya.

**7.6 Zagrożenia trafności**
- *Wewnętrzne:* stany gęste (start) i limit top-K; raportujemy, jak często limit działa.
- *n = 500:* jedna ocena trwa ~36 s, więc wynik to jeden ruch (jak w Dynasearch i Motzkinie). Z tej klasy nie wyciągamy wniosków o c.
- *Python:* koszt wyboru jest mały wobec delt, więc numba przyspieszyłaby wszystkie warianty podobnie. Wnioski o c nie zależą
  od stałych implementacyjnych, inaczej niż przy k.
- *Zewnętrzne:* tylko Taillard i C_max, jak dotąd.

---

## 8. Struktura artykułu (GECCO: zwykle 8 stron + referencje, do potwierdzenia w CFP)

| § | Sekcja | Strony | Zawartość | Rysunki/tabele |
|---|---|---|---|---|
| 1 | Introduction | 0,8 | ruchy złożone, oś krzyżowań, pytanie *Should swaps cross?*, wkład C1–C6 | — |
| 2 | Background | 0,7 | PFSP, Dynasearch, Motzkin, VLSN przez skojarzenia, analogia z RNA | — |
| 3 | The crossing family | 1,2 | definicje, dwa łańcuchy, T1–T4 (J-ułamek, prawo Poissona, P₁), wspólne optima lokalne (T7) | rys. 1 (łuki i łańcuchy), tab. 1 (taksonomia) |
| 4 | Exact move selection | 0,9 | T5–T6, algorytm 1 (DP z rdzeniami H), skojarzenie, koszt wobec delt | alg. 1 |
| 5 | Additivity by pair geometry | 0,9 | T8, E1: addytywność wg geometrii, zysk przewidywany i realny | rys. 2–3 |
| 6 | Experiments | 2,4 | E2–E3: tabela główna, plain vs safe, H5, model drugiego rzędu | tab. 2–3, rys. 4 |
| 7 | Quantum structure | 0,6 | E4 (krótko) | tab. 4 |
| 8 | Conclusions | 0,3 | rekomendacja zależna od wyników (pilot: Motzkin-safe albo inwolucje-safe; krzyżowania tylko z modelem par) | — |

---

## 9. Prace implementacyjne

| P | Zadanie | Dni |
|---|---|---|
| P0 | `crossing.py` (z prototypu) + tryby w `get_neighbor` + testy | 1 |
| P0 | `run_crossing_campaign.py`, blok w config.yaml, RUN_LOCAL.md, dry-run | 0,5 |
| P0 | `crossing_anatomy.py` (stany z trajektorii ILS) | 0,5 |
| P0 | analiza (uogólnienie `analyze_kbonacci.py`: Page, TOST, udział bezpiecznika) | 0,5 |
| P1 | DP dla c₂ (rdzenie 3-łukowe, plecak po budżecie krzyżowań) + testy | 1 |
| P1 | model drugiego rzędu: szybki dokładny solver dla K ≤ 16 + tabu na QUBO dla K ≤ 40 | 1 |
| P1 | `quantum_qubo/involution.py`, `gate_qaoa/involution.py`, kalibracja kątów, skrypt wyroczni | 1 |
| P2 | spisanie dowodów T1–T2, kontrprzykłady do T8 | 1–2 |
| P2 | numba dla delt (wspólne z k-bonaccim; go/no-go jak D5 tam) | opc. |

---

## 10. Harmonogram (propozycja) i punkty kontrolne

| Okres | Zadania | Wynik |
|---|---|---|
| do 15.11 | bez prac nad krzyżowaniami (ICAISC: Hamming; Mac zajęty kampanią k-bonacciego) | — |
| 16–22.11 | Etap 0 w chmurze (P0), dry-run | kod gotowy do uruchomienia na Macu |
| 23.11–6.12 | Mac: E2 (ILS, ~21 h), E1, ablacja filtra; teoria §3–4 | dane do H1–H5 |
| 7–20.12 | analiza, decyzja o SA; pierwszeństwo mają ICAPS (14.12) i ANNSIM (18.12) | — |
| 21.12–10.01 | pisanie §1–6; kwanty MIN/OPT (po kredytach) | szkic bez §7 |
| **10.01.2027** | **punkt kontrolny** | GECCO, jeśli E1–E2 są gotowe i jest szkic §3–6 |
| do terminu GECCO (~koniec stycznia / początek lutego) | §7, poprawki promotora, anonimowy artefakt | zgłoszenie |

**Jeśli nie zdążymy:** SOCO 2027 (termin do potwierdzenia; SOCO 2026 miało 16.02) albo rozszerzona wersja do czasopisma
(np. Journal of Heuristics, Computers & OR).

---

## 11. Ryzyka i plan B

| Ryzyko | P-stwo | Skutek | Środek zaradczy |
|---|---|---|---|
| Wynik „negatywny”: krzyżowania nie pomagają (H5) | wysokie | słabszy przekaz | przekaz o mechanizmie (H2), algorytmach (C2) i bezpieczniku (H4); negatywny wynik z wyjaśnieniem jest wartościowy |
| Recenzent: „to tylko skojarzenie o maksymalnej wadze, znane z VLSN” | średnie | odrzucenie | oś c, prawo Poissona, DP z rdzeniami H, addytywność wg geometrii, struktura QUBO; uczciwy przegląd (Ahuja i in., Tabaska i in.) |
| Bezpiecznik zmienia ocenę wcześniejszych wyników Motzkina | pewne | korekta opisu | decyzja z promotorem (§14, D4 w §12) |
| Gęste stany (start, n ≥ 100): koszt rdzeni i skojarzenia | średnie | wolny start przebiegu | limit top-K + log |
| Mac zajęty kampanią k-bonacciego | wysokie | przesunięcie | E2 po k-bonaccim; MIN (~12 h) jako wariant awaryjny |
| Kolizja z ICAISC (15.11), ICAPS (14.12), ANNSIM (18.12) | wysokie | brak czasu | teoria w grudniu, eksperymenty na Macu w tle |
| Niepotwierdzone terminy GECCO i SOCO | średnie | przesunięcie | sprawdzić CFP w grudniu |
| Nakładanie się z k-bonaccim (H5b tam: dokładna ocena najlepszej pojedynczej zamiany) | średnie | ten sam wynik w dwóch pracach | ustalić, gdzie opisujemy bezpiecznik (D3 w §12) |
| Prawo Poissona (T2) jest już znane | średnie | mniejsza nowość C1 | sprawdzić literaturę i OEIS; wtedy cytujemy i zostaje zastosowanie |

---

## 12. Decyzje do podjęcia (z promotorem)

| # | Decyzja | Rekomendacja |
|---|---|---|
| D1 | Główna oś: c (krzyżowania w całym ruchu, jak w katalogu) czy r (krzyżowania na łuk, P_r) | obie: N₁ i P₁ to dwa sposoby rozumienia „pojedynczych krzyżowań”; w kampanii c ∈ {0, 1, ∞} + P₁ |
| D2 | Konferencja | GECCO 2027; zapas SOCO 2027 albo czasopismo |
| D3 | Bezpiecznik: w tym artykule, w k-bonaccim, czy osobno | w tym artykule (tu jest mechanizm); w k-bonaccim jako ablacja |
| D4 | Wcześniejsze wyniki Motzkina przy n ≥ 50 | jedno uczciwe zdanie w tym artykule; co z pracami w recenzji, decyduje promotor |
| D5 | Zakres kwantów | wyrocznia na stanach (MIN/OPT); pętla tylko w MAX |
| D6 | SA | po ILS, jak w k-bonaccim |
| D7 | Filtr | test ostatniej maszyny (porównywalność), dokładny filtr w ablacji |
| D8 | Model drugiego rzędu | krótko tutaj; pełne studium w pracy kwantowej (fronty A/E w ROADMAP) |
| D9 | Opcja | OPTIMUM |

---

## 13. Literatura do przeczytania lub sprawdzenia

- **Kombinatoryka:** Touchard 1952 (Canad. J. Math.); Riordan 1975 (Math. Comp.); Flajolet 1980 (Discrete Math., ułamki łańcuchowe);
  Ismail, Stanton, Viennot 1987 (Europ. J. Combin., q-Hermite); Kasraoui, Zeng 2006 (krzyżowania i zagnieżdżenia w skojarzeniach);
  Chen, Deng, Du, Stanley, Yan 2007 (Trans. AMS); Pilaud, Rué 2014 (diagramy z k krzyżowaniami); Flajolet, Sedgewick 2009
  (*Analytic Combinatorics*); Jin, Qin, Reidys 2008 (k-nieprzecinające struktury RNA).
  **Pilne: sprawdzić, czy prawo Poissona (T2) jest znane** (OEIS: trójkąt „inwolucje wg liczby krzyżowań”; sieć sesji nie otwiera OEIS).
- **RNA:** Nussinov, Pieczenik, Griggs, Kleitman 1978; Rivas, Eddy 1999; Akutsu 2000; Lyngsø, Pedersen 2000; Tabaska, Cary, Gabow, Stormo 1998.
- **VLSN i skojarzenia:** Ahuja, Ergun, Orlin, Punnen 2002; Deineko, Woeginger 2000; Gutin 1999; Sarvanov, Doroshko 1981; Edmonds 1965; Gabow 1990.
- **Szeregowanie:** Congram, Potts, van de Velde 2002; Ergun, Orlin 2006; Grosso, Della Croce, Tadei 2004; Bożejko, Wodecki 2007 (multimoves);
  prace promotora (Sci. Rep. 2026, IFAC 2025): czy jest w nich coś o krzyżowaniach.
- **QUBO:** Lucas 2014 (Ising formulations); Glover, Kochenberger, Du 2019 (tutorial QUBO); prace własne (`windowed_qubo`, `gate_qaoa`).
- **Statystyka:** Demšar 2006; Page 1963; Schuirmann 1987 (TOST); Vargha, Delaney 2000.

Szczegóły bibliograficzne (tomy, strony) zweryfikować w Crossref przed pisaniem, jak w ACIIDS.

---

## 14. Ważne ustalenie poboczne: bezpiecznik i słabe wyniki Motzkina przy n ≥ 50

W `get_neighbor` ruch złożony Motzkina jest przyjmowany bez sprawdzenia, a ILS (tabu) przyjmuje też ruchy pogarszające.
Pilot pokazuje, że na stanach z trajektorii ruch złożony wybrany z sumy delt **średnio pogarsza** C_max:

| Klasa | c₀ (Motzkin) | c₁ | P₁ | inwolucje |
|---|---|---|---|---|
| 20×10 | −0,09 | −0,22 | −0,24 | −0,18 |
| 50×10 | −0,37 | −0,45 | −0,50 | −0,84 |
| 50×20 | −0,42 | −0,50 | −0,48 | −0,69 |
| 100×10 | −0,68 | −0,84 | −0,80 | −1,04 |

(zysk realny/przewidywany; wartość ujemna oznacza, że ruch średnio pogarsza C_max)

Skutek w ILS (4 instancje × 3 ziarna, RPD do LB [%]):

| | tai20_5, 1 s | tai20_10, 1 s | 50×10, 1 s | tai20_5, 5 s | tai20_10, 5 s | 50×10, 5 s |
|---|---|---|---|---|---|---|
| stary Motzkin | 3,53 | 9,91 | 16,28 | | | |
| stary Dynasearch | 3,91 | 9,75 | 7,73 | | | |
| Swap | 3,65 | 9,48 | 6,63 | 3,14 | 8,97 | 5,83 |
| c₀ plain (nowy kod) | 3,38 | 9,36 | 17,29 | 3,10 | 8,78 | 13,69 |
| c₀-safe | 3,23 | 9,40 | **5,93** | 2,96 | 8,97 | **5,16** |
| inwolucje plain | 6,53 | 16,95 | 24,16 | 4,41 | 13,33 | 22,82 |
| inwolucje-safe | **3,07** | 9,56 | 6,09 | **2,95** | **8,79** | 5,60 |

**Wnioski:**
1. Ten sam mechanizm tłumaczy tabelę skalowania z `windowed_qubo` (Motzkin 13,49% wobec 5,39% Dynasearch przy n = 50; 14,21% wobec
   4,67% przy n = 100). Dynasearch ma tylko pary rozłączne, a te są najbardziej addytywne (H2).
2. Tamte wyniki nie są błędne: to przebiegi z regułą „przyjmij ruch bez sprawdzenia”. Opis „Motzkin jest najbogatszy, ale najdroższy”
   jest jednak niepełny, bo przyczyną nie jest koszt, tylko addytywność. Co z tym zrobić w pracach w recenzji, to decyzja z promotorem (D4).
3. Ta sama poprawka prawdopodobnie pomoże k-bonacciemu przy dużym k (to H5b z planu k-bonacciego przy t = 1); warto dodać ją tam jako ablację.

Odtworzenie: `cd articles/crossings/pilot && python pilot_ils.py 1000` (~2 min na 3 rdzeniach), potem `python analyze_pilots.py ils`.

---

## 15. Status szczegółowy (01.11.2026)

- ✅ `pilot/counts.py`, `pilot/counts_asymptotics.py`: liczności, J-ułamek, prawo Poissona, wzrost P₁.
- ✅ `pilot/crossing_select.py`: c₀, c₁, P₁, inwolucje (networkx), MILP dla dowolnego c (scipy/HiGHS); walidacja 2000/2000.
- ✅ `pilot/pilot_anatomy.py` (anatomia i zejście), `pilot/pilot_pairq.py` (model drugiego rzędu), `pilot/pilot_ils.py` (ILS z repo bez
  zmian; nowe tryby podpięte w skrypcie), `pilot/pilot_qubo_structure.py` (minorminer, qiskit).
- ✅ Wyniki w `pilot/measurements/`; tabele: `python analyze_pilots.py all`.
- ⚠️ Piloty liczone w chmurze (4 rdzenie, czasy zaszumione) na instancjach z generatora Taillarda. tai20_5 i tai20_10 to prawdziwe
  ta001–ta004 i ta011–ta014 (LB zgodne z tabelą Taillarda). „lcg50_10” to instancje z tego samego generatora, ale ich nasion nie
  potwierdziłem jako ta041–ta044. RPD liczone do maszynowego dolnego ograniczenia.
- ⏳ Etap 0 (§9) i decyzje D1–D9.

---

## 16. Estymata zasobów (podsumowanie)

| Zasób | MIN | OPT | MAX |
|---|---|---|---|
| Mac, klasyka (ILS) | ~12 h | ~28 h | ~39 h |
| Mac, SA (opcjonalnie) | +11,5 h | +21 h | +30 h |
| D-Wave QPU | ~3 s | ~8 s | ~30–40 s |
| IQM | ~7–10 kr. (darmowa pula) | ~20–29 kr. | ~60–86 kr. |
| Praca w chmurze | ~2,5 dnia | ~3 dni | ~5–6 dni |

Najpierw idzie kampania k-bonacciego (ILS ~22 h, ewentualnie SA kolejne ~22 h). Razem z OPT stąd to ~2 doby Maca samego ILS,
a z SA w obu kampaniach ~4 doby.
