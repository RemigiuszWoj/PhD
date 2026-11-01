# Plan artykułu i badań: rodzina sąsiedztw k-bonacci (Fibonacci → tribonacci → … → Dynasearch) ze studium parametru k

Wersja 2, 01.11.2026, gałąź `kbonacci-icaps`. Zastępuje plan v1 (historia w gicie).
Powiązane: `articles/katalog_sasiedztw.md`, `ROADMAP.md` (front C), szkice `main.tex` (EN) i `wersja_pl.tex` (PL).

---

## A. USTALENIA WIĄŻĄCE I PLAN WYKONAWCZY (01.11.2026, po rozmowie)

> Ta sekcja **nadpisuje** sprzeczne punkty planu v2 poniżej. W szczególności nie robimy: NEH, budżetów t·n·m/2, IG, BKS, adaptacyjnego k, Friedmana.
> Zasada nadrzędna: **badamy sąsiedztwa, nie metaheurystyki.** Protokół i wszystkie stałe badawcze są jak w poprzednich artykułach, żeby nowe wyniki
> dało się bezpośrednio zestawić ze starymi. Nowa jest tylko klasa sąsiedztw.

### A.0 Aktualizacja decyzji (01.11.2026, wieczór) — obowiązuje przed A.1–A.6

| Sprawa | Decyzja |
|---|---|
| Klasyka na Macu | **opcja MAKSIMUM**: k ∈ {2, 3, 4, 5, 6, 8, 12, 16, n} + Insert + ablacja dokładnego filtra (k ∈ {3, 8}) + walidacja wyciągania budżetów |
| Kwanty | **etapami**: najpierw MIN; jeśli starczy budżetu, OPT; potem MAX (skrypt `scripts/run_quantum_tiers.py` sam pilnuje budżetu i wznawia) |
| Dodane, bo nie zmieniają obliczeń | RPD do BKS (jeśli będzie plik BKS), poprawka Holma, test Friedmana + Nemenyi (CD) obok Wilcoxona |
| Odłożone na później lub na kolejną konferencję | start z NEH, budżety t·n·m/2, Iterated Greedy, adaptacyjne k |
| ILS / SA | bez zmian w logice. Do SA dodano wyłącznie log ruchów (`move_log`); sprawdzone, że przebiegi z logiem i bez są identyczne (deterministyczny zegar) |
| Filtr blokowy | bez zmian w głównej kampanii. Dokładny filtr ścieżki krytycznej tylko w ablacji (`kbonacci_k3_cp`, `kbonacci_k8_cp`) |

**Gotowe w kodzie (01.11):**

| Plik | Co robi |
|---|---|
| `scripts/run_kbonacci_campaign.py` | etapy validate → main → insert → ablation, sekwencyjnie, z wznawianiem; dry-run: ~44 h |
| `scripts/extract_budgets.py` | wyciąganie budżetów 0,1–5 s z przebiegu 10 s + porównanie z niezależnymi przebiegami i kontrolą szumu |
| `scripts/analyze_kbonacci.py` | tabele RPD i iteracji, Wilcoxon (+Holm), Friedman + CD, opcjonalnie BKS, najlepsze k |
| `scripts/run_quantum_tiers.py` | kwanty MIN → OPT → MAX z budżetem i księgą zużycia |
| `scripts/verify_taillard.py` | weryfikacja 120 instancji generatorem Taillarda |
| `src/neighborhoods/classical/insert.py` | Insert z akceleracją Taillarda (1,85 s / ocena przy 500×20) |
| `src/neighborhoods/quantum_qubo/kbonacci.py` | QUBO k-bonacciego dla D-Wave, z `L_max = k` także na QPU |
| `src/neighborhoods/gate_qaoa/kbonacci.py` | k-bonacci w oknach dla IQM + darmowa wycena okien |
| `RUN_LOCAL.md` | instrukcja dla lokalnego Claude’a |

**Walidacja wyciągania budżetów (mini, chmura: tai20_5, 2 instancje × 4 ziarna, k = 3):**
- ILS: |różnica| 0,00–0,06 pp, a dwa niezależne przebiegi różnią się o 0,00–0,04 pp (ILS jest deterministyczny).
- SA: |różnica| 1,4–1,7 pp, przy szumie dwóch niezależnych przebiegów 1,4–2,5 pp.
- Wniosek: SA reheatuje po stagnacji mierzonej zegarem, więc **nawet powtórzenie tego samego przebiegu** daje inny wynik.
  Wyciąganie z logów jest więc tak samo dobre jak osobny przebieg. Pełna walidacja na Macu w etapie `validate`.

**Poziomy kwantowe (z dry-runu skryptu):**

| | MIN | OPT (przyrost) | MAX (przyrost) |
|---|---|---|---|
| D-Wave: zakres | tai20_5, k ∈ {3, 4}, ILS + SA | + k = 8 na tai20_5; + tai20_10, tai50_10, tai100_10 (ILS, k ∈ {3, 4}) | + tai200_10, tai500_20; + reszta pełnego Taillarda (ILS + SA, k ∈ {3, 4}) |
| D-Wave: przebiegi / QPU / zegar | 200 / ≤ 18 s / ~0,4 h | 400 / ≤ 36 s / ~0,8 h | 1 900 / ≤ 171 s / ~3,7 h |
| IQM: zakres | tai20_5 inst. 0, k ∈ {3, 4} | tai20_5 inst. 0–9, k ∈ {3, 4} | + k = 8 na tai20_5; + tai20_10, tai20_20 (k ∈ {3, 4}) |
| IQM: ruchy / obwody | 2 / 12 | 18 / 108 | 50 / 300 |
| IQM: kredyty sirius / garnet | 5,7 / 8,9 | 50,9 / 80,1 | 141,5 / 222,5 |

D-Wave łącznie (MIN + OPT + MAX) to ≤ 225 s QPU, czyli ~56% quoty 400 s albo ~12,5% przy 1800 s.
IQM łącznie: ~198 kredytów na sirius (~120 USD) albo ~312 na garnet (~185 USD).
Na IQM ILS i SA przy jednym ruchu z permutacji identycznościowej wybierają ten sam ruch, dlatego MAX zamiast SA dokłada kolejne klasy instancji.

### A.1 Decyzje

| # | Decyzja |
|---|---|
| D1 | Konferencja: ICAPS albo IJCAI, zależnie od tego, na którą zdążymy |
| D2 | Start z permutacji identycznościowej, jak dotąd |
| D3 | Budżety 0,1 / 0,5 / 1 / 2 / 5 / 10 s, 5 ziaren (0–4), jak dotąd |
| D4 | ILS i SA **bez zmian** (kadencja tabu 10, mushroom 10; SA: T0 = 1000, α = 0,99, podłoga 100, reheat 1,5, stagnacja 100 ms) |
| D5 | Miara jak dotąd: RPD do LB Taillarda, test Wilcoxona, α = 0,05 |
| D6 | Filtr blokowy **bez zmian**: k-bonacci używa tego samego testu co Fib/Dyn/Motzkin, czyli porównywalność. Dokładny filtr najwyżej jako ablacja w opcji MAX |
| D7 | Warianty: k ∈ {2, 3, 4, 8, n}. Adjacent, Fibonacci, Dynasearch i Motzkin bierzemy z wcześniejszych wyników. Insert tylko w opcji MAX |
| D8 | Liczymy na Macu (M4 Pro), sekwencyjnie jak wcześniej. Przygotowanie w chmurze → `git pull` na Macu → lokalny Claude uruchamia |
| D9 | Kolejność: najpierw klasyka, potem kwanty (D-Wave, IQM) po decyzji o kredytach |

**Insert** to ruch „wyjmij zadanie z pozycji i i wstaw na pozycję j” ((n−1)² sąsiadów). Klasycznie najsilniejszy dla PFSP, ale nie było go
w poprzednich pracach. Dodanie go to nowa implementacja i nowa linia odniesienia, dlatego jest tylko w MAX.

**k = 2 i k = n liczymy na nowo**, mimo że Fibonacci i Dynasearch już są:
1. Krzywa jakość(k) musi pochodzić z jednego kodu.
2. Stary `fibonacci` przy braku poprawy nie robi ruchu, tylko wywołuje kick, a k-bonacci (jak Dynasearch i Motzkin) robi najlepszą
   pojedynczą zamianę. To daje różnicę w pilocie: 1,61 vs 1,31.
3. Stary `dynasearch_full` ma rekurencyjne DP (2–3× wolniejsze) i problem z limitem czasu przy n = 500 (audyt).
Stare wiersze zostają w tabeli jako odniesienie, a różnice opisujemy.

### A.2 Jeden przebieg 10 s zamiast sześciu (ekstrakcja budżetów z logów)

- **Dlaczego wolno:** w ILS i SA limit czasu jest **wyłącznie warunkiem stopu**. Chłodzenie (α na iterację), reheat (stała stagnacja 100 ms),
  tabu i pula elitarna nie zależą od budżetu. Przebieg z budżetem t jest więc prefiksem przebiegu 10 s z tym samym ziarnem.
- **Reguła dokładna:** pętla sprawdza zegar *przed* ruchem. Wynik dla t to najlepsze rozwiązanie po ostatnim ruchu, który *rozpoczął się* przed t,
  czyli po ruchu, którego poprzednik skończył się przed t. Ruch rozpoczęty przed t, a skończony po t, się liczy, dokładnie jak w osobnym przebiegu.
- **Co trzeba dodać:**
  - SA nie zapisuje `move_log` (ILS zapisuje). Dodajemy samo logowanie, bez zmian logiki.
  - Skrypt ekstrakcji.
  - Walidacja na małym zestawie prawdziwych przebiegów.
- **Oszczędność:** 10 s zamiast 18,6 s na konfigurację, czyli −46%.

### A.3 Plan wykonawczy: co i dlaczego, po kolei

**Etap 0: przygotowanie w chmurze (Claude, gałąź `kbonacci-icaps`, ~1 dzień)**

| # | Co | Dlaczego |
|---|---|---|
| 0.1 | Spisać stałe protokołu z poprzednich prac (config.yaml, windowed_qubo §setup) w `articles/kbonacci/PROTOKOL.md` | jedno źródło prawdy; porównywalność |
| 0.2 | Sprawdzić, że k-bonacci idzie tą samą ścieżką co stare sąsiedztwa (`get_neighbor`, ten sam filtr, ten sam runner) | żadnych ukrytych różnic poza sąsiedztwem |
| 0.3 | `move_log` w SA (samo logowanie) + test, że wyniki SA z logowaniem i bez są identyczne przy tym samym ziarnie | ekstrakcja dla SA; dowód, że logika się nie zmieniła |
| 0.4 | `scripts/extract_budgets.py`: z wyników 10 s tworzy wiersze dla 0,1 / 0,5 / 1 / 2 / 5 s (reguła z A.2), w formacie zgodnym z dotychczasowymi CSV/tabelami | mniejsze budżety bez dodatkowych przebiegów |
| 0.5 | Plan walidacji ekstrakcji: tai20_5 i tai50_10, k ∈ {3, 8}, ILS + SA, wszystkie budżety jako osobne przebiegi (~2 h na Macu) | recenzent zapyta; to tanie |
| 0.6 | Blok konfiguracji „FAZA K” w config.yaml (tryby `kbonacci_k2/k3/k4/k8/kn`, 12 plików, budżet 10 s, osobny katalog wyników) | uruchomienie jedną komendą |
| 0.7 | Testy jednostkowe + test dymny (wszystkie tryby, krótki budżet) w chmurze | żeby na Macu nic nie wybuchło |
| 0.8 | `articles/kbonacci/RUN_LOCAL.md`: instrukcja dla lokalnego Claude’a (pull, venv, kolejność n, wznawianie, kontrola postępu, co i gdzie zapisać, co wypchnąć) | przejęcie uruchamiania |
| 0.9 | Skrypt analizy: tabele RPD (k × budżet × n, ILS i SA), Wilcoxon vs stare sąsiedztwa, liczba iteracji, rysunki | wyniki od razu do artykułu |

**Etap 1: kampania klasyczna (Mac, lokalny Claude)**

| # | Co | Dlaczego |
|---|---|---|
| 1.1 | Sprawdzić, że stare wyniki (Adj/Fib/Dyn/Motz) dla wszystkich 12 plików są na Macu i kompletne (audyt: braki w zagregowanych CSV, Motzkin n = 500) | wiersze odniesienia |
| 1.2 | Walidacja ekstrakcji (0.5) → raport | zanim zainwestujemy kilkanaście godzin |
| 1.3 | Kampania: n = 20 → 50 → 100 → 200 → 500, sekwencyjnie, z wznawianiem | najpierw szybkie klasy; błąd wychodzi wcześnie |
| 1.4 | Ekstrakcja budżetów + analiza (0.9) → wypchnięcie wyników na gałąź | dane do artykułu |

**Etap 2: analiza i pisanie (chmura).** Tabele i rysunki do szkicu; rozdziały 3–5 można pisać już teraz.
**Etap 3: kwanty (po klasyce, po kredytach).** Patrz A.4.

### A.4 Trzy opcje: minimum / optimum / maksimum

Założenia: Mac sekwencyjnie, 120 instancji × 5 ziaren × 2 algorytmy, przebiegi 10 s z ekstrakcją, +10% narzutu.
k = n przy n = 500: jedna ocena trwa ~36 s, więc przebieg = 1 ruch z przekroczeniem limitu (+~0,7 h na kampanię; tak samo jak stary Dynasearch).

| | **MINIMUM** | **OPTIMUM (rekomendacja)** | **MAKSIMUM** |
|---|---|---|---|
| **Klasyka: warianty** | k ∈ {2, 3, 4, 8, n} | k ∈ {2, 3, 4, 5, 6, 8, 12, 16, n} | OPTIMUM + Insert + ablacja dokładnego filtra (k ∈ {3, 8}) |
| Klasyka: czas na Macu | **~19 h** (1 noc + dzień) | **~34 h** (~1,5 doby) | **~48 h** (2 doby) |
| Dodatki | walidacja ekstrakcji (~2 h) | + anatomia i addytywność na stanach z trajektorii (bez ILS/SA, ~2 h) | + implementacja Insert (~1 dzień pracy) |
| **D-Wave** (protokół jak `windowed_qubo`: tai20_5, 10 inst. × 5 ziaren, num_reads = 100, ILS + SA, przebieg 5 s → ekstrakcja 0,1–2 s) | k ∈ {3, 4}: 600 wywołań, **~18 s QPU**, ~0,5 h | k ∈ {3, 4, 8} na n = 20 + skalowanie k ∈ {3, 4} dla n = 50, 100: ~2 100 wywołań, **~63 s QPU**, ~1,5 h | + n = 200, 500 (k = 3) i tai20_10/20: ~3 900 wywołań, **~117 s QPU**, ~2,5 h + embedding |
| % miesięcznej quoty D-Wave (≈400 s wg repo / 1800 s, jeśli 30 min) | 5% / 1% | 16% / 3,5% | 29% / 6,5% |
| **IQM** (protokół jak `gate_qaoa`: tai20_5, ziarno 0, p = 1, 4096 shotów, okno 6, 1 ruch na przebieg; ~0,25 kr./zadanie + 0,43–0,70 kr./obwód, ~6 obwodów na ruch — do potwierdzenia darmową transpilacją) | demo: 1 instancja, k ∈ {3, 4}, ILS: **~6–9 kredytów** (darmowa pula) | 10 instancji, k ∈ {3, 4}, ILS: **~57–89 kredytów** (~35–55 USD; 2–3 mies. darmowej puli albo prośba o kredyty) | 10 instancji, k ∈ {3, 4, 8}, ILS + SA: **~170–270 kredytów** (~100–160 USD) |
| Co daje artykułowi | pełna krzywa k na całym Taillardzie, porównanie ze starymi wynikami | + kształt krzywej (gdzie się nasyca), anatomia ruchu, addytywność, pomost kwantowy | + najsilniejsza klasyczna linia odniesienia (Insert), ablacja filtra, pełniejsza część kwantowa |

**Uwagi:**
- **Quota D-Wave:** w repo (config.yaml) zapisano empirycznie ~400 s/mies. (alokacja w maju skończyła się po ~403 s). Jeśli to 30 min, czyli 1800 s,
  zapas jest ~4,5× większy. Sprawdzić w Leap. Każda opcja mieści się w jednym miesiącu.
- **D-Wave w kodzie:** `quantum_dynasearch` na backendzie `dwave` **ignoruje `L_max`** (src/neighborhoods/quantum_qubo/dynasearch.py:106).
  QUBO k-bonacciego wymaga małej nakładki `quantum_kbonacci`, czyli nowego sąsiedztwa, bez zmian w ILS/SA.
  Na IQM `gate_dynasearch` przekazuje `L_max` poprawnie.
- **Embedding dla n ≥ 50 nie jest pewny:** QUBO pasmowe ~(k−1)n·(udział po filtrze). Fibonacci n = 500 się zmieścił, ale pasmo k = 3 jest szersze. Ryzyko w opcji MAX.
- **Rekomendacja: OPTIMUM.**
  - Klasyka ~1,5 doby na Macu daje pełną krzywą k i odpowiedź, czy korzyść się nasyca.
  - D-Wave ~1 min QPU to akapit-pomost do prac kwantowych.
  - IQM: prosimy o ~100 kredytów.

### A.5 Kwanty na pełnym Taillardzie (n = 20…500): ile by to kosztowało

Liczby w A.4 dotyczą protokołów z poprzednich prac kwantowych: D-Wave głównie n = 20 (tai20_5) plus skalowanie do n = 100, IQM tylko tai20_5.
Pełny Taillard to 12 plików × 10 instancji × 5 ziaren × 2 algorytmy = 1 200 przebiegów na wartość k, każdy po 10 s, mniejsze budżety z logów.

**D-Wave.** Rozliczany jest czas QPU na wywołanie (~30 ms przy num_reads = 100), a nie czas zegarowy. Wywołanie trwa ≥ 2,1 s zegarowo, więc w 10 s mieszczą się 1–4 wywołania.

| Wariant | Wywołania | QPU | Quota (400 s / 1800 s) | Zegar |
|---|---|---|---|---|
| k ∈ {3, 4, 8}, 1 wyw./przebieg | 3 600 | 108 s | 0,3 mies. / 6% | ~10 h |
| k ∈ {3, 4, 8}, 4 wyw./przebieg | 14 400 | 432 s | 1,1 mies. / 24% | ~10 h |
| k ∈ {2, 3, 4, 8}, 1 wyw./przebieg | 4 800 | 144 s | 0,4 mies. / 8% | ~14 h |
| k ∈ {2, 3, 4, 8}, 4 wyw./przebieg | 19 200 | 576 s | 1,4 mies. / 32% | ~14 h |

→ **Na D-Wave pełny Taillard jest tani w kredytach.** Ograniczeniem nie są kredyty, tylko:
1. **Embedding:** QUBO ma ~(k−1)·n·(udział po filtrze) zmiennych. Przy n = 500 i k = 8 to tysiące zmiennych, więc potrzebne są okna, czyli kilka wywołań na ruch.
   Fibonacci do n = 500 już działał; k = 3–4 prawdopodobnie też, k = 8 przy n ≥ 200 raczej nie.
2. **Czas zegarowy** ~10–14 h.
3. **Jakość:** 1–4 iteracje w 10 s (reżim jednego ruchu jak w `windowed_qubo`, RPD ~20–25%).
Wartość naukowa to opis tego reżimu, nie wygrana z klasyką. Do dopisania w opcji MAKSIMUM jako „D-Wave pełny Taillard, k ∈ {3, 4}”.

**IQM.** Płaci się za każdy obwód. Kubitów jest 16–54, więc QUBO dzieli się na okna po 6 pozycji, czyli ~n/3 obwodów na **jeden** ruch.
Przy n = 500 to ~166 obwodów na ruch.

| Wariant (1 ruch na przebieg) | Obwody | Kredyty | USD |
|---|---|---|---|
| 1 ziarno, ILS, k ∈ {3, 4} | ~9 400 | ~4 100–6 600 | ~2 500–4 000 |
| 1 ziarno, ILS, k ∈ {3, 4, 8} | ~14 100 | ~6 200–10 000 | ~3 700–6 000 |
| 5 ziaren, ILS + SA, k ∈ {3, 4} | ~94 000 | ~41 000–66 000 | ~25 000–40 000 |

→ **Na IQM pełny Taillard jest nierealny:** 150–2 200 miesięcy darmowej puli i kolejki na tysiące godzin. Dlatego `gate_qaoa` zrobił tylko tai20_5.
Sensowne maksimum na IQM: tai20_5 (A.4) albo dodatkowo tai20_10/20 (n = 20, wciąż ~6 obwodów na ruch).

### A.6 Co odrzuciliśmy z planu v2 i co to miało dawać

| Element v2 | Co to jest | Po co go proponowałem | Dlaczego odpada | Koszt powrotu |
|---|---|---|---|---|
| **Start z NEH** | heurystyka Nawaza–Enscore–Hama: buduje dobrą permutację startową przez kolejne wstawianie zadań (standard w PFSP) | konferencje A* oczekują startu z NEH; wyniki bliżej stanu wiedzy | zmienia ILS/SA i przekreśla porównanie ze starymi wynikami (identyczność) | nowa kampania dla wszystkich sąsiedztw |
| **Budżety t·n·m/2** | limit czasu rosnący z rozmiarem instancji (Ruiz i Stützle 2007), np. t = 30 ms: 1,5 s dla 20×5, 150 s dla 500×20 | standard porównań w literaturze PFSP; duże instancje dostają więcej czasu | inne budżety = brak porównania ze starymi tabelami | nowa kampania |
| **Iterated Greedy (IG)** | najsilniejsza prosta metaheurystyka dla PFSP: niszczy d = 4 zadania, wstawia je z powrotem najlepiej, plus lokalne przeszukiwanie Insert | silna linia odniesienia; recenzent A* pyta „a jak to wypada względem IG?” | badamy sąsiedztwa, nie metaheurystyki | ~1,5 dnia implementacji + kampania |
| **BKS (best-known solutions)** | najlepsze znane makespany z literatury (wiele to optima); RPD do BKS zamiast do dolnego ograniczenia | RPD do BKS to standard i pokazuje rzeczywistą odległość od najlepszych | dotychczas raportowaliśmy RPD do LB | **zero kosztu obliczeń**: ta sama kampania, tylko inna wartość odniesienia; wystarczy plik BKS |
| **Adaptacyjne k** | k zmieniane w trakcie przeszukiwania (reguła z długości bloku, eskalacja jak w VND albo bandyta) | „wersja AI-owa” na IJCAI; odpowiada na pytanie, jak dobrać k bez strojenia | to już element algorytmu przeszukiwania | ~1 dzień + kampania; dobre na osobny artykuł |
| **Test Friedmana** | nieparametryczny test porównujący naraz wiele wariantów (wszystkie k), z post-hoc Nemenyi i diagramem CD | Wilcoxon porównuje tylko pary; przy 9 wartościach k to 36 par (problem wielokrotnych porównań) | dotychczas tylko Wilcoxon | **zero kosztu obliczeń**: to analiza tych samych danych |

**Propozycja:** BKS i Friedman nie zmieniają protokołu ani kampanii, więc warto je dodać jako uzupełnienie:
RPD do BKS w dodatkowej kolumnie lub suplemencie, Friedman obok Wilcoxona. Tabele główne zostają jak dotąd.

---

## 0. Skrót

- **Pytanie główne** (tytuł roboczy EN: *How far should a swap reach?*): jak daleko powinny sięgać zamiany w złożonym ruchu dla PFSP,
  skoro koszt rośnie z zasięgiem, a korzyść, jak zakładamy, się nasyca?
- **Teza (do potwierdzenia):**
  - Parametr k wyznacza ciągły kompromis między Fibonaccim (k = 2) a Dynasearch (k = n).
  - Pierwsze poziomy hierarchii są prawie darmowe: z dokładnym filtrem blokowym ocena kosztuje O(mn + m²k³), czyli niewiele więcej niż samo policzenie C_max.
  - Korzyść z k nasyca się w skali długości bloku ścieżki krytycznej.
  - Wniosek: przy realnym budżecie czasu optymalne k jest umiarkowane i da się je wyznaczyć prostą regułą albo adaptacyjnie.
- **Cel:** ICAPS 2027 (termin ~14.12.2026, niepotwierdzony). Zapas: IJCAI 2027 (~11–15.01.2027). Decyzja w punkcie kontrolnym 22.11.
- **Co już mamy:**
  - moduł `kbonacci.py` + 96 testów;
  - dane Taillarda zweryfikowane generatorem;
  - pilot (PRD maleje z k; przy n ≤ 50 i 1 s wygrywa k = n);
  - szkice EN i PL.
- **Nowe ustalenie (01.11):** obecny test granic bloków (`compute_block_boundaries`) **odrzuca 2–7% zamian poprawiających**.
  Filtr oparty na prawdziwej ścieżce krytycznej nie odrzuca żadnej i zostawia 2–5× mniej kandydatów. Szczegóły w §14.
  **Pilot trzeba powtórzyć po poprawce.**

---

## 1. Kontekst i luka badawcza

**Co wiadomo:**
1. Dynasearch (Congram, Potts, van de Velde 2002) pokazał, że wykładniczo duże sąsiedztwo niezależnych zamian przeszukuje się dokładnie
   metodą DP. Ergun i Orlin (2006) zeszli z przeszukaniem do O(n²), a Grosso i in. (2004) dodali wstawienia. To wszystko dotyczy jednej maszyny
   (1||ΣwT), gdzie zmiany kosztu niezależnych zamian **sumują się**.
2. Dla PFSP mamy rodzinę sąsiedztw liczonych ciągami: Adjacent, Fibonacci (rozłączne zamiany sąsiednie), Dynasearch i Motzkin
   (prace własne oraz Bożejko, Uchroński, Wodecki, Sci. Rep. 2026).
3. Własności blokowe ścieżki krytycznej (Nowicki i Smutnicki 1996, Grabowski i Wodecki 2004) wskazują ruchy, które nie mogą poprawić C_max.
4. W innych dziedzinach ograniczanie zasięgu ruchu to standardowe narzędzie: Or-opt z segmentami ≤ 3, granular tabu search
   (Toth i Vigo 2003), listy sąsiadów w TSP. VNS (Mladenović i Hansen 1997) systematycznie zmienia rozmiar sąsiedztwa.

**Luka:**
- Sąsiedztwa złożone dla PFSP porównuje się jako **odosobnione punkty** (Fib vs Dyn vs Motzkin). Nikt nie zbadał ciągłej osi między nimi.
- Nie wiadomo, **które zamiany wybiera Dynasearch**: długie czy krótkie.
- Nie wiadomo, **ile kosztuje i ile daje każdy kolejny poziom zasięgu**.
- Nikt nie zmierzył, **jak bardzo założenie addytywności zawodzi w PFSP**.

---

## 2. Definicje (ustalone 01.11.2026)

**Ruch elementarny:** zamiana końców (i, j) przy odległości d = j − i ∈ {1, …, k−1}. Zadania wewnątrz przedziału zostają na miejscu.
**Ruch złożony:** zbiór takich zamian na parami rozłącznych przedziałach [i, j]. N_k(π) to wszystkie permutacje osiągalne jednym ruchem złożonym.

| Sąsiedztwo | Odległości zamian | k | Liczność | Wzrost λ_k |
|---|---|---|---|---|
| Fibonacci | 1 | 2 | F(n+1) | 1,618 |
| tribonacci | 1, 2 | 3 | T(n); n = 10: 274 | 1,839 |
| tetranacci | 1, 2, 3 | 4 | n = 10: 401 | 1,928 |
| k-bonacci | 1, …, k−1 | k | a_n = a_{n−1} + … + a_{n−k} | → 2 |
| Dynasearch | 1, …, n−1 | n | 2^{n−1} | 2 |

Część całego Dynasearch, którą pokrywa N_k, czyli |N_k| / 2^{n−1} (policzone 01.11):

| n | k = 2 | k = 3 | k = 4 | k = 6 | k = 8 | k = ⌊log₂n⌋ | k = ⌊log₂n⌋ + 2 |
|---|---|---|---|---|---|---|---|
| 20 | 2·10⁻² | 0,23 | 0,54 | 0,89 | 0,98 | 0,54 | 0,89 |
| 100 | 9·10⁻¹⁰ | 3·10⁻⁴ | 0,03 | 0,46 | 0,83 | 0,46 | 0,83 |
| 500 | 10⁻⁴⁶ | 8·10⁻¹⁹ | 10⁻⁸ | 0,02 | 0,38 | 0,38 | 0,79 |

Wniosek: **przy k ≈ log₂n + 2 sąsiedztwo N_k zawiera ~80% wszystkich ruchów Dynasearch**. Jest to prawo Erdősa–Rényiego o najdłuższej
serii: najdłuższa część losowej kompozycji n ma długość ~log₂n. Przy stałym k jest to natomiast wykładniczo znikoma część.

---

## 3. Wkład artykułu (C1–C6) i dowody

| # | Wkład | Czym udowodniony |
|---|---|---|
| C1 | Rodzina N_k: łańcuch Adj ⊂ N_2 = Fib ⊂ N_3 ⊂ … ⊂ N_n = Dyn, liczność, stała wzrostu, pokrycie Dynasearch | T1–T4 (dowody) |
| C2 | Efektywna ocena: **bezstratny** filtr ścieżki krytycznej, koszt O(mn + m²k³), pozycyjne DP O(n + mk²) | T5–T7 + E0 |
| C3 | Anatomia: jak sięgają zamiany wybierane przez Dynasearch i gdzie korzyść z k się nasyca (skala bloku) | H1, H2 (E1, E2) |
| C4 | Kompromis przy budżecie czasu i model koszt–zysk wyznaczający k* | H3, H4 (E0, E3) |
| C5 | Praktyka: reguła/adaptacja k oraz wariant świadomy addytywności | H5, H6 (E1, E4, E5) |
| C6 | Pozycja względem metod referencyjnych (ILS-Insert, IG Ruiza i Stützlego) | H7 (E6) |

**Element podsumowujący** (z tabeli konferencji: „artykuł podsumowujący + tribonacci/k-bonacci”):
- Tabela-taksonomia wszystkich sąsiedztw (Adjacent, Insert, Fib, k-bonacci, Dyn, Motzkin): liczność, wzrost, koszt oceny, DP,
  zgodność z filtrem, struktura QUBO.
- Oś k jako brakujące ogniwo hierarchii. Motzkin (c = 0 krzyżowań) to kolejny poziom.
- Własne prace cytujemy oszczędnie i w trzeciej osobie (double-blind).

---

## 4. Część teoretyczna: twierdzenia do udowodnienia

| # | Treść | Status |
|---|---|---|
| T1 | **Liczność:** \|N_k\| = a_n, gdzie a_0 = 1 i a_p = Σ_{L=1}^{min(k,p)} a_{p−L}. Dowód: pokrycie kafelkami 1…k | szkic gotowy, testy brute force n ≤ 9 ✅ |
| T2 | **Różnowartościowość:** różne pokrycia dają różne permutacje. Kafelek [i, j] to najmniejszy przedział od i, który przesunięcie pozycji przeprowadza na siebie | do spisania (krótki) |
| T3 | **Łańcuch zawierań:** N_k ⊊ N_{k+1} dla k < n; N_k = N_n dla k ≥ n; Adj ⊂ N_2 | trywialny |
| T4 | **Wzrost:** λ_k to największy pierwiastek x^k = x^{k−1} + … + 1. Zachodzi 2 − λ_k = λ_k^{−k} (sprawdzone numerycznie), więc λ_k = 2 − Θ(2^{−k}). **Pokrycie:** \|N_k\|/\|N_n\| to P(najdłuższa część losowej kompozycji ≤ k); dla k = log₂n + O(1) jest to stała, dla stałego k Θ((λ_k/2)^n) | do spisania; dobrać źródło dla prawa najdłuższej serii |
| T5 | **Bezstratny filtr:** niech U to kolumny zmiany maszyny dowolnej jednej ścieżki krytycznej (\|U\| ≤ m−1). Jeśli [i, j] ∩ U = ∅, to δ_ij ≥ 0 (ścieżka przechodzi te same komórki w jednym wierszu, a suma się nie zmienia). Wniosek: **DP na przefiltrowanych kandydatach ma tę samą wartość optymalną** co bez filtra | dowód krótki; empirycznie 0 utraconych z ~87,5 tys. poprawiających ✅ |
| T6 | **Koszt:** kandydatów jest ≤ \|U\|·(k−1)(k+2)/2 ≤ (m−1)(k−1)(k+2)/2, każda delta kosztuje O(mk). Ocena: O(mn) Head/Tail + O(n+m) ścieżka + O(m²k³) delty + O(n + mk²) DP. Dla k = n: O(mn³), jak w Dynasearch | do spisania; zweryfikować w E0 |
| T7 | **DP pozycyjne:** B[p] = min(B[p−1], min_d B[p−d−1] + δ_{p−d,p}), poprawność i złożoność; wariant top-t w O(ntk) | poprawność ✅ (testy); top-t do implementacji |
| T8 | **Addytywność:** w ujęciu ścieżek kratowych (max-plus) C' = max_P[ℓ(P) + Σ_s Δ_s(P)]. Kontrprzykłady w obie strony, warunek wystarczający (wspólna ścieżka krytyczna przed i po), wyjaśnienie, dlaczego dominuje zawyżanie (przejmuje „druga” ścieżka) | do spisania; minimalne kontrprzykłady brute force (E0) |
| T9 (opc.) | **Delty max-plus dla długich zamian:** macierz segmentu (m×m) rozszerzana przyrostowo daje wszystkie delty w O(m²nk), a Dynasearch w O(m²n²) zamiast O(mn³). Opłaca się dla k > m | pomysł; sprawdzić literaturę i zaimplementować tylko przy zapasie czasu |

---

## 5. Hipotezy badawcze

### 5.1 Zestawienie

| ID | Hipoteza (skrót) | Eksperyment | Test | Kryterium potwierdzenia | Dane wstępne |
|---|---|---|---|---|---|
| H0 | Implementacja poprawna (liczność, DP, równoważności, filtr) | testy, E0 | brute force | 100% zgodności | ✅ (bez nowego filtra) |
| H1 | Przy równej liczbie kroków jakość rośnie monotonicznie z k | E2, E1 | test trendu Page’a | trend istotny w ≥ 10/12 klas | pilot: zysk/krok rośnie z k ✅ |
| H2 | Korzyść z k nasyca się w skali bloku: k_sat ∝ L̄ ≈ n/(\|U\|+1), a nie ∝ n | E1 | regresja log k_sat ~ log n + log m, Spearman | współczynnik przy log m < 0 (CI bez 0), ρ(k_sat, L̄) ≥ 0,5 | słabe: średnia odległość rośnie z n/m |
| H3 | Przy budżecie T = t·n·m/2 istnieje k* wewnętrzne (2 < k* < n) dla n ≥ 100, a k* rośnie z t | E3 | Friedman + Holm, Page dla k*(t) | k = 2 i k = n istotnie gorsze od k* w ≥ 4/6 klas n ≥ 100 | pilot: n ≤ 50 → k* = n |
| H4 | Model koszt–zysk przewiduje k* | E0 + E1 → E3 | R², trafność | R² ≥ 0,95; k̂ w plateau w ≥ 70% komórek | — |
| H5a | Błąd addytywności rośnie z k i m; dominuje zawyżanie poprawy | E1 | Page, Spearman | trend istotny; pred < real w > 50% ocen dla k ≥ 4 | pilot ✅ (losowe permutacje) |
| H5b | Wybór top-t z dokładną oceną poprawia ARPD dla dużych k, a dla k = 2 nie | E5 | Wilcoxon + Holm | istotna poprawa dla k ≥ k_sat | — |
| H6 | Adaptacyjne k ≈ wyrocznia i > najlepsze globalne stałe k | E4 | Wilcoxon + Holm (zbiór testowy) | nieistotnie gorsze od wyroczni; istotnie lepsze od globalnego k | — |
| H7 | ILS(k*) > Adj/Fib/Dyn/Motzkin w ILS; IG_ins + k-bonacci > IG_ins | E6 | Friedman + Nemenyi (CD), Wilcoxon + Holm | jak w hipotezie | — (ryzyko: IG) |

### 5.2 Karty hipotez

**H1. Moc sąsiedztwa rośnie z k (przy równej liczbie kroków)**
- *Treść:* zejście do optimum lokalnego z tych samych startów daje optima tym lepsze, im większe k. Średni rzeczywisty zysk jednego kroku
  (na tym samym stanie) rośnie z k.
- *Uzasadnienie:* N_k ⊂ N_{k+1}, więc przewidywany zysk jest monotoniczny z definicji. Pilot (losowe permutacje): rzeczywisty zysk
  rośnie od −42,5 (k = 2) do −96,3 (k = n) przy 20×5 oraz od −84,3 do −196,2 przy 50×10.
- *Dlaczego może nie wyjść:* błąd addytywności rośnie z k (H5a), a większe ruchy mogą prowadzić do gorszych atraktorów.
- *Pomiar:* ARPD optimum lokalnego, liczba kroków i czas zejścia (E2); sparowany zysk/krok na stanach z E1.
- *Analiza:* test trendu Page’a (bloki = instancje, kolejność k); między sąsiednimi k Wilcoxon + Holm.
- *Jeśli odrzucona:* wzmacnia to wątek addytywności (H5b) jako warunek korzystania z dużego k.
- *Teza w artykule (EN):* “Larger k yields better local optima per step.”

**H2. Nasycenie w skali bloku ścieżki krytycznej**
- *Treść:* korzyść z k nasyca się przy k_sat proporcjonalnym do średniej długości bloku L̄ = n/(|U|+1), a nie do n.
- *Uzasadnienie:*
  - poprawiająca zamiana musi obejmować granicę bloku (T5);
  - realny zysk daje przeniesienie zadania przez granicę do sąsiedniego bloku, czyli na odległość rzędu L̄;
  - pilot (losowe permutacje, słaba przesłanka): średnia odległość zamian przy k = n wynosi 4,6–5,3 przy n/m = 2 (20×10),
    6,4–7,7 przy n/m = 4 (20×5) i 7,9–8,9 przy n/m = 5 (50×10).
- *Pomiar (E1):*
  - k_sat(stan) = min k, przy którym zysk(k) ≥ 95% zysku(k_max);
  - kwantyle q50 i q90 odległości zamian wybranych przy k_max;
  - |U| i długości bloków.
- *Test rozstrzygający:* siatka Taillarda rozdziela n i m (20×{5, 10, 20}, 50×{5, 10, 20}, …). Regresja
  log k_sat = a + b·log n + c·log m. Skala bloku przewiduje b ≈ 1 i c ≈ −1, skala globalna c ≈ 0. Dodatkowo Spearman ρ(k_sat, L̄) i bootstrap CI.
- *Konsekwencja praktyczna:* reguła k = ⌈c·L̄⌉ (E4).
- *Teza w artykule (EN):* “How far a swap should reach is set by the block length of the critical path, not by n.”

**H3. Wewnętrzne optimum przy budżecie czasu**
- *Treść:* przy budżecie T = t·n·m/2 ms (protokół Ruiza i Stützlego) krzywa ARPD(k) ma minimum wewnętrzne 2 < k* < n dla n ≥ 100,
  a k* rośnie z t. Dla n ≤ 50 dopuszczamy k* = n, jak w pilocie.
- *Uzasadnienie:*
  - zysk się nasyca (H2), a koszt rośnie (T6);
  - ekstrapolacja z pilota (do potwierdzenia w E0): ocena k = n kosztuje ~0,15 s przy 100×10, ~2,3 s przy 200×20 i ~36 s przy 500×20;
  - dla n = 500 optimum wewnętrzne jest więc trywialne, a **klasy informatywne to n = 100 i 200**.
- *Pomiar:* ARPD względem BKS dla każdej kombinacji (klasa, t, k). k* = argmin. Plateau P = zbiór k nieistotnie gorszych od k*.
- *Analiza:* Friedman w każdej klasie (bloki = 10 instancji, średnia z ziaren), Holm względem k*, test Page’a dla k*(t). Mapa ciepła ARPD (klasa × k).
- *Jeśli odrzucona* (k = n najlepsze wszędzie, gdzie wykonalne): przekaz brzmi „duże k, ograniczone kosztem”, a H2 i tak daje regułę.
- *Teza w artykule (EN):* “Under realistic budgets the best k is moderate.”

**H4. Model koszt–zysk przewiduje k\***
- *Treść:*
  - (a) t_eval(k) ≈ α·mn + β·m²k³ (małe k) z R² ≥ 0,95;
  - (b) k̂ = argmax_k zysk(k)/t_eval(k), wyznaczony na stanach z E1, trafia w plateau P z E3 w ≥ 70% komórek (klasa, t).
- *Dlaczego może nie wyjść:* zysk/czas opisuje fazę zejścia, a ignoruje jakość atraktorów i dynamikę ILS (kick, tabu).
- *Wartość:* jeśli wyjdzie, k da się wybrać bez kampanii strojenia. Jeśli nie, to też uczciwy wynik o granicach modeli „lokalnych”.

**H5a. Addytywność zawodzi tym bardziej, im większe k**
- *Treść:* błąd |Σδ − ΔC_max| (bezwzględny i względny) rośnie z k i z m. Przeważnie suma delt zawyża poprawę.
- *Dane:* pilot na losowych permutacjach: trafienie dokładne w 42% ocen (k = 2), 16–20% (k = 3, 4) i 10–12% (k ≥ 6).
  Zawyżenie wynosi 10% (k = 2), 26% (k = 3, 4), 41% (k = 6) i ~50% (k ≥ 8). Do potwierdzenia na stanach z trajektorii ILS, czyli blisko optimów.
- *Teoria:* T8.

**H5b. Wybór świadomy addytywności pomaga przy dużym k**
- *Treść:* dokładna ocena t najlepszych ruchów DP (t ∈ {3, 5}) oraz najlepszej pojedynczej zamiany i wybór najlepszego rzeczywiście
  poprawia ARPD przy równym czasie dla k ≥ k_sat. Dla k = 2 nie ma efektu.
- *Koszt:* t dodatkowych ocen O(mn) na iterację, co jest małe wobec delt przy dużym k.
- *Wariant opcjonalny:* DP z wiązką (beam) i dokładną oceną prefiksu.

**H6. Adaptacyjny wybór k**
- *Treść:* wybór k online nie jest istotnie gorszy od wyroczni (najlepsze stałe k w klasie) i jest istotnie lepszy od najlepszego
  jednego k wspólnego dla wszystkich klas. Kandydaci na mechanizm wyboru:
  - reguła blokowa k_t = ⌈c·L̄_t⌉;
  - eskalacja VND po K;
  - bandyta (sliding-window UCB, nagroda = zysk/czas).
- *Ochrona przed przeuczeniem:* c i hiperparametry bandyty stroimy na instancjach 1–5, oceniamy na 6–10.
- *Rola:* w ICAPS sekcja krótka, w IJCAI centralna („wersja AI-owa”).

**H7. Pozycja względem metod referencyjnych**
- *Treść:*
  - (a) ILS z k* (albo z adaptacją) jest lepsze od ILS z Adjacent, Fib (k = 2), Dyn (k = n) i Motzkinem (gdzie wykonalny)
    i nie gorsze od ILS-Insert;
  - (b) IG_ins z k-bonaccim jako drugim etapem lokalnego przeszukiwania jest lepsze od samego IG_ins.
- *Ryzyko:* IG_ins najpewniej wygra z samym ILS. Wtedy uczciwie przedstawiamy wkład jako projekt sąsiedztwa, a (b) jako test komplementarności.

---

## 6. Eksperymenty

K, czyli zbiór badanych wartości k: {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64} ∩ [2, n], plus n, jeśli wykonalne (n ≤ 200).
Budżet: T = t·n·m/2 ms. Σ n·m/2 po 12 klasach wynosi 10 975, więc jeden przebieg (instancja, ziarno, wariant) przez wszystkie klasy
trwa 329 s przy t = 30. Pełny Taillard (10 instancji × 5 ziaren) to **4,6 h CPU na wariant** przy t = 30.

### E0. Walidacja i model kosztu (H0, H4a; T5–T6, T8) — tydzień 1, ~1 h CPU
- Poprawka filtra (§14) i testy:
  - bezstratność: brute force dla n ≤ 12 oraz losowe instancje n = 50;
  - niezmienniczość wyniku DP.
- Mikro-benchmark t_eval(k) dla 12 klas × K × 30 stanów (NEH i optima lokalne), z rozbiciem na Head/Tail, ścieżkę krytyczną,
  delty i DP. Liczba kandydatów a ograniczenie (m−1)(k−1)(k+2)/2.
- Minimalne kontrprzykłady addytywności (brute force n ≤ 6, m ≤ 3) dla T8.
- Decyzja numba (D5).
- **Do artykułu:** rysunek kosztu (log-log t_eval vs k dla kilku n) i parametry modelu.

### E1. Anatomia sąsiedztwa na stanach z trajektorii (H1, H2, H5a) — tydzień 2, ~4 h CPU
- Stany: 12 klas × 10 instancji × 20 stanów, losowane momenty ILS z k_ref = 8 przy t = 30 plus optima lokalne ze startu NEH.
- Na każdym stanie, dla każdego k ∈ K (sparowanie):
  - zysk przewidywany i rzeczywisty;
  - liczba zamian i histogram odległości;
  - |U| i długości bloków;
  - liczba kandydatów i czas.
- Analiza: krzywe nasycenia zysk(k)/zysk(k_max), rozkład k_sat, regresja z H2, błąd addytywności w funkcji k i m.
- **Do artykułu:** rysunek nasycenia, rysunek k_sat vs L̄ (log-log, kolor = m), rysunek addytywności.

### E2. Zejście do optimum lokalnego (H1) — tydzień 2, ~3 h CPU
- Starty: NEH i 10 losowych na instancję. Zejście best-improvement z każdym k aż do braku rzeczywistej poprawy.
  Przy n = 500 ograniczamy się do k ≤ 64 i 3 instancji.
- Metryki: ARPD optimum, liczba kroków, czas. Test trendu Page’a.

### E3. Studium k przy budżecie czasu (H3, H4b) — tydzień 3, ~61 h CPU (~8 h na 8 rdzeniach)
- ILS ze startu NEH, k ∈ K, 12 klas × 10 instancji × 3 ziarna:

| Budżet | Zakres | CPU |
|---|---|---|
| t = 10 | pełna siatka k | 9 h |
| t = 30 | pełna siatka k | 27 h |
| t = 90 | n ≥ 100, 5 wartości k wokół plateau, 2 ziarna | 24 h |

- Metryki: ARPD względem BKS, iteracje, krzywe anytime (najlepsze dotąd rozwiązanie w funkcji czasu).
- Analiza: Friedman + Holm w każdej klasie, plateau, k*(t), porównanie z predykcją H4.
- **Do artykułu:** mapa ciepła ARPD (klasa × k) przy t = 30 i tabela k* z plateau.

### E4. Wybór k (H6) — tydzień 4, ~12 h CPU
- Warianty:
  - reguła blokowa;
  - eskalacja VND;
  - bandyta;
  - wyrocznia (k* z E3);
  - najlepsze globalne k.
- Strojenie na instancjach 1–5, ocena na 6–10, t = 30, 5 ziaren.
- **Do artykułu:** tabela porównania.

### E5. Addytywność w praktyce (H5b) — tydzień 3–4, ~8 h CPU
- k ∈ {4, k*, 2k*} × {zwykły, top-5 z dokładną oceną + najlepsza pojedyncza zamiana}.
- Klasy n ∈ {50, 100, 200}, 10 instancji × 3 ziarna, t = 30.

### E6. Porównanie końcowe (H7) — tydzień 5, ~67 h CPU (~9 h na 8 rdzeniach)
- Warianty (wszystkie ze startu NEH, ten sam ILS):
  - sąsiedztwa: Adjacent, Insert, k = 2, k = 3, k*, najlepszy wariant adaptacyjny z E4, k = n (n ≤ 200), Motzkin (n ≤ 200);
  - IG: IG_ins, IG_ins + kb, IG_kb.
- 12 klas × 10 instancji × 5 ziaren; t = 30 (50 h) i t = 10 (17 h).
- Analiza: Friedman + Nemenyi (diagram CD) globalnie i dla każdego n; Wilcoxon + Holm dla kluczowych par; efekt A12 Varghi i Delaneya.
- **Do artykułu:** tabela główna i diagram CD.

**Łącznie ~157 h CPU, czyli ~20 h na 8 rdzeniach wydajnościowych M4 Pro.**
Budżety czasowe są stałe, więc numba nie skraca E3, E4 ani E6. Zwiększa tylko liczbę iteracji (realizm), a skraca E0–E2.

### Mapa: hipoteza → eksperyment → materiał w artykule

| H | E | Materiał | Sekcja |
|---|---|---|---|
| H1 | E2, E1 | rysunek nasycenia, liczby z E2 | 6.3 |
| H2 | E1 | rysunek k_sat vs L̄ | 6.3 |
| H3 | E3 | mapa ciepła, tabela k* | 6.4 |
| H4 | E0, E1, E3 | rysunek kosztu, kolumna „predykcja” w tabeli k* | 6.2, 6.4 |
| H5 | E1, E5 | rysunek addytywności, tekst | 5, 6.5 |
| H6 | E4 | tabela wyboru k | 6.6 |
| H7 | E6 | tabela główna, diagram CD | 6.7 |

---

## 7. Protokół i trafność wyników

**7.1 Uczciwość porównań**
- Wszystkie k idą **tą samą ścieżką kodu** (`kbonacci.py`). Fibonacci to `kbonacci_k2`, Dynasearch to `kbonacci_kn`.
- Stary `dynasearch_full` (rekurencyjne DP, 2–3× wolniejsze) **nie jest baseline’em**, bo recenzent słusznie uznałby go za słomianą kukłę.
- Stary tryb `fibonacci` różni się w ILS obsługą optimum lokalnego: zwraca „brak ruchu”, co wyzwala kick, zamiast wymuszonej zamiany.
  Dlatego w pilocie wyszło 1,61 vs 1,31.
- Ten sam start (NEH), te same tabu i kick, ten sam filtr (dokładny) dla wszystkich wariantów, jeden język i jedna implementacja.

**7.2 Pomiar czasu**
- Liczba workerów = liczba rdzeni wydajnościowych (`sysctl -n hw.perflevel0.physicalcpu`), bez przeciążania.
- Kolejność przebiegów losowana, żeby ewentualny dryf maszyny rozłożył się na warianty.
- Zawsze zapisujemy liczbę iteracji, co pozwala na analizę przy równej liczbie iteracji.
- **Jedna tabela = jedna maszyna.** Kontenery chmurowe tylko do developmentu, nie do pomiarów czasowych.

**7.3 Dane i wartości odniesienia**
- Taillard 120 instancji, zweryfikowane generatorem ✅.
- **Aktualne BKS:** UB w plikach pochodzą z 1993 roku, więc trzeba zdobyć bieżące (od promotora albo z literatury, np. przeglądu
  Fernandez-Viagasa, Ruiza i Framinana 2017).
- RPD względem LB raportujemy dodatkowo dla ciągłości z wcześniejszymi pracami.
- Opcjonalnie dla IJCAI: benchmark VRF (Vallada, Ruiz, Framinan 2015).

**7.4 Statystyka**
- α = 0,05, korekta Holma.
- Bloki = instancje, wartości uśrednione po ziarnach, bez pseudoreplikacji.
- **Uwaga:** przy 5 instancjach dokładny Wilcoxon nie zejdzie poniżej p = 0,0625, dlatego testy w klasach robimy zawsze na 10 instancjach,
  a testy na zbiorze testowym (6–10) łączymy po klasach (60 bloków).
- Test Page’a dla hipotez o monotoniczności (H1, H5a, k*(t)).
- Friedman + Nemenyi z diagramem CD (Demšar 2006), efekt A12, bootstrap CI dla k* i współczynników regresji.

**7.5 Reprodukowalność**
- Każdy wynik JSON zawiera commit, maszynę, wersję Pythona, ziarno i konfigurację.
- Skrypty E0–E6 i analizy w repo.
- Anonimowe repozytorium dla recenzentów (np. anonymous.4open.science), bo repo na GitHubie jest publiczne i podpisane.

**7.6 Zagrożenia trafności**
- *Wewnętrzne:* stałe implementacyjne Pythona przesuwają k*, bo numba przyspieszy delty bardziej niż narzut pętli.
  → raportujemy model kosztu, a wnioski formułujemy względem niego. Opcjonalnie jedna klasa z numbą dla porównania.
- *Zewnętrzne:* tylko Taillard i tylko C_max. → VRF w wersji IJCAI; w dyskusji przenośność na ΣC_j (tam też brak addytywności).
- *Konstrukcyjne:* ARPD do BKS zależy od jakości BKS. → podać źródło i datę BKS.

---

## 8. Struktura artykułu (ICAPS: zwykle 8 stron + referencje, do potwierdzenia w CFP)

| § | Sekcja | Strony | Zawartość | Rysunki/tabele | Zależy od |
|---|---|---|---|---|---|
| 1 | Introduction | 0,9 | PFSP, ruchy złożone, luka „odosobnionych punktów”, pytanie „how far should a swap reach?”, wkład C1–C6, główne wyniki | — | całość |
| 2 | Background & Related Work | 0,7 | PFSP i bloki; Dynasearch/VLSN; sąsiedztwa liczone ciągami; ograniczone/granularne sąsiedztwa; VNS/ALNS | — | lektura §13 |
| 3 | The k-bonacci Family | 1,2 | definicja, T1–T4, łańcuch, pokrycie Dynasearch (log₂n) | rys. 1 (kafelki + hierarchia), tab. 1 (taksonomia, element podsumowujący) | teoria |
| 4 | Efficient Evaluation | 1,0 | Head/Tail, bezstratny filtr T5, koszt T6, DP T7, algorytm 1 | alg. 1 | E0 |
| 5 | Additivity | 0,5 | T8 (ścieżki kratowe, kontrprzykłady), wariant top-t | — | E0, E1 |
| 6 | Experimental Study | 3,2 | 6.1 setup; 6.2 koszt; 6.3 nasycenie i skala bloku; 6.4 budżet czasu; 6.5 addytywność; 6.6 wybór k; 6.7 porównanie | rys. 2–5, tab. 2–4 | E0–E6 |
| 7 | Discussion & Limitations | 0,3 | Python, addytywność, inne kryteria, pasmowe QUBO (1 zdanie, pomost do prac kwantowych) | — | — |
| 8 | Conclusions | 0,2 | rekomendacja k, reguła | — | — |
| — | Supplement | — | dowody, pełne tabele, artefakt | — | — |

**Rysunki:**
- Must-have: rys. 1 (kafelki + hierarchia), rys. 3+4 (nasycenie i k_sat vs L̄, połączone), rys. 5 (mapa ciepła), diagram CD.
- Nice-to-have: rys. 2 (koszt), rys. addytywności. Mogą zastąpić je liczby w tekście.

**Szkice `main.tex` i `wersja_pl.tex` wymagają po Fazie 0:** zmiany opisu filtra (T5–T6), nowych liczb z pilota,
przebudowy pod strukturę z tej tabeli i oficjalnego szablonu.

---

## 9. Prace implementacyjne

| P | Zadanie | Dni | Do |
|---|---|---|---|
| P0 | `compute_critical_path_boundaries` (backtracking po Head, O(n+m)) + przełączenie filtra w `kbonacci.py` + testy bezstratności i niezmienniczości DP | 0,5 | 4.11 |
| P0 | NEH z akceleracją Taillarda (start dla wszystkich; potrzebne też w IG) | 0,5 | 5.11 |
| P0 | ILS: opcja startu NEH; logowanie statystyk iteracji (jsonl, próbkowane) | 0,5 | 5.11 |
| P0 | Plik BKS + ARPD do BKS w skryptach | 0,5 | 6.11 |
| P0 | Runner E0–E6 (konfiguracja, losowa kolejność, pula workerów = rdzenie P, metadane) | 1 | 7.11 |
| P0 | Biblioteka analiz (Friedman, Nemenyi, Holm, Wilcoxon, Page, A12, bootstrap) + skrypty rysunków i tabel LaTeX | 1 | 8.11 |
| P1 | Insert best-improvement z akceleracją Taillarda (baseline ILS i LS w IG) | 0,5 | 11.11 |
| P1 | IG Ruiza i Stützlego (d = 4, T = 0,4·Σp/(10nm), LS: ins / kb / ins + kb) | 1–1,5 | 14.11 |
| P1 | DP top-t + dokładna ocena (H5b) | 0,5 | 18.11 |
| P1 | Adaptacja k: reguła blokowa, VND, bandyta | 1 | 25.11 |
| P2 | numba dla Head/Tail, delt zamian, delt wstawień i C_max (go/no-go po E0) | 1–2 | opc. |
| P2 | Delty max-plus (T9) | 1 | opc. |
| P2 | Analiza pasma QUBO k-bonacciego (bez QPU) | 0,5 | opc. |

---

## 10. Harmonogram (wariant ICAPS) i punkty kontrolne

| Tydzień | Zadania | Wynik |
|---|---|---|
| 1 (2–8.11) | P0; E0; powtórka pilota z dokładnym filtrem; sprawdzenie CFP; PDF-y prac promotora; spotkanie: decyzje D1–D9 | infrastruktura gotowa |
| 2 (9–15.11) | E1, E2; Insert, IG; pisanie §3–5 (teoria nie zależy od wyników). ⚠️ 15.11 to też termin ICAISC (Hamming) | H1, H2, H5a rozstrzygnięte |
| 3 (16–22.11) | E3 (~8 h zegarowo), E5; analiza | **punkt kontrolny 22.11** |
| 4 (23–29.11) | E4; pisanie §1–2; rysunki 1–5 | pełna treść bez E6 |
| 5 (30.11–6.12) | E6 (~9 h zegarowo); analiza H6, H7; pełny szkic; **abstrakt do 7.12** | kompletny draft |
| 6 (7–14.12) | draft do promotora do 9.12; poprawki; anonimowy artefakt; **zgłoszenie 14.12** | submisja |

**Punkt kontrolny 22.11. Idziemy na ICAPS, jeśli spełnione są wszystkie warunki:**
1. Dokładny filtr, E0–E2 zrobione, a H1 i H2 rozstrzygnięte (w którąkolwiek stronę).
2. E3 dla t ∈ {10, 30} zrobione.
3. IG zaimplementowany i przetestowany.
4. Szkic §3–5 gotowy.

**Jeśli nie, przechodzimy na IJCAI.** Zyskujemy +4 tygodnie na E4 jako oś „AI” (adaptacyjne k), VRF i dłuższe budżety.
Abstrakt ~4.01, artykuł ~11–15.01 (do potwierdzenia). Limit to zwykle 7 stron + 2 strony referencji.

---

## 11. Ryzyka i plan B

| Ryzyko | P-stwo | Skutek | Środek zaradczy |
|---|---|---|---|
| Recenzent: „to tylko Dynasearch z limitem odległości” | wysokie | odrzucenie | teoria C1–C2 (pokrycie log₂n, bezstratny filtr, O(mn + m²k³)), prawo nasycenia H2, studium k; dokładny przegląd literatury (§13) |
| Brak wewnętrznego k* poza trywialnym n = 500 | średnie | słabszy przekaz H3 | przekaz „duże k ograniczone skalą bloku” (H2) + model kosztu; H3 zostaje dla n = 200 |
| IG wyraźnie lepszy | wysokie | słabsza pozycja SOTA | uczciwe pozycjonowanie; test komplementarności IG_ins + kb (H7b) |
| Python psuje pomiary czasu | średnie | zarzut metodologiczny | §7.2, analiza przy równej liczbie iteracji, model kosztu; opcjonalnie numba |
| Filtr blokowy (§14) a wcześniejsze prace | pewne | korekta opisu | decyzja z promotorem; w tym artykule mała ablacja: test ostatniej maszyny vs dokładne granice |
| Brak aktualnych BKS (sieć sesji blokuje strony) | średnie | ARPD tylko do LB i UB-1993 | BKS od promotora albo z literatury |
| Niepotwierdzone terminy | średnie | przesunięcie | sprawdzić CFP w tygodniu 1 |
| Kolizja z ICAISC (15.11) i ANNSIM (18.12) | wysokie | brak czasu | zdecydować priorytety (D9) |
| Prace promotora (Sci Rep 2026, IFAC 2025) zawierają już tę hierarchię | niskie/średnie | utrata nowości | przeczytać w tygodniu 1; w razie czego przesunąć akcent na C2–C5 |

---

## 12. Decyzje do podjęcia (z promotorem)

| # | Decyzja | Rekomendacja |
|---|---|---|
| D1 | Konferencja | ICAPS; punkt kontrolny 22.11 |
| D2 | Rozwiązanie startowe | NEH dla wszystkich (obecny ILS startuje z permutacji identycznościowej) |
| D3 | Budżety | t·n·m/2 ms, t ∈ {10, 30, (90)}; stałe ms tylko w suplemencie, dla ciągłości |
| D4 | Metaheurystyki | ILS + IG w tekście głównym; obecne SA („best-neighbor SA”) poza artykułem, bo jest niestandardowe |
| D5 | numba | go/no-go po E0, jeśli ≥ 10× w jeden dzień pracy |
| D6 | Filtr blokowy | dokładne granice ścieżki krytycznej + omówienie skutków dla wcześniejszych prac (§14) |
| D7 | Zakres „podsumowania” | tabela-taksonomia + akapit; bez osobnej sekcji o pracach kwantowych |
| D8 | Adaptacyjne k w ICAPS | krótko (reguła + bandyta); pełnie w IJCAI |
| D9 | Priorytety vs ICAISC 15.11 i ANNSIM 18.12 | do ustalenia |

---

## 13. Literatura do przeczytania lub sprawdzenia

- **Pilne:** Bożejko, Uchroński, Wodecki, Sci. Rep. 2026 (doi 10.1038/s41598-026-66284-9) i praca IFAC 2025 o nieprzecinających się
  przedziałach. Pytanie: czy zawierają hierarchię k / ograniczony zasięg? (sieć sesji blokuje doi.org i nature.com, potrzebne PDF-y)
- Ergun i Orlin 2006 (ORL 34(1)); Grosso, Della Croce, Tadei 2004 (ORL 32(1)): dokładne wyniki dla Dynasearch.
- Bożejko i Wodecki 2007 (multimoves, ORL): czy analizują ograniczony zasięg.
- Toth i Vigo 2003 (granular TS, INFORMS JoC); listy sąsiadów i Or-opt: analogia ograniczonego zasięgu.
- Mladenović i Hansen 1997 (VNS); Ropke i Pisinger 2006 (ALNS): baseline’y dla adaptacji k.
- Ruiz i Stützle 2007 (IG); Fernandez-Viagas, Ruiz, Framinan 2017 (przegląd SOTA PFSP); Taillard 1990 (akceleracja wstawień).
- Vallada, Ruiz, Framinan 2015 (benchmark VRF).
- Balzereit i in., arXiv 2409.00771 (parametryzowane lokalne przeszukiwanie; zweryfikować autorów).
- Statystyka: Demšar 2006 (CD), Page 1963 (test trendu), Vargha i Delaney 2000 (A12).
- Kombinatoryka: prawo najdłuższej serii (Erdős–Rényi) i źródło dla λ_k (liczby k-bonacciego, OEIS A000073 / A000078).

---

## 14. Ważne ustalenie poboczne: test granic bloków w dotychczasowym kodzie

`compute_block_boundaries` (`src/neighborhoods/accelerator.py`) uznaje pozycję j za granicę, gdy ostatnia maszyna zaczyna zadanie zaraz
po przedostatniej. Patrzy więc **tylko na dwie ostatnie maszyny**, a nie na ścieżkę krytyczną. Tymczasem twierdzenie (T5) wymaga granic
zmiany maszyny na ścieżce krytycznej. Sprawdziłem to 01.11 na losowych instancjach: 40 instancji na klasę, permutacje losowe
i optima lokalne tribonacciego.

| Klasa | Stan | Zamian poprawiających | Odrzuconych przez obecny test | Odrzuconych przez ścieżkę krytyczną |
|---|---|---|---|---|
| 20×5 | losowe / opt. lok. | 2 888 / 657 | 3,1% / 5,9% | 0 / 0 |
| 20×20 | losowe / opt. lok. | 3 791 / 1 134 | 2,1% / 2,7% | 0 / 0 |
| 50×5 | losowe / opt. lok. | 17 809 / 4 967 | 7,2% / 2,3% | 0 / 0 |
| 50×10 | losowe / opt. lok. | 20 626 / 5 722 | 3,8% / 4,9% | 0 / 0 |
| 50×20 | losowe / opt. lok. | 22 339 / 7 615 | 4,0% / 3,6% | 0 / 0 |

Odsetek kandydatów, które przechodzą filtr (optima lokalne tribonacciego):

| Klasa | k = 4: obecny / ścieżka kr. | k = n: obecny / ścieżka kr. | Liczba granic: obecny / ścieżka kr. |
|---|---|---|---|
| 20×5 | 57% / 27% | 81% / 47% | 7,5 / 2,5 |
| 50×5 | 52% / 11% | 79% / 46% | 15,5 / 2,6 |
| 50×20 | 69% / 44% | 92% / 85% | 22,4 / 10,5 |
| 100×10 | 44% / 15% | 82% / 61% | 26,3 / 6,2 |

**Wnioski:**
1. Dokładny filtr jest **jednocześnie poprawny i silniejszy**: zostawia 2–5× mniej kandydatów przy małym k. To podstawa wyniku O(mn + m²k³) (T6).
2. Ten sam test był używany w Adjacent, Fibonaccim, Dynasearch i Motzkinie oraz w QUBO we wcześniejszych pracach. W `windowed_qubo`
   (sekcja accelerators) jest opisany jako twierdzenie o granicach „H[m][u] = H[m−1][u] + p”.
   Tamte wyniki nie są błędne: to przebiegi z nieco mniejszym, heurystycznie przyciętym sąsiedztwem. **Opis „nie może poprawić, więc pomijamy”
   jest jednak nieścisły.** Co z tym zrobić w pracach w recenzji i w przygotowaniu, to decyzja z promotorem.
3. W tym artykule: dokładne granice dla wszystkich wariantów. Opcjonalnie krótka ablacja „przybliżony test ostatniej maszyny vs dokładne granice”.

Odtworzenie: `python scripts/check_block_filter.py` (obie tabele, ~5 min). Przy P0 dodać test bezstratności do `tests/`.

---

## 15. Status szczegółowy (01.11.2026)

- ✅ `src/neighborhoods/classical/kbonacci.py`: tryby `tribonacci`, `tetranacci`, `kbonacci_k<K>` i `kbonacci_kn`; statystyki addytywności.
- ✅ `tests/test_kbonacci.py`: 96 przypadków; cały zestaw 121/121.
- ✅ Dane Taillard (12 × 10) w `data/` (gitignored), zgodne z generatorem Taillarda. Źródło: aosabook/500lines.
- ✅ Pilot `scripts/run_kbonacci_pilot.py`, wyniki w `articles/kbonacci/pilot/`. ⚠️ Liczony ze starym filtrem, do powtórzenia.
- ✅ Szkice `main.tex` i `wersja_pl.tex` (niekompilowane, brak LaTeX-a w sesji).
- ⏳ P0 (§9), prace promotora, CFP, BKS.

**Wyniki pilota (ILS, 1000 ms, 4 instancje × 3 ziarna, RPD [%] do UB z 1993 r., stary filtr):**

| | 20×5 | 20×10 | 50×10 |
|---|---|---|---|
| Adjacent | 1,59 | 3,26 | 11,73 |
| k = 2 (Fib) | 1,31 | 2,46 | 8,37 |
| k = 3 (trib) | 0,92 | 1,84 | 5,94 |
| k = 4 | 0,89 | 1,43 | 5,51 |
| k = 6 | 0,74 | 1,09 | 4,85 |
| k = 8 | 0,72 | 1,35 | 4,66 |
| k = n | **0,55** | **0,99** | **3,62** |
| stary Dynasearch (rekurencyjny) | 0,94 | 1,49 | 5,30 |

---

## 16. Estymata zasobów: 2 algorytmy × pełny Taillard (01.11.2026)

Założenia: 120 instancji × 5 ziaren × 2 algorytmy (ILS + SA albo ILS + IG); przebiegi z limitem czasu, więc koszt CPU = suma budżetów.
Warianty: V = 7 (k ∈ {2, 3, 4, 8, n} + Adjacent + Insert) albo V = 10 (k ∈ {2, 3, 4, 6, 8, 12, 16, n} + Adjacent + Insert).
k = n i Motzkin przy n = 500 to n/a, bo jedna ocena trwa ~36 s (ekstrapolacja).

**Lokalnie (M4 Pro, 8 rdzeni P, +10% narzutu):**

| Protokół | h CPU / (wariant × alg.) | V = 7 | V = 10 |
|---|---|---|---|
| A: 6 budżetów 0,1–10 s (jak we wcześniejszych pracach) | 3,1 | 43 h CPU → **~6 h** | 62 h CPU → **~8,5 h** |
| B: t·n·m/2, t = 30 | 4,6 | 64 h CPU → **~9 h** | 92 h CPU → **~12,5 h** |
| B: t = 10 i 30 | 6,1 | 85 h CPU → **~12 h** | 122 h CPU → **~17 h** |

Kontener chmurowy (4 rdzenie) wychodzi ~2× dłużej i daje zaszumione pomiary czasu, więc służy tylko do testów, nie do tabel.

**D-Wave Advantage** (dane z `windowed_qubo`: 30 ms rozliczanego czasu QPU na wywołanie przy num_reads = 100, ~2,1 s zegarowo; quota ~400 s/mies.):

| Scenariusz (4 sąsiedztwa × 2 alg. × 120 inst. × 5 ziaren) | Wywołania | QPU | Quota | Zegar |
|---|---|---|---|---|
| 1 iteracja na przebieg (reżim jednego ruchu) | 4 800 | 144 s | 0,36 mies. | ~3 h |
| 3 iteracje | 14 400 | 432 s | 1,1 mies. | ~8 h |
| 10 iteracji | 48 000 | 1 440 s | 3,6 mies. | ~28 h |

Z dokładnym filtrem QUBO k-bonacciego ma ≤ (m−1)(k−1)(k+2)/2 zmiennych niezależnie od n. Dla m = 20 i k = 4 to ≤ 171, czyli na granicy embeddingu;
dla k ≥ 6 przy m = 20 potrzebne okna. Gotowa baza: `quantum_dynasearch` z `L_max_dynasearch = k`.

**IQM Resonance** (dane z `gate_qaoa`: ~0,95 kredytu/obwód na garnet, ~0,68 na sirius; 1 kredyt ≈ 2 s ≈ 0,60 USD; darmowe 30 kredytów/mies.;
garnet 20 kubitów, sirius 16):

| Scenariusz (4 sąsiedztwa × 2 alg., 1 ruch na przebieg) | Obwody | Kredyty (garnet / sirius) | Darmowe miesiące | USD |
|---|---|---|---|---|
| tai20_5, 10 instancji, 1 ziarno | 80 | 76 / 54 | 2,5 / 1,8 | 46 / 33 |
| tai20_5, 10 instancji, 5 ziaren | 400 | 380 / 272 | 12,7 / 9,1 | 228 / 163 |
| pełny Taillard | 4 800 | 4 560 / 3 264 | 152 / 109 | 2 736 / 1 958 |

Przy m > 5 QUBO przekracza 16–20 kubitów, więc potrzebne są okna, a koszt rośnie kilkukrotnie. Pełny Taillard na IQM jest **nierealny**.
Sensowne jest tylko demo na tai20_5.
