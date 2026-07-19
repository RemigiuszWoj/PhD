# Skalowalne kwantowe przeszukiwanie sąsiedztwa dla permutacyjnego problemu przepływowego przez okienkową dekompozycję QUBO

**Remigiusz Wojewódzki, Wojciech Bożejko** — Katedra Sterowania i Obliczeń
Kwantowych, WIT, Politechnika Wrocławska. Cel: Computers & Operations Research.

> Polska wersja robocza do szybkiego czytania (2026-07-17). Wiążąca jest
> wersja angielska w main.tex; tu treść wiernie przełożona i lekko zagęszczona.

## Abstrakt

Rozszerzamy framework kwantowych sąsiedztw dla permutacyjnego problemu
przepływowego (PFSP) o trzecią klasę — *quantum QUBO enhanced* — działającą
na znacznie większych instancjach, niż pozwalały pierwotne formulacje na QPU
D-Wave. Dotychczasowy framework: cztery sąsiedztwa klasyczne (Adjacent,
Fibonacci, Dynasearch, Motzkin) i cztery kwantowe QUBO ograniczone do n=20.
Klasa enhanced usuwa dwa ograniczenia: zbędny filtr delty w strukturach
liniowych (Adjacent/Fibonacci) — co rozszerza stosowalność QPU do n≤200
i n≤500 — oraz wprowadza okienkową dekompozycję QUBO dla struktur
kwadratowych (Dynasearch/Motzkin), przesuwając wykonalność do n≤50.
Oba frameworki metaheurystyczne (ILS z dywersyfikacją tabu przez elitarną
pulę BackTrackJumpList oraz Termodynamiczne-SA — reguła w stylu SA
aplikująca akceptację Metropolisa do najlepszego sąsiada z pełnego skanu —
z adaptacyjnym podgrzewaniem) używają akceleratora własności blokowych,
tnącego liczbę zmiennych QUBO o 30–60%. Dwanaście sąsiedztw oceniamy na
benchmarkach Taillarda przy sześciu budżetach (100–10000 ms). Pomiary QPU
na Advantage_system4.1 potwierdzają skalowanie: średni RPD fibonacciego
enhanced spada monotonicznie z 31.7% (n=20) do 15.0% (n=200), a luka do
klasycznego fibonacciego zwęża się z 20.4 pp (n=20) do 4.6 pp (n=500),
gdzie wariant kwantowy dorównuje klasycznemu adjacent. Okienkowanie jest
niemal darmowe tam, gdzie interakcje swapów są lokalne (dynasearch),
a kosztuje ok. 4 pp — rosnąco z budżetem — tam, gdzie są globalne (motzkin).

**Highlights:** klasa enhanced łamie barierę n=20 na QPU D-Wave; usunięcie
filtra delty rozszerza Adjacent/Fibonacci do n≤200/n≤500; okienkowa
dekompozycja umożliwia Dynasearch i Motzkin do n=50; akcelerator
Smutnickiego tnie zmienne QUBO o 30–60%; luka kwantowo-klasyczna zwęża się
z 20.4 pp (n=20) do 4.6 pp (n=500); okienkowanie prawie darmowe dla
struktur lokalnych, strata narasta dla globalnych.

## 1. Wstęp

PFSP szuka permutacji n zadań na m maszynach szeregowo minimalizującej
makespan C_max. Problem jest NP-trudny dla m≥3 (Garey–Johnson–Sethi 1976),
a dla dużych instancji najlepsze rozwiązania dają metaheurystyki oparte na
sąsiedztwach (monografia Bożejko 2010). Wybór struktury sąsiedztwa — zbioru
rozwiązań osiągalnych jednym dopuszczalnym ruchem (van Laarhoven i in.
1992) — jest głównym motorem jakości.

**Klasyczna linia flow shopu:** Johnson 1954 rozwiązuje przypadek dwóch
maszyn wielomianowo; heurystyka NEH (1983) pozostaje standardowym startem.
Linia metaheurystyczna przeszła przez SA (Osman–Potts 1989) i GA (Reeves
1995), ale jej kręgosłupem jest tabu search: implementacja Taillarda
(1990), blokowy fast tabu Nowickiego–Smutnickiego (1996a) i przyspieszony
następca Grabowskiego–Wodeckiego (2004) kolejno cięły czas i dystans do
optimum na benchmarku Taillarda (1993). Konsolidacja: przegląd
Ruiza–Maroto (2005) i iterated greedy Ruiza–Stützlego (2007), który
pokazał, że prosta pętla destrukcja–rekonstrukcja dorównuje bardziej
wyszukanym maszynom.

**Linia otoczeń:** w każdym mocnym algorytmie PFSP powtarzają się dwa
składniki — struktura ruchu i własności ścieżki krytycznej, które ją
przycinają. Dynasearch (Congram–Potts–van de Velde 2002) pokazał, że
wykładniczo duży zbiór parami niezależnych ruchów da się przeszukać
dokładnie przez DP w czasie wielomianowym — jednostką przeszukiwania staje
się ruch złożony. Teorię multimoves rozwinęli Bożejko–Wodecki (2007),
a paradygmat rozrósł się w very large-scale neighborhood search (Ahuja
i in. 2002). Własności blokowe (Nowicki–Smutnicki, Smutnicki 1998,
Grabowski–Wodecki) atakują to samo wąskie gardło z drugiej strony — mówią,
których kandydatów można pominąć hurtem. Nasze sąsiedztwa Fibonacci
i Motzkin (ICAISC 2026; EXIT 2025) kontynuują linię ruchów złożonych:
zbiory ruchów zliczane klasycznymi ciągami, przeszukiwane przez DP —
i, co ten artykuł wykorzystuje, wyrażalne jako QUBO.

**Kwanty w schedulingu** weszły od job shopu: Venturelli i in. (annealing
całych instancji jako QUBO), Kurowski i in. (hybrydy z heurystykami),
Amaro i in. (obwody wariacyjne) — wszystkie kodują cały harmonogram
naraz, co ogranicza do rozmiarów zabawkowych. Nasza alternatywa: QPU
**wewnątrz** metaheurystyki zamiast w jej miejsce — QUBO staje się
sąsiedztwo, nie harmonogram.

W pracy BPASTS sformułowaliśmy cztery klasyczne sąsiedztwa jako QUBO
i wykonaliśmy je na QPU D-Wave; zakres ograniczał się do n=20 (pojemność
embeddingu ~170–180 zmiennych logicznych dla gęstych grafów na Pegasus P16,
5627 aktywnych kubitów). Ten artykuł usuwa oba blokery: (1) zbędny filtr
delty — obejście pod klasyczny symulator, szkodliwe na QPU, bo odrzuca
informację o strukturze kar, podczas gdy QPU sam przypisuje x_k=0 ruchom
niepoprawiającym; dla Adjacent/Fibonacci (K=n−1 niezależnie od filtra)
usunięcie nic nie kosztuje; (2) sufit zmiennych O(n²) dla
Dynasearch/Motzkin — wprowadzamy dekompozycję okienkową: nakładające się
okna rozmiaru w≤19, QUBO per okno, scalanie zachłanne. Razem: dwanaście
sąsiedztw w trzech klasach, po raz pierwszy pokrywających pełen zakres
Taillarda.

**Kontrybucje:** okienkowa dekompozycja QUBO (QPU do n=50, automatyczny
dobór okna z pojemności sprzętu); formulacje bez filtra delty (Adjacent
n≤200, Fibonacci n≤500); akcelerator blokowy Smutnickiego we wszystkich
dwunastu sąsiedztwach (−30–60% zmiennych); BackTrackJumpList zamiast
restartów losowych w ILS; pierwsze porównanie trzech klas na
n∈{20,50,100,200,500} przy identycznych frameworkach i budżetach.

## 2. Definicja problemu

Sekcja ustala model i notację — materiał standardowy, przytaczany dla
samodzielności tekstu, w notacji klasycznej pracy Nowickiego–Smutnickiego
(1996a).

**PFSP:** zbiór zadań J={1..n}, maszyn M={1..m}; zadanie j to ciąg
operacji O_1j..O_mj, operacja O_ij na maszynie i trwa p_ij>0 bez przerwań.
Każda maszyna przetwarza w tej samej kolejności — porządek to permutacja
π∈Π. Czasy zakończenia: C_{i,π(j)} = max{C_{i−1,π(j)}, C_{i,π(j−1)}} +
p_{i,π(j)}, z zerowymi warunkami brzegowymi. Makespan C_max(π)=C_{m,π(n)};
szukamy π* minimalizującej C_max. Równoważnie makespan to długość
najdłuższej (krytycznej) ścieżki w grafie siatkowym:
C_max(π) = max_{1=j_0≤j_1≤…≤j_m=n} Σ_i Σ_{j=j_{i−1}}^{j_i} p_{i,π(j)},
a maksymalne podciągi ścieżki krytycznej na jednej maszynie tworzą
**bloki** — strukturę, którą sekcja 3.2 zamienia w filtr kandydatów.
Pełny opis: Nowicki–Smutnicki 1996a i monografia Smutnickiego 2012;
relacje między permutacjami mają też ujęcie algebraiczne przez funkcje
dodatnio określone na grupach permutacji (Bożejko–Bożejko 2015), którego
tu nie potrzebujemy.

**Dekompozycja Head–Tail** (Wodecki–Bożejko 2004): delta ruchu w O(1)
z macierzy H[i][j] (najwcześniejsze zakończenia od lewej) i T[i][j]
(wkłady od prawej), budowanych w O(mn) raz na iterację.

**Metaheurystyki.** Samo sąsiedztwo to jeszcze nie algorytm — potrzebna
jest warstwa sterująca akceptująca ruchy i wyciągająca trajektorię
z optimów lokalnych. Żeby nie zaburzała porównania, każde sąsiedztwo
działa w tych samych dwóch frameworkach z identycznymi parametrami —
jedyną zmienną jest sąsiedztwo.

*ILS* (Lourenço i in. 2003) działa z listą tabu (Glover 1989) o tenurze
τ=10 i kryterium aspiracji (ruch tabu przyjęty, jeśli ściśle poprawia
globalne minimum), w duchu fast taboo search Nowickiego–Smutnickiego
(1996b, job shop).

*SA* (Kirkpatrick 1983) działa z T₀=1000, α=0.99, podłogą T_min=100
i dywersyfikacją reheat-kick. **Nasz wariant różni się od kanonicznego SA
kryterium eksploracji** (komponent frameworku Franzina–Stützle 2019):
kanoniczne SA losuje sąsiada; Ishibuchi i in. (1995) — właśnie dla flow
shopu — poddawali Metropolisowi najlepszego z losowej próbki, co uodparnia
SA na harmonogram chłodzenia. My doprowadzamy mechanizm do granicy
wyczerpującej: test dostaje najlepszy ruch CAŁEGO otoczenia ze wspólnego
oraculum. Granica nie jest niewinna: po odrzuceniu losowej próbki losuje
się świeżą, a deterministyczny pełny skan re-proponuje ten sam ruch przy
coraz niższym T — to patologia zamarzania, którą diagnozuje sekcja 2.2.
Konstrukcję wymusza porównanie: QPU zwraca najlepszy ruch złożony, nie
losowy. Ewaluacja pełnego otoczenia pod akceptacją Metropolisa to
ugruntowany reżim w sprzęcie QUBO (parallel-trial Digital Annealera —
Aramon i in. 2019) i w rodzinie rejection-free (Rosenthal i in. 2021,
następca losowany z wag zamiast argmax). Według naszej wiedzy konkretny
łańcuch "wyczerpujący argmax + Metropolis" nie został nazwany ani
przeanalizowany — dokumentujemy go empirycznie.

### 2.1. BackTrackJumpList (elitarna dywersyfikacja)

Restart losowy wyrzuca wszystko, czego szukanie się nauczyło; BTJL to
zachowuje. Pula k=10 najlepszych różnych permutacji; przy konflikcie tabu
bez aspiracji algorytm aplikuje perturbację double-bridge (4-opt) do
kolejnej elity (round-robin). Double-bridge tnie π w trzech losowych
punktach i skleja A|C|B|D — ruch poza otoczeniem 2-opt, więc szukanie
naprawdę opuszcza basen. Round-robin nie tłucze wciąż globalnego
najlepszego; pula aktualizowana przy każdej ścisłej poprawie; przed
pierwszą elitą fallbackiem jest restart losowy.

### 2.2. SA z reheat-kick

Wspólne oracula zmuszają krok SA do konsumpcji NAJLEPSZEGO ruchu.
Z deterministycznym oraculum szukanie zapada się w minimum lokalnym X:
oraculum daje najlepszego sąsiada X′ (Δ>0), Metropolis czasem przyjmie,
z X′ oraculum wraca do X (Δ<0, zawsze). Trajektoria wpada w 2-cykl
X↔X′, a podgrzewanie samo go nie łamie — oraculum zwraca tę samą parę
niezależnie od T. Rozwiązanie sprzęga reheat z perturbacją w przestrzeni
rozwiązań: przy stagnacji s=100 ms bez poprawy (1) T←T₀ (mnożnikowy
reheat z wychłodzonego T nic nie daje), (2) bieżące rozwiązanie :=
double-bridge kolejnej elity, (3) reset licznika stagnacji. Podłoga
T_min=100 utrzymuje ruchliwość pod górę. Dwucykl znika, a SA zbiega
z trajektoriami zależnymi od seeda tam, gdzie pierwotny harmonogram
zamarzał w pierwszym minimum.

## 3. Taksonomia sąsiedztw

Przechodzimy dwanaście sąsiedztw w trzech klasach; definicje klasyczne są
znane — przypominamy je, bo konstrukcje QUBO budują wprost na ich zbiorach
ruchów.

### 3.1. Klasyczne

- **Adjacent (N_adj):** ruch zamienia zadania na pozycjach i oraz i+1,
  więc |N_adj|=n−1. W literaturze znany jako API (adjacent pairwise
  interchange); hierarchia uogólnionych wymian — Della Croce 1995.
- **Fibonacci (N_fib):** ruch aplikuje jednocześnie zbiór niekolidujących
  swapów sąsiednich (S poprawny, gdy i∈S ⇒ i+1∉S); liczba takich
  podzbiorów to F_{n+1} — stąd nazwa; wprowadzone w ICAISC 2026.
  Optymalny podzbiór wybiera DP w O(n).
- **Dynasearch (N_dyn):** swapy końców odcinków dowolnej długości —
  dla pary (i,j) zamiana π(i) z π(j), wnętrze bez zmian. DP (Congram
  2002) wybiera podzbiór par parami nienakładających się przedziałowo,
  łącznie minimalizując C_max. K=O(n²) kandydatów, DP w O(n³),
  akcelerowane filtrem NPI.
- **Motzkin (N_motz):** te same swapy końców co dynasearch, ale
  dopuszczalne podzbiory wg kombinatoryki ścieżek Motzkina (Motzkin
  1948; SOCO 2026; EXIT 2025): pary nie mogą się krzyżować ani dzielić
  końca, ścisłe zagnieżdżenie dozwolone. Liczba podzbiorów = liczba
  Motzkina M_n (M_5=21, M_10=4862, M_20≈1.7·10^10). DP w O(n³).

### 3.2. Akceleratory blokowe

Wszystkie sąsiedztwa stosują akcelerator Smutnickiego (1998) przed
ewaluacją/budową QUBO. Pozycja u jest granicą bloku, gdy maszyna m jest
zajęta tuż przed π(u+1): H[m][u] = H[m−1][u] + p_{m,π(u+1)}. Zbiór granic
U(π) liczony w O(mn) z gotowej macierzy H. Twierdzenie: swap pozycji
(i,j) może ściśle poprawić C_max tylko, gdy U(π)∩[i,j]≠∅.
Dla adjacent/fibonacci: kandydat i przechodzi tylko gdy i∈U(π) — z n−1
kandydatów zostaje |U(π)| (typowo O(m); na tai z m=5, n=100 średnio
4–8 z 99). Dla dynasearch/motzkin (NPI — non-adjacent pairwise
interchange, Della Croce 1995): filtr rozpiętości tnie O(n²) do
O(n·|U|); w QUBO tylko przefiltrowane pary stają się zmiennymi, co
wprost zmniejsza koszt embeddingu. Na Taillardzie filtr usuwa 35–65%
par kandydackich.

### 3.3. Kwantowe QUBO (oryginalne)

Cztery klasyczne przeszukiwania przeformułowane jako QUBO na QPU D-Wave
Advantage przez EmbeddingComposite (za BPASTS): x_k=1 ⇔ swap k wybrany;
QUBO koduje cel (Σδ_k x_k) i wykonalność (co najwyżej jeden swap /
niekonfliktowe swapy). Wykonalne dla n≤20.

### 3.4. Kwantowe QUBO enhanced

**Pełny QUBO bez filtra delty (Adjacent, Fibonacci):** oryginalne
formulacje odfiltrowywały δ_k≥0 pod symulator; na QPU filtr szkodzi
(usuwa strukturę kar), a jego brak nic nie kosztuje przy K=n−1.
Adjacent enhanced (gęsty Q) do n≤200, Fibonacci enhanced (tridiagonalny
Q) do n≤500.

**Okienkowa dekompozycja (Dynasearch, Motzkin):** pełny QUBO ma K=O(n²)
— ponad pojemność QPU dla n>20. Okno ℓ pokrywa pozycje [s_ℓ, s_ℓ+w),
s_ℓ = ℓ·⌊w(1−ρ)⌋, overlap ρ=0.5. QUBO per okno nad C(w,2) parami
(po filtrze blokowym). Rozmiar okna z warunku K(w)=w(w−1)/2 ≤ C_eff=180
(konserwatywna pojemność Advantage dla gęstych QUBO): w_max = 19.
Okna sekwencyjnie; scalanie: najlepszy ruch z każdego okna, konflikty
rozstrzygane zachłannie po |δ_k|. Umożliwia n≤50.

**Tabela wykonalności (skrót):** klasyczne — K:O(m)..O(nm), n_max=∞, CPU;
kwantowe oryginalne — K: n−1 (adj/fib), O(n²) (dyn/motz), n_max=20, QPU;
enhanced — adj n−1→n_max=200, fib n−1→500, dyn/motz ≤171 okienkowane→50.

## 4. Formulacje QUBO

Forma ogólna: min x^T Q x, x∈{0,1}^K, z przejściem do Isinga
H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j (przegląd: Lucas 2014). Diagonala
Q_kk=δ_k (delta z Head–Tail); pozadiagonalne Q_kl=λ karzą konflikt,
λ = (Σ_k |δ_k|) + 1 — dominuje cel i wymusza wykonalność.

- **Adjacent enh.:** one-hot — Q_ii=δ_i, Q_ij=2λ dla i≠j; dokładnie
  jeden swap.
- **Fibonacci enh.:** QUBO tridiagonalny — Q_ii=δ_i, Q_{i,i+1}=λ;
  liczba rozwiązań dopuszczalnych = F_{n+1}.
- **Dynasearch enh.:** Q_kl=λ ⇔ pary przedziałowo nachodzą
  (max(i₁,i₂) ≤ min(j₁,j₂)); nienachodzące niezależne — ruchy multi-swap.
- **Motzkin enh.:** Q_kl=λ ⇔ konflikt wg reguł Motzkina (wspólny koniec
  lub krzyżowanie); zagnieżdżone i rozłączne wolne; liczba rozwiązań = M_n.

## 5. Setup eksperymentalny

Projekt trzyma wszystko stałe poza sąsiedztwem — różnice w tabelach
przypisywalne wyłącznie strukturze ruchu. **Benchmarki:** Taillard,
n∈{20,50,100,200,500}, m∈{5,10,20}, po 10 instancji na konfigurację,
5 seedów. Sześć budżetów od 100 do 10000 ms; klasyczne przy wszystkich
sześciu, kampanie QPU raportują tl=1000 i 5000 w tabelach, konwergencja
100–5000 dla enhanced. **Sprzęt:** klasyka na Apple M4 Pro; kwanty na
D-Wave Advantage_system4.1 (Pegasus P16, 5627 kubitów, 40279 sprzęgieł),
EmbeddingComposite, num_reads=100 (per okno). Praktyczna pojemność
gęstych grafów ~190–200 zmiennych logicznych (łańcuchy ~30 kubitów
fizycznych); granica miękka — K₁₉₉ adjacent enhanced weszła w 50/50
runów, ~190-zmiennowy dynasearch padał sporadycznie, a porażki
przechodziły przy retry; grafy łańcuchowe (fibonacci) embedują się
trywialnie do n≤500. Tożsamość solvera potwierdzona programistycznie.
Klasyczna ewaluacja QUBO: SimulatedAnnealingSampler (dimod).
**Metryki:** RPD = (C_max−C*)/C* ×100% względem najlepszych znanych;
istotność: dwustronny Wilcoxon dla par wariantów (α=0.05).
**Ograniczenia instancji:** oryginalne kwantowe tylko n=20; adj enh
n≤200; fib enh n≤500; dyn/motz enh n≤50 (w=19, ρ=0.5).

## 6. Wyniki i dyskusja

Najpierw baseline klasyczny, potem kampanie QPU przy n=20, na końcu
skalowanie do n=500. Każda liczba kwantowa pochodzi z realnych runów
Advantage_system4.1 — niczego nie symulujemy.

### Tabela 2 — wszystkie 12 sąsiedztw, tai20×5, tl=1000 ms (RPD %, ILS/SA)

| Klasa | Sąsiedztwo | ILS | SA |
|---|---|---|---|
| Klasyczne | Adjacent | 3.89 | 11.28 |
| | Fibonacci | 4.09 | 15.00 |
| | Dynasearch | 3.32 | 4.19 |
| | Motzkin | 3.19 | 4.00 |
| Quantum QUBO | Adjacent | 25.07 | 25.04 |
| | Fibonacci | 23.79 | 23.46 |
| | Dynasearch | 26.93 | 26.35 |
| | Motzkin | 21.66 | 21.50 |
| Enhanced | Adjacent | 25.04 | 25.04 |
| | Fibonacci | 23.60 | 23.66 |
| | Dynasearch | 25.56 | 25.71 |
| | Motzkin | 25.98 | 25.45 |

Caption-info: latencja sieci+kolejki > tl, więc każdy run kwantowy =
dokładnie 1 iteracja (mediana rejestrowana per run) — wartości mierzą
pojedynczy złożony ruch QPU, nieporównywalne wprost z klasyką.
Dynasearch: 48/50 runów (2 porażki embeddingu ~190 gęstych zmiennych).
Enhanced: 3 okna na wywołanie, zero porażek embeddingu w 400 runach.

### Tabela 3 — baseline klasyczny, tl=10 s, n=20 (ILS/SA)

| | m=5 | | m=10 | | m=20 | |
|---|---|---|---|---|---|---|
| Adjacent | 3.41 | 6.98 | 11.22 | 15.61 | 22.58 | 26.27 |
| Fibonacci | 3.82 | 9.24 | 11.23 | 17.81 | 22.17 | 26.82 |
| Dynasearch | 2.68 | 3.36 | 9.99 | 10.75 | 21.41 | 21.99 |
| Motzkin | 2.64 | 3.25 | 10.21 | 10.68 | 21.46 | 21.73 |

Dynasearch i motzkin prowadzą przy każdym m; przewaga rośnie z m
(0.7–1.2 pp przy m=5, 1.2–1.4 przy m=20). SA adjacent/fibonacci tracą
7–10 pp do ILS przy m=5 — oraculum best-move nad rzadkim sąsiedztwem
więzi SA w wąskich basenach, które kick tylko częściowo rozwiązuje;
SA dyn/motz w granicach 0.6–0.7 pp od ILS.

### Kwanty na QPU przy n=20

Pierwsze pomiary w protokole z instrumentacją: każdy run rejestruje
liczbę iteracji, więc reżim porównania jest jawny. Przy tl=1000 każdy
run kwantowy kończy dokładnie 1 iterację (pojedyncze wywołanie ~2.1 s
ściennie dla adj/fib, więcej dla reszty). Dwie obserwacje przeżywają ten
reżim: (1) ranking odwraca się względem klasyki — motzkin prowadzi
(21.7%), potem fibonacci (23.8), adjacent (25.1), dynasearch (26.9);
gdy budżet starcza na jeden ruch, wygrywa najbogatszy dopuszczalny zbiór
ruchów, a niekrzyżujące łuki Motzkina są najbogatsze. (2) Jakość podąża
za liczbą iteracji: przy tl=5000 adj/fib robią 3 iteracje i poprawiają
o 2.8/3.7 pp (fibonacci 19.9%, motzkin 19.5%); dynasearch stoi na
1 iteracji. Tabela 6 rozpisuje to per budżet bez poolingu — płaskie
wiersze enhanced między 100 a 1000 ms to reżim 1 iteracji w tabeli
zamiast na wykresie. Dynasearch obnaża też granicę embeddingu: bez
filtra delty ~190 gęsto sprzężonych zmiennych, 4/200 runów bez
embeddingu; motzkin przy podobnej liczbie zmiennych, ale rzadszym grafie
konfliktów, embedował się zawsze — wykonalność QPU rządzi się gęstością
grafu konfliktów, nie liczbą zmiennych.

### Anatomia wywołania QPU (Tabela 5; 1332 wywołania, num_reads=100)

| Sąsiedztwo | Ściennie/wywołanie | Dostęp QPU | Udział QPU |
|---|---|---|---|
| Q-Adjacent | 2.12 s | 30 ms | 1.4% |
| Q-Fibonacci | 2.06 s | 30 ms | 1.5% |
| Q-Motzkin | 18.5 s | 30 ms | 0.16% |
| Q-Dynasearch | 140 s | 30 ms | 0.02% |

Rozliczany dostęp ~30 ms (16 programowanie + 14 próbkowanie; 20 µs anneal,
102 µs odczyt per próbka). Reszta to infrastruktura: transfer i kolejka
Leap (adj/fib), budowa QUBO O(mn³) po stronie klienta (motzkin),
minorminer na gęstym grafie (dynasearch). Te proporcje to rdzeń
protokołu benchmarkowego, za którym argumentuje artykuł: porównanie
ścienne, które je ignoruje, mierzy chmurę, nie procesor.

### Wiersze enhanced (kontrolowany pomiar efektu filtra delty)

Dla sąsiedztw liniowych enhanced odtwarza oryginały niemal dokładnie
(adjacent 25.04 vs 25.05, fibonacci 23.60/23.66 vs 23.79/23.46; Wilcoxon
po parach instancja×seed p≥0.25) — ich QUBO nigdy nie były filtrowane,
więc zgoda działa jako replikacja między kampaniami. Okienkowany
dynasearch ZYSKUJE 0.9 pp nad pełną formulacją (25.78 vs 26.64 parowane,
p=0.012), eliminując przy tym porażki embeddingu (0 w 400 runach vs 4%).
Okienkowany motzkin TRACI 4.1 pp (25.72 vs 21.58, p<10⁻⁴). Asymetria ma
odczyt strukturalny: konflikty dynasearcha są lokalne (nakładanie
odcinków) — granica okna mało odcina; siła motzkina leży w zagnieżdżonych
systemach łuków rozpinających całą permutację — okna tną właśnie je.
Budżet wyostrza obraz: przy tl=5000 okienkowany dynasearch trzyma
przewagę (25.07 vs 26.15, p=0.005), a luka motzkina rośnie z 4.1 do
5.6 pp (25.06 vs 19.45, p<10⁻⁴): pełna formulacja zamienia każdą dodatkową
iterację w zyski zagnieżdżeń, których żaden ruch wewnątrzokienny nie
wyrazi — strata okienkowania się kumuluje, zamiast amortyzować.

### Tabela 4 — skalowanie (RPD %, ILS, tl=5000; m=10 dla n≤200, m=20 przy n=500)

| Sąsiedztwo | n=20 | n=50 | n=100 | n=200 | n=500 |
|---|---|---|---|---|---|
| Klasyczny Adjacent | 11.61 | 10.03 | 9.54 | 10.37 | 15.36 |
| Klasyczny Fibonacci | 11.27 | 7.74 | 6.15 | 6.19 | 11.12 |
| Klasyczny Dynasearch | 10.23 | 5.39 | 4.67 | 11.63 | 14.53† |
| Klasyczny Motzkin | 10.45 | 13.49 | 14.21 | 13.80 | 16.26† |
| Q-Adjacent | 34.91 | n/a | n/a | n/a | n/a |
| Q-Fibonacci | 31.60 | n/a | n/a | n/a | n/a |
| Q-Dynasearch | 39.30 | n/a | n/a | n/a | n/a |
| Q-Motzkin | 34.41 | n/a | n/a | n/a | n/a |
| Q-Adjacent Enh. | 34.86 | 26.70 | 21.73 | 16.40 | n/a |
| Q-Fibonacci Enh. | 31.68 | 23.56 | 19.62 | 14.99 | 15.75 |
| Q-Dynasearch Enh. | 38.55 | 26.37 | n/a | n/a | n/a |
| Q-Motzkin Enh. | 38.44 | 26.03 | n/a | n/a | n/a |

† nieporównywalne wprost (pojedyncze wywołanie DP przekracza każdy
budżet — 1 iteracja, wartości niezależne od budżetu). n/a = poza
zakresem traktowalności (embedding K₄₉₉ i pełne QUBO O(n²) poza n=20;
bariera ścienna okienkowanych poza n=50).

**Skalowanie:** warianty enhanced budowano, by choć utrzymały poziom przy
rosnącym n. Pomiary pokazują więcej: RPD kwantowy spada monotonicznie
z rozmiarem. Fibonacci enhanced: 31.7→23.6→19.6→15.0% (n=20→200);
adjacent enhanced tym samym torem ~1.5 pp za nim (34.9→16.4). Luka do
klasycznego fibonacciego zwęża się z 20.4 pp (n=20) przez 13.5 (n=100)
i 8.8 (n=200) do 4.6 przy n=500, gdzie wariant kwantowy (15.75%) osiąga
parytet z klasycznym adjacent (15.36%). Prawy koniec krzywej jest
osiągalny dzięki tridiagonalności: embedding w milisekundach nawet przy
K₄₉₉, trzy iteracje mieszczą się w 5 s przy każdym rozmiarze; adjacent
spada do 1 iteracji od n=100 (minorminer gęstego QUBO zjada budżet).
Kolumna n=20 replikuje asymetrię okien przy m=10: okienkowany dynasearch
znów wyprzedza pełny (38.6 vs 39.3, p=0.18 przy 50 parach — bez
istotności), okienkowany motzkin znów płaci ~4 pp (38.4 vs 34.4,
p<10⁻⁴). Gęsty one-hot skaluje się dalej, niż sugerowała kampania n=20:
K₁₉₉ adjacenta enhanced — tuż za szacunkiem pojemności ~190 —
embedował się w 50/50 runów (~8 s minorminera); porażki n=20 okazały
się stochastyczne (trzy nieudane przeszły przy retry). Twarda ściana:
K₄₉₉ (adjacent n=500) ponad pojemność klik Pegasusa; okolice 200
zmiennych to loteria o korzystnych szansach, nie klif.

**Klasyczny crossover:** przy n=100 wygrywa jeszcze dynasearch (4.67 vs
6.15 fibonacciego); przy n=200 odwrócenie (6.19 vs 11.63) — DP O(n²)
zjada budżet; motzkin z DP O(n³) odpada wcześniej (14.21 przy n=100).
Przy n=500 pojedyncze wywołanie DP dynasearcha (~200 s) i motzkina
(~30 s) przekracza każdy budżet — 1 iteracja, wartości "dla kompletności"
(zakresy 13.30–16.05 i 13.00–18.40). Mimo to jedno wywołanie dynasearcha
bije adjacenta z tysiącami iteracji (14.53 vs 15.36) — struktura ruchu
znaczy więcej niż ich liczba, o ile ruch niesie dość swapów.
Okienkowane dyn/motz kończą na n=50 (26.37/26.03, ~3 pp za fibonaccim
23.56): bariera ścienna (minorminer na ~5 oknach; budowa O(n³)), nie
embedding — ta sama ściana, która zdjęła klasyczne dyn/motz przy n=500,
tu przychodzi dwie klasy rozmiaru wcześniej. Poza n=200 fibonacci
enhanced jest jedynym kwantowym sąsiedztwem bez obcięć.

### Tabela 6 — RPD per budżet, n=20, m=5, bez poolingu (panele ILS i SA)

ILS: klasyka 6.60/5.96/3.92/4.33 (100 ms) → 3.41/3.82/2.68/2.64 (10 s);
kwanty — patrz pełna tabela w main.tex; wiersze enhanced płaskie
100→1000 ms (każdy run = 1 iteracja; budżet poniżej ~2 s kosztu drugiego
wywołania QPU nic nie kupuje), spadek dopiero przy 5000 ms (2–3 iteracje).
Kreski = warstwy budżetowe jeszcze niezmierzone w protokole transz.

### Jakość dekompozycji okienkowej

Podejście okienkowe wprowadza ograniczone przybliżenie: pary przecinające
granice okien nigdy nie są ewaluowane. Strata zmierzona przy n=20
(największy rozmiar, gdzie pełny QUBO jeszcze się embeduje): bliska zera
dla dynasearcha, ~4 pp i narastająca z budżetem dla motzkina. Wszystkie
runy z ρ=0.5; większy overlap zmniejszałby artefakty granic kosztem
większej liczby submisji — czułość na ρ pozostaje niezbadana.

### Wkład BackTrackJumpList

Kontrolowana ablacja na sąsiedztwach klasycznych (3600 parowanych runów:
oba ramiona przy n=20, m∈{5,10,20}, tl∈{1,5,10} s, 10 inst × 5 seedów)
potwierdza konstrukcję: ILS z BTJL bije ILS z restartem losowym w KAŻDEJ
komórce maszyny×budżetu o 1.1–2.6 pp RPD (Wilcoxon p<10⁻¹⁷ per komórka).
Margines podąża za bogactwem sąsiedztwa: rzadkie zyskują 4.1 pp
(adjacent) i 3.5 pp (fibonacci), dynasearch/motzkin — z własnymi
ścieżkami ucieczki — wciąż istotne 0.4/0.3 pp (p<10⁻¹⁹).

### Analiza przepustowości

Enhanced wysyłają wiele wywołań QUBO na iterację (jedno per okno);
kompromis pokrycie-vs-latencja widać wprost w Tabeli 5 i licznikach
iteracji: wywołanie okienkowane mnoży 30 ms dostępu przez liczbę okien,
a koszt ścienny pozostaje zdominowany przez embedding i kolejkę.

## 7. Wnioski

Klasa quantum QUBO enhanced łamie barierę n=20 dwiema zmianami:
usunięciem filtra delty (sąsiedztwa liniowe) i dekompozycją okienkową
(kwadratowe). Akcelerator Smutnickiego tnie zmienne o 30–60% przed
submisją, obniżając długość łańcuchów i ryzyko zerwań. BTJL zastępuje
restarty perturbacjami elit — zmiana darmowa w czasie działania, bijąca
restart losowy w każdej komórce ablacji 3600 runów (1.1–2.6 pp, p<10⁻¹⁷).

Kampanie QPU z lipca 2026 wypełniły każdą traktowalną komórkę tabel
realnymi pomiarami Advantage_system4.1 i odpowiedziały na obie hipotezy:
(i) fibonacci enhanced trzyma poziom w całym zakresie — RPD spada
monotonicznie 31.7→15.0% (n=20→200), luka do klasycznego fibonacciego
zwęża się 20.4→4.6 pp przy n=500, gdzie wariant kwantowy dorównuje
klasycznemu adjacent (15.75 vs 15.36); (ii) koszt okienkowania okazał
się strukturalny, nie jednolity — okienkowany dynasearch replikuje lub
lekko bije pełny QUBO przy n=20, eliminując porażki embeddingu, podczas
gdy okienkowany motzkin płaci ~4 pp, rosnąco z budżetem, bo zagnieżdżenia
międzyokienne są dokładnie tym, w co pełna formulacja obraca dodatkowe
iteracje. Przy n=50, gdzie żaden pełny QUBO O(n²) się nie embeduje, oba
okienkowane warianty dostarczają działające ruchy QPU w granicach 3 pp
od fibonacciego enhanced. Reżim jednej iteracji, anatomia kosztu QPU
i stochastyczność embeddingu przy granicy gęstej pojemności raportowane
są obok liczb jakości, bo definiują, co uczciwe porównanie ścienne
kwantowego i klasycznego przeszukiwania może twierdzić.

**Future work:** adaptacyjny dobór okien z konwergencji per okno, obwody
QAOA na procesorach bramkowych, dekompozycje zachowujące interakcje
międzyokienne, rozszerzenie na hybrydowy flow shop.
