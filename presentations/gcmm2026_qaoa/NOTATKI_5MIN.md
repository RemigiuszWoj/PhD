# Notatki do plakatu GCMM 2026 – około 5 minut mówienia

Plakat czyta się kolumnami z góry na dół:

| Lewa – metoda | Środkowa – kompilacja i IBM | Prawa – porównanie i wnioski |
|---|---|---|
| One QUBO, two quantum back-ends | Circuit size of one move | Gate model, annealer and classical |
| Four neighborhoods, four conflict graphs | *Circuit size: same order on every chip* (ramka) | Same circuits on three chips |
| One circuit per move, angles frozen (mapa kątów) | IBM Heron r2: one move, ten instances | *Short circuits agree, long ones do not* (ramka) |
| Three chip layouts, same four QUBOs | Why only one move: the queue | Conclusions · QR do Zenodo |

Dwie rzeczy do zapamiętania przez widza są w różowych ramkach. Poniżej najpierw tekst do powiedzenia
(po angielsku, około 780 słów, czyli 5–6 minut spokojnym tempem; gdy czasu brakuje, pomiń
kafelek 9 albo 10–11), a potem po polsku wyjaśnienia trudniejszych
miejsc i odpowiedzi na pytania, które mogą paść.

Krótsza wersja (30 s), gdy ktoś tylko przechodzi:
> "We take the same QUBO that used to go to a quantum annealer and run it on gate-model chips with QAOA,
> with the angles fixed in advance. The size of the compiled circuit orders the neighborhoods the same way on
> every chip. On IBM one move improves the schedule from 27.9 to about 22–25 percent above the lower bound, and
> short circuits give the same answer on a 156-qubit and a 16-qubit chip. The bottleneck is the queue, not
> the computation."

---

## Część 1. Tekst do powiedzenia (EN), kafelek po kafelku

### 1. One QUBO, two quantum back-ends (lewa, góra) – ok. 40 s
*Pokaż schemat ze strzałkami.*

> "The problem is the permutation flow shop: n jobs go through m machines in the same order, and we want the
> order with the shortest makespan. Local search improves the order step by step; one step, a *move*, is a
> set of job swaps applied together. Choosing the best set of swaps is itself a small optimisation problem,
> and we write it as a QUBO, a quadratic problem over 0/1 variables. In earlier work this QUBO went to a
> D-Wave annealer. Here the very same QUBO goes to gate-model chips through QAOA. So the metaheuristic, the
> instances and the matrix are identical; only the solver changes."

### 2. Four neighborhoods, four conflict graphs – ok. 30 s
*Pokaż łańcuch Fibonacciego i gęsty Dynasearch.*

> "We use four neighborhoods. Each candidate swap becomes one qubit, and each pair of swaps that cannot be
> applied together becomes one two-qubit gate. Fibonacci gives a simple chain. In Adjacent every pair
> clashes, and Dynasearch and Motzkin are dense too, but Adjacent has only five candidates in a window
> against their fifteen. Density and size decide everything that follows."

### 3. One circuit per move, angles frozen – ok. 40 s
*Pokaż gwiazdkę na mapie.*

> "A QAOA circuit has two angles. Normally you tune them on the device for every problem, which costs
> hundreds of runs. We cannot afford that, because cloud access gives us minutes per month. So we tune the
> angles once, on a classical simulator, over a set of training QUBOs. This map shows the expected cost of
> the quantum state for every pair of angles: blue is better than random guessing, red is worse, and the
> star is the pair we freeze. After that, every move on hardware is exactly one submission. We sample the
> circuit 4096 times and take the sample with the lowest QUBO value as the move."

### 4–6. Chips (lewa, dół), circuit size and the pink box (środek, góra) – ok. 60 s
*Pokaż trzy mapy sprzężeń na dole lewej kolumny, potem słupki, potem różową ramkę.*

> "Real chips cannot connect every pair of qubits. IBM uses a sparse heavy-hex lattice, IQM uses square
> lattices and a star with a central resonator. Whenever the circuit needs a link the chip does not have,
> the compiler adds extra gates. We compiled the same four QUBOs onto four chips with three layouts – the two
> IQM Crystal chips, with 20 and 54 qubits, share the square lattice. The absolute numbers move
> by about a factor of two – heavy-hex is the least favourable – but the order never changes: Fibonacci is
> tiny, with fourteen CZ gates on every chip, Adjacent is next, and Motzkin and Dynasearch need from about
> six hundred to well over a thousand two-qubit gates, depending on the chip. A deeper circuit collects more noise. And this is the same order in which the neighborhoods fit
> on the annealer, for a completely different reason."

### 7–8. IBM Heron results and the queue (środek, dół) – ok. 60 s
*Pokaż duży napis 27.9 → 22–25, potem kropki z czasem.*

> "Then we ran all four on the IBM Heron r2 processor, ten Taillard instances with twenty jobs, both iterated
> local search and simulated annealing – eighty runs, about eight minutes of QPU time. Starting from the
> identity order, one QAOA move lowers the mean deviation from the lower bound from 27.9 to between 22 and 25
> percent; Dynasearch makes the largest single step. But each run made only one move. The computation itself
> takes 3 to 7 seconds of QPU time, while a move took between 10 seconds and almost 5 minutes on the clock,
> because of the queue. So with a 10-second budget no run ever reached a second move."

### 9. Gate model, annealer and classical search (prawa, góra) – ok. 30 s
*Pokaż szare słupki przy zerze.*

> "Here is the honest comparison. The gate-model bars are after one move, the others after 10 seconds. On
> Dynasearch the single gate move is the lowest quantum bar, on Motzkin it falls between the two annealer
> variants, and on Adjacent and Fibonacci it is higher. But all
> quantum versions are far from classical local search, which reaches about 3 percent at this size. This
> paper is about whether the pipeline works and what it costs, not about speed."

### 10–11. Same circuits on three chips – ok. 40 s
*Pokaż płaskie linie Adjacent i Fibonacci, potem skaczący Motzkin.*

> "Finally we ran the same circuits on three different chips: the 156-qubit IBM Heron, a 20-qubit IQM
> square lattice and a 16-qubit IQM star. For the short circuits the chip simply does not matter: Adjacent
> gives the same makespan on every instance that fits, and Fibonacci on six of nine. For the deep circuits
> the result changes from chip to chip – Motzkin goes from 17 to 13 percent. That is only one instance, so
> we cannot yet separate the chip from sampling noise, but it suggests a threshold rather than a gradual
> effect."

### 12. Conclusions – ok. 20 s

> "So: one QUBO drives both quantum paradigms; compiled circuit size gives a hardware-independent ranking of
> neighborhoods; and a move costs one submission and about thirty seconds, with a price that grows linearly
> with instance size. Next we want the full protocol on every chip, once we have a larger allocation. The code
> and all runs are behind the QR code."

---

## Część 2. Trudniejsze miejsca – wyjaśnienia po polsku

### Mapa kątów (kafelek „One circuit per move, angles frozen”)
- **Co jest na osiach.** QAOA przy głębokości p = 1 ma dwa pokrętła. γ (oś pozioma, „cost angle”) mówi, jak
  mocno obwód „wsłuchuje się” w funkcję celu QUBO. β (oś pionowa, „mixer angle”) mówi, jak mocno potem miesza
  stany, żeby nie utknąć.
- **Co oznacza kolor.** Dla każdej pary (γ, β) symulator liczy średnią oczekiwaną wartość QUBO stanu
  kwantowego, uśrednioną po treningowych QUBO i znormalizowaną. **0 = losowe zgadywanie**, bo przy γ = 0 obwód
  daje równą superpozycję wszystkich odpowiedzi. **Niebieskie** pola są lepsze od losowania, więc próbki częściej
  trafiają w dobre ruchy. **Czerwone** są gorsze od losowania, bo obwód aktywnie promuje złe ruchy.
- **Gwiazdka.** To minimum na siatce 25 × 13 punktów. Ten sam punkt jest w artykule: dla Fibonacciego
  γ = 21π/24 i β = 5π/24. Skrypt plakatu sprawdza asercją, że gwiazdka leży dokładnie w minimum siatki.
- **Na czym liczone.** Zbiór treningowy to 3 instancje Taillarda (tai20_5 nr 0 i 1, tai20_10 nr 0) przy
  permutacji tożsamościowej, cięte na okna po 6 pozycji z krokiem 3. Daje to 21 QUBO dla Adjacent i
  Fibonacciego oraz 11 dla Dynasearch i Motzkina.
- **Po co to wszystko.** Kąty są zamrożone, więc na sprzęcie nie ma pętli strojenia. Jeden ruch to jedno
  zadanie na QPU, jak jedno wywołanie annealera. Bez tego jeden ruch kosztowałby dziesiątki do setek
  wykonań obwodu, a przy ~10 minutach QPU miesięcznie cały protokół by się nie zmieścił.
- **Obwód** (bramki R_Z, R_ZZ, R_X) usunąłem z plakatu, bo z metra był nieczytelny. Jeśli ktoś spyta, jak
  wygląda: warstwa Hadamardów, potem po jednej bramce R_ZZ na każdą krawędź grafu konfliktów, potem R_X na
  każdym kubicie (rysunek jest w artykule, Fig. 1).
- **Pozostałe sąsiedztwa** mają podobny obraz: czerwona plama przy małym γ, niebieski obszar przy γ ≈ 0,7–0,9.
  Mają tylko własne gwiazdki. Dlatego na plakacie jest jeden panel, a cztery są w `img/p_landscape.png`.

### Mapy sprzężeń i „circuit size”
- Bramka dwukubitowa działa tylko między kubitami połączonymi na chipie. Brakujące połączenie kompilator
  obchodzi łańcuchem bramek SWAP, więc obwód rośnie i zbiera więcej szumu.
- Na gwieździe IQM wszystko łączy się przez centralny rezonator: zamiast SWAP-ów jest operacja MOVE. Bramek
  CZ jest mniej, ale MOVE-y idą po kolei przez jeden rezonator, więc głębokość ląduje między dwiema kratami.
  Hasło z artykułu brzmi: „connectivity buys gate count, not depth”.
- Na wykresie Star 16 słupki „two-qubit gates” to CZ + MOVE (tak jest w tabeli w artykule). W tekście artykułu
  liczby 90/576 to same CZ. Obie wersje są spójne.
- Fibonacci jest kontrolą. Jego graf to ścieżka, więc na każdej mapie daje 14 CZ bez routingu, i nic w tym
  wierszu nie zależy od sprzętu.

### Wykres porównania – czego nie mówić
- Nie mówić, że bramkowy jest „lepszy od annealera”. Na Dynasearch jeden ruch bramkowy daje 22,1 wobec 25,9 przy
  10 s annealera. Na Motzkinie jest lepszy od annealera w oknach (23,1 wobec 25,2), ale gorszy od pełnego
  annealera. Na Adjacent i Fibonaccim jest gorszy o 5–5,5 punktu. Budżety nie są te same.
- Przy budżetach ≤ 1 s, gdzie annealer też robi jeden ruch, wiersze annealera i bramkowe dla Adjacent i
  Fibonacciego różnią się o kilka dziesiątych punktu. Część luki to więc czas dostępu, a nie jakość.
- Nie używać słów „quantum advantage” ani „first”.

### Dlaczego ILS i SA dają prawie to samo
Oba algorytmy startują z tej samej permutacji i robią tylko jeden ruch, więc wykonują to samo obliczenie.
Różnice (< 0,5 punktu średnio, do 5,8 na pojedynczej instancji) biorą się z losowości próbkowania na chipie.

---

## Część 3. Pytania, które mogą paść

| Pytanie | Odpowiedź (sedno) |
|---|---|
| Why not tune the angles on the device? | It costs tens to hundreds of circuit runs per move; the IBM Open plan gives about 10 minutes of QPU time per 28 days. Fixed angles make one move = one submission. |
| Why only p = 1? | Deeper circuits are too noisy on current chips and need more angles to fix in advance. Larger p is outside this paper. |
| Why the lowest-energy sample and not the most frequent one? | At p = 1 the most frequent bitstring is usually all zeros (no swap) or a bad state; the lowest-energy sample is the useful readout. |
| Why only n = 20? | Quota. Dense moves are cut into windows of 6 positions; at n = 20 a move needs 6 circuits, at n = 500 it would need 166, and billing is per circuit. One of the ten instances did not even fit the 16 qubits of the IQM star. |
| How expensive is the full protocol? | About 550 credits per processor for 10 instances × 5 seeds × 6 budgets × 2 metaheuristics, against 30 credits a month. |
| Is it faster than classical? | No. Classical local search reaches about 3 % at this size; all quantum versions stay above 17 %. The contribution is the pipeline, the depth analysis and the measured cost. |
| What is the "threshold"? | Below some circuit size the chip does not change the answer (Adjacent identical on 9/9 instances); above it the chip matters (Motzkin 17 → 13 %). Dense rows: one instance only. |
| Does the queue count against the algorithm? | Under a strict wall-clock rule yes, and that is an open methodological question: identical jobs wait times that differ by an order of magnitude. |
| Are the training instances also test instances? | Two of the three are. Every run starts from the identity, so their first moves meet the training windows; at p = 1 the best angles are nearly the same across instances, so we do not expect a bias (stated in the paper). |
| Why 30 s per move, and is that the chip? | It is mostly the queue on a shared processor. The same move took 12 s on a less heavily shared processor. |
| Where do the Crystal 20 numbers come from? | The raw output of the four Crystal 20 runs was lost; the values come from the lab notes, and the paper says so. |
| Error mitigation? | None: no mitigation, no dynamical decoupling, no twirling; optimisation level 1. |
| Where is the code? | QR code → Zenodo 10.5281/zenodo.24106982 (code, angles, all hardware runs). |
