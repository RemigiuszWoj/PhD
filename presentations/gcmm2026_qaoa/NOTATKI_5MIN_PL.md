# Notatki do plakatu GCMM 2026 – wersja polska (ok. 5 minut)

Ten sam tekst co w `NOTATKI_5MIN.md`, część 1, tylko po polsku, do mówienia naturalnym językiem.
Wyjaśnienia trudniejszych miejsc i pytania z odpowiedziami są w tamtym pliku (części 2 i 3).

Plakat czyta się kolumnami z góry na dół: **lewa to metoda**, **środkowa to kompilacja i wyniki na IBM**,
**prawa to porównanie i wnioski**. Najważniejsze dwie rzeczy są w różowych ramkach.

## Wersja 30-sekundowa

> „Ten sam QUBO, który wcześniej rozwiązywał annealer kwantowy, puszczamy na komputery bramkowe przez QAOA,
> z kątami ustalonymi z góry. Wielkość skompilowanego obwodu ustawia sąsiedztwa w tej samej kolejności na
> każdym chipie. Na IBM jeden ruch poprawia rozwiązanie z 27,9 do około 22–25 procent powyżej dolnego
> ograniczenia, a krótkie obwody dają ten sam wynik na chipie 156- i 16-kubitowym. Wąskim gardłem jest
> kolejka, a nie obliczenia.”

---

## Tekst na 5 minut, kafelek po kafelku

### 1. One QUBO, two quantum back-ends (lewa, góra) – ok. 40 s
*Pokaż schemat ze strzałkami.*

> „Problemem jest permutacyjny problem przepływowy: n zadań przechodzi przez m maszyn w tej samej kolejności,
> a szukamy kolejności o najkrótszym czasie zakończenia, czyli makespanie. Przeszukiwanie lokalne poprawia
> kolejność krok po kroku. Jeden krok, czyli ruch, to zestaw zamian zadań wykonanych naraz. Wybór najlepszego
> zestawu zamian to sam w sobie mały problem optymalizacyjny i zapisujemy go jako QUBO, czyli problem
> kwadratowy na zmiennych 0/1. We wcześniejszych pracach ten QUBO trafiał na annealer D-Wave. Tutaj dokładnie
> ten sam QUBO trafia na komputery bramkowe przez QAOA. Metaheurystyka, instancje i macierz są te same,
> zmienia się tylko solver.”

### 2. Four neighborhoods, four conflict graphs – ok. 30 s
*Pokaż łańcuch Fibonacciego i gęsty Dynasearch.*

> „Używamy czterech sąsiedztw. Każda kandydująca zamiana to jeden kubit, a każda para zamian, których nie
> można wykonać razem, to jedna bramka dwukubitowa. Fibonacci daje prosty łańcuch, a pozostałe trzy są gęste:
> Adjacent jest wręcz pełny, Dynasearch i Motzkin są blisko pełnego. Adjacent i tak zostaje mały, bo ma tylko
> kilka kandydatów w oknie, a Dynasearch i Motzkin mają ich dużo więcej. Gęstość razem z liczbą kandydatów
> decyduje o wszystkim, co dalej.”

### 3. One circuit per move, angles frozen – ok. 40 s
*Pokaż gwiazdkę na mapie.*

> „Obwód QAOA ma dwa kąty. Zwykle stroi się je na urządzeniu osobno dla każdego problemu, co kosztuje setki
> uruchomień. Nas na to nie stać, bo dostęp w chmurze to kilka minut miesięcznie. Dlatego stroimy kąty raz, na
> klasycznym symulatorze, na zbiorze treningowych QUBO. Ta mapa pokazuje oczekiwany koszt stanu kwantowego
> dla każdej pary kątów: niebieskie jest lepsze od losowego zgadywania, czerwone gorsze, a gwiazdka to para,
> którą zamrażamy. Potem każdy ruch na sprzęcie to dokładnie jedno zadanie. Obwód próbkujemy 4096 razy i jako
> ruch bierzemy próbkę o najniższej wartości QUBO.”

### 4–6. Chips (lewa, dół), circuit size i różowa ramka (środek, góra) – ok. 60 s
*Pokaż trzy mapy połączeń na dole lewej kolumny, potem słupki, potem różową ramkę.*

> „Prawdziwe chipy nie łączą każdej pary kubitów. IBM ma rzadką kratę heavy-hex, IQM ma kraty kwadratowe
> i gwiazdę z centralnym rezonatorem. Gdy obwód potrzebuje połączenia, którego chip nie ma, kompilator dokłada
> dodatkowe bramki. Skompilowaliśmy te same cztery QUBO na cztery chipy o trzech układach połączeń, bo oba
> chipy IQM Crystal, 20- i 54-kubitowy, mają tę samą kwadratową kratę. Wartości bezwzględne zmieniają się
> mniej więcej dwukrotnie, a heavy-hex wypada najgorzej, ale kolejność się nie zmienia. Fibonacci jest
> malutki, ma czternaście bramek CZ na każdym chipie, potem jest Adjacent, a Motzkin i Dynasearch potrzebują
> grubo ponad tysiąca. Im głębszy obwód, tym więcej szumu. To jest ta sama kolejność, w jakiej sąsiedztwa
> mieszczą się na annealerze, tylko z zupełnie innego powodu.”

### 7–8. Wyniki na IBM Heron i kolejka (środek, dół) – ok. 60 s
*Pokaż duży napis 27,9 → 22–25, potem kropki z czasem.*

> „Potem uruchomiliśmy wszystkie cztery sąsiedztwa na procesorze IBM Heron r2: dziesięć instancji Taillarda
> z dwudziestoma zadaniami, iterowane przeszukiwanie lokalne i symulowane wyżarzanie, razem osiemdziesiąt
> biegów i około ośmiu minut czasu QPU. Startując z kolejności identycznościowej, jeden ruch QAOA obniża
> średnie odchylenie od dolnego ograniczenia z 27,9 do 22–25 procent; największy pojedynczy krok robi
> Dynasearch. Ale każdy bieg zrobił tylko jeden ruch. Samo obliczenie to 3 do 7 sekund QPU, a ruch trwał na
> zegarze od 10 sekund do prawie 5 minut, przez kolejkę. Przy budżecie 10 sekund żaden bieg nie doszedł do
> drugiego ruchu.”

### 9. Gate model, annealer and classical search (prawa, góra) – ok. 30 s
*Pokaż szare słupki przy zerze.*

> „Tu jest uczciwe porównanie. Słupki bramkowe są po jednym ruchu, pozostałe po 10 sekundach. Na Dynasearch
> jeden ruch bramkowy wypada niżej niż annealer, na pozostałych trzech sąsiedztwach wyżej. Ale wszystkie
> wersje kwantowe są daleko za klasycznym przeszukiwaniem lokalnym, które przy tej wielkości dochodzi do
> około 3 procent. Ta praca jest o tym, czy cały potok działa i ile kosztuje, a nie o szybkości.”

### 10–11. Same circuits on three chips – ok. 40 s
*Pokaż płaskie linie Adjacent i Fibonacciego, potem skaczący Motzkin.*

> „Na koniec uruchomiliśmy te same obwody na trzech różnych chipach: 156-kubitowym IBM Heron,
> 20-kubitowej kwadratowej kracie IQM i 16-kubitowej gwieździe IQM. Przy krótkich obwodach chip po prostu
> nie ma znaczenia: Adjacent daje ten sam makespan na każdej instancji, która się mieści, a Fibonacci na sześciu
> z dziewięciu. Przy głębokich obwodach wynik zmienia się z chipu na chip, Motzkin przechodzi z 17 na 13
> procent. To tylko jedna instancja, więc nie da się jeszcze oddzielić wpływu chipu od szumu próbkowania, ale
> wygląda to na próg, a nie na stopniową zmianę.”

### 12. Wnioski – ok. 20 s

> „Podsumowując: jeden QUBO obsługuje oba kwantowe paradygmaty; wielkość skompilowanego obwodu daje
> niezależny od sprzętu ranking sąsiedztw; a ruch kosztuje jedno zadanie i około trzydziestu sekund, przy czym
> koszt rośnie liniowo z wielkością instancji. Dalej chcemy przeprowadzić pełny protokół na każdym chipie, gdy
> dostaniemy większy przydział czasu. Kod i wszystkie biegi są pod kodem QR.”

---

## Słowniczek do mówienia po polsku

| Na plakacie (EN) | Po polsku |
|---|---|
| permutation flow shop (PFSP) | permutacyjny problem przepływowy |
| makespan, C_max | czas zakończenia wszystkich zadań, makespan |
| move / neighborhood | ruch / sąsiedztwo |
| gate-model chip | komputer (procesor) bramkowy |
| quantum annealer | annealer kwantowy, wyżarzacz kwantowy |
| coupling map | mapa połączeń (sprzężeń) kubitów |
| compiled depth | głębokość obwodu po kompilacji |
| two-qubit gate (CZ) | bramka dwukubitowa (CZ) |
| frozen angles | zamrożone (ustalone z góry) kąty |
| shots | próbki, strzały |
| lower bound (LB) | dolne ograniczenie |
| PRD | procentowe odchylenie od dolnego ograniczenia |
| iterated local search / simulated annealing | iterowane przeszukiwanie lokalne / symulowane wyżarzanie |
| queue | kolejka zadań na komputerze kwantowym |
