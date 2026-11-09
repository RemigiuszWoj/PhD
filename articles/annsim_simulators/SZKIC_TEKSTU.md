# ANNSIM: szkic tekstu do akceptacji (08.11.2026)

Etap 3 (n = 50…500) liczy się w tle; tam, gdzie potrzebne są jego wyniki, zostaje `\tofill{stage 3}`.
Zasady: wyniki słownie i krótko (ok. 1 strony), Discussion + Conclusions w jednej sekcji (max pół strony),
pełne zdania, krótkie podpisy. Przed finalizacją tekst przejdzie przez /humanizer.

---

## A. Nowy tekst (EN → main.tex)

### Abstract

Simulators carry most of the development of algorithms for noisy quantum processors, but whether they
predict what a device does is rarely tested on the task the simulation stands for. We feed nine freely
available circuit simulators with the calibration of IQM Sirius, a 16-qubit processor with a star-shaped
topology, and compare their predictions with the device on fixed-angle QAOA that selects moves in iterated
local search for the permutation flow shop problem. Three noise scenarios range from a noiseless circuit
through depolarizing gate and readout errors to added relaxation during gates. The engines that can run a
scenario agree with one another within sampling noise, so the choice among them is a matter of cost: only
trajectory simulators reach all 16 qubits with noise, and Qulacs does so fastest. On whole search runs, the
simulators with relaxation reproduce the schedules of the device within its own run-to-run spread. On
single circuits they do not: the simulated output distributions differ from the measured ones about ten
times more than the device differs from itself between two days, and at depth one the device even
outperforms the noiseless simulation, which points to relaxation of idle qubits that no scenario models.

### Introduction, last sentence (replaces `\tofill{One sentence with the main finding…}`)

The simulators agree with each other and with the device on the outcome of the search but not on what a
single circuit measures, so a simulator validated on the final schedule cannot be trusted to predict output
distributions.

### Results

Every simulator that supports a scenario reproduces the same distributions. The exact methods agree to
machine precision, and the sampled ones stay within the shot-noise floor in all but the few percent of
windows expected by chance. The exceptions are failures rather than disagreements: the noise simulators of
DDSIM return wrong distributions even without noise and crash on larger windows, the trajectory method of
Aer stops with an error once S3 adds relaxation, and the noise simulator of Qulacs never applies amplitude
damping. The choice of engine therefore does not change the prediction, only its cost.

Figure~\ref{fig:cost} and Table~\ref{tab:t6} show that cost. The density-matrix simulators stop at nine to
thirteen qubits within 300\,s per circuit, while the trajectory methods of qsim and Qulacs run all 16 qubits
of Sirius with noise. Qulacs is the fastest program at every size. Its explicit loop over shots needs only
the memory of a pure state, whereas its built-in noise simulator is slightly faster but needs gigabytes and
gets S3 wrong. The trajectories of Qibo are correct but one to two orders of magnitude slower.

The compiler matters more than the engine. The IQM adapter for Cirq compiles the same circuits with the same
number of CZ gates as the adapter for Qiskit but with 30 to 60\,\% more MOVE gates (Table~\ref{tab:compilers}),
and at $p = 2$ the extra noise lowers how often the lowest-energy sample is the optimal move.

On whole ILS runs with the move budget of the device, the scenario with relaxation cannot be told apart from
Sirius for either simulator (Figure~\ref{fig:prd}, Table~\ref{tab:t5}). The mean deviation from the lower
bound differs by less than half a percentage point, well inside the spread between two runs of the device.
Without relaxation, qsim gives significantly worse schedules than the device, and Qulacs points the same way
without reaching significance. \tofill{stage 3: one or two sentences on $n = 50$--$500$.}

Single circuits tell a different story. The device reproduces itself closely, since the mean energy of a
circuit changes by less than 0.01 between the two days, but every simulator in every scenario misses it by
about ten times as much (Figure~\ref{fig:h3}, Table~\ref{tab:h3}). At $p = 1$ the device does better than the
noiseless simulation: it samples the optimal move in 79 of 84 histograms, against 46 in S1, and its samples
contain fewer ones. The optimal move sets few variables to one, so a drift of the qubits towards their ground
state favours it, which is what relaxation of idle qubits would cause. At $p = 2$ the scenarios S2 and S3
match the mean energy over all circuits but flatten the differences between circuits that the device shows.

### Discussion and Conclusions (one section, replaces both `\tofill`s)

For a study like this one, any trajectory simulator predicts the outcome of the optimization as well as a
second run on the device would, provided that its noise model includes relaxation, and Qulacs does so at the
lowest cost. The agreement owes much to the readout of the search, which keeps the best of 4096 samples and
is insensitive to the shape of the distribution. The same simulators fail once the question concerns the
distribution itself, such as the probability of the optimal move or the effect of depth, and there the
device behaves in ways no scenario predicts. The most likely missing ingredient is noise on idle qubits: on
a star processor every two-qubit gate passes through the resonator, so most qubits wait during most of the
circuit.

These conclusions have limits. The noise models use one calibration snapshot and treat the resonator as a
qubit, the circuit-level data cover one instance class at two depths, and the run-level comparison beyond
$n = 20$ rests on \tofill{stage 3}. The next step is a scenario with idle noise built from the timing of the
compiled circuits and tested against the same histograms.

---

## B. Poprawki istniejącego tekstu (co na co, EN; PL analogicznie)

1. **Methodology › Noise scenarios**, ostatnie zdanie
   - było: „DDSIM receives S2 and S3 with averaged rates.”
   - jest: „The noise simulators of DDSIM fail already on noiseless circuits, so DDSIM runs S1 only.”
2. **Methodology › Two tracks**
   - było: „exported as OpenQASM with MOVE written as a swap between a qubit and the resonator”
   - jest: „exported as a list of native gates with MOVE written as a swap between a qubit and the resonator”
   - było: „passes through three compilers (the IQM adapters for Qiskit and Cirq and TKET)”
   - jest: „passes through two compilers (the IQM adapters for Qiskit and Cirq)” (TKET nie był liczony)
3. **Methodology › Hardware references**
   - było: „and at $p = 2, 3$ \tofill{H2}. … a small run on two different days \tofill{H3} gives them …”
   - jest: „… Circuit-level comparisons need histograms of single windows; H3 runs the 84 circuits of the
     first move of tai20 instance 0 at $p = 1, 2$ on two days (6 and 7 November 2026), 4096 shots each, and
     the simulators replay the compiled circuits with the calibration of the same day.”
4. **Methodology › Metrics** (cały akapit; dotychczasowy opisywał TVD, Hellingera, test znaków i τ Kendalla,
   których nie liczyliśmy)
   - jest: „For single circuits we compare the mean QUBO energy of the samples, normalised so that 0 is the
     optimum and 1 the worst assignment, and whether the optimal move is sampled at all. The total variation
     distance is reported as well, but with 4096 shots over up to $2^{15}$ outcomes it mostly measures
     sampling noise. The hardware floor is the difference between the two days of H3. For whole runs we
     compare the relative deviation of the makespan from the Taillard lower bound after as many moves as the
     device made, with the Wilcoxon signed-rank test, Holm's correction and a bootstrap interval over
     instances; repeated device runs give the floor. Cost is wall time per circuit and peak memory.”
5. **Experimental Setup**, zdania o etapie 3
   - było: „Third, the two or three simulators that best predict the device at acceptable cost run the full
     Taillard set … We fix the selection criteria before the second stage.”
   - jest: „Third, qsim and Qulacs, the two simulators that run all 16 qubits with noise at the lowest cost,
     replay the 153 device runs with $n = 50$ to $500$ in scenario S2.”
6. **Tabela tab:hw**: H2 „planned” → „done (Nov 2026), not used here”; H3 „optional” → „done (6--7 Nov 2026)”.
7. **Tabela tab:sims**: usunąć dwie kolumny pomiarowe z `\tofill{?}` (dublowały T6) i w podpisie zostawić
   „Noisy method from the documentation; measured limits are in Table~\ref{tab:t6}.”
8. **Tabela tab:compilers** (mediana na okno, $p = 1$, 161 okien): kolumna TKET usunięta;
   Adjacent 72 / 72 vs 72 / 92; Fibonacci 16 / 16 vs 16 / 20; Dynasearch 164 / 178 vs 164 / 286;
   Motzkin 134 / 144 vs 134 / 214 (CZ / MOVE, qiskit-iqm vs IQM Cirq).
9. **Rysunek fig:heat** (zaślepka mapy cieplnej) → rysunek H3 `figures/h3_energy.pdf`, etykieta `fig:h3`,
   podpis: „Each point is one H3 circuit on one day. The grey band shows the mean difference between the two
   days on Sirius.”; pod nim tabela `\input{measurements/h3_table}`.

---

## C. Wersja polska nowego tekstu (→ main_pl.tex; ta wersja jest w nagraniu)

### Streszczenie

Symulatory niosą większość pracy nad algorytmami dla zaszumionych procesorów kwantowych, ale rzadko
sprawdza się, czy przewidują zachowanie urządzenia w zadaniu, które symulacja zastępuje. Dziewięć darmowych
symulatorów obwodów dostaje kalibrację IQM Sirius, 16-kubitowego procesora o topologii gwiazdy, i porównujemy
ich przewidywania z urządzeniem na QAOA o stałych kątach, które wybiera ruchy w iteracyjnym przeszukiwaniu
lokalnym dla permutacyjnego problemu przepływowego. Trzy scenariusze szumu prowadzą od obwodu bez szumu przez
błędy depolaryzujące bramek i odczytu do dodanej relaksacji w czasie bramek. Silniki, które obsługują dany
scenariusz, zgadzają się ze sobą w granicach szumu próbkowania, więc wybór między nimi jest kwestią kosztu:
tylko symulatory trajektoriowe obejmują z szumem wszystkie 16 kubitów, a najszybciej robi to Qulacs. Na
całych przebiegach przeszukiwania symulatory z relaksacją odtwarzają uszeregowania urządzenia w granicach jego
własnego rozrzutu między przebiegami. Na pojedynczych obwodach już nie: rozkłady z symulacji różnią się od
zmierzonych mniej więcej dziesięć razy bardziej niż urządzenie samo od siebie między dwoma dniami, a przy
głębokości jeden urządzenie wypada nawet lepiej niż symulacja bez szumu, co wskazuje na relaksację kubitów w
bezczynności, której nie modeluje żaden scenariusz.

### Wstęp, ostatnie zdanie

Symulatory zgadzają się ze sobą i z urządzeniem co do wyniku przeszukiwania, ale nie co do tego, co mierzy
pojedynczy obwód, więc symulatorowi sprawdzonemu na końcowym uszeregowaniu nie można ufać w przewidywaniu
rozkładów wyników.

### Wyniki

Każdy symulator, który obsługuje dany scenariusz, odtwarza te same rozkłady. Metody dokładne zgadzają się do
precyzji maszynowej, a próbkujące mieszczą się w progu szumu próbkowania we wszystkich oknach poza kilkoma
procentami, których należy się spodziewać przez przypadek. Wyjątki to awarie, a nie rozbieżności: symulatory
szumu DDSIM zwracają złe rozkłady nawet bez szumu i wywracają się na większych oknach, metoda trajektorii Aer
kończy się błędem, gdy S3 dodaje relaksację, a symulator szumu Qulacs nigdy nie stosuje tłumienia amplitudy.
Wybór silnika nie zmienia więc przewidywania, tylko jego koszt.

Ten koszt pokazują rysunek~\ref{fig:cost} i tabela~\ref{tab:t6}. Symulatory macierzy gęstości kończą się na
dziewięciu do trzynastu kubitach przy limicie 300\,s na obwód, a metody trajektorii qsim i Qulacs obejmują z
szumem wszystkie 16 kubitów Siriusa. Qulacs jest najszybszym programem przy każdym rozmiarze. Jego jawna
pętla po shotach potrzebuje tylko pamięci stanu czystego, a wbudowany symulator szumu jest nieco szybszy, ale
zajmuje gigabajty i liczy S3 źle. Trajektorie Qibo są poprawne, ale o jeden do dwóch rzędów wielkości
wolniejsze.

Kompilator ma większe znaczenie niż silnik. Adapter IQM dla Cirq kompiluje te same obwody z tą samą liczbą
bramek CZ co adapter dla Qiskit, ale z 30--60\,\% większą liczbą bramek MOVE (tabela~\ref{tab:compilers}), a
przy $p = 2$ dodatkowy szum sprawia, że próbka o najniższej energii rzadziej jest ruchem optymalnym.

Na całych przebiegach ILS z budżetem ruchów urządzenia scenariusza z relaksacją nie da się odróżnić od
Siriusa dla żadnego z dwóch symulatorów (rysunek~\ref{fig:prd}, tabela~\ref{tab:t5}). Średnie odchylenie od
dolnego ograniczenia różni się o mniej niż pół punktu procentowego, czyli dużo mniej niż dwa przebiegi
samego urządzenia. Bez relaksacji qsim daje istotnie gorsze uszeregowania niż urządzenie, a Qulacs wskazuje
ten sam kierunek, choć bez istotności. \tofill{etap 3: jedno lub dwa zdania o $n = 50$--$500$.}

Pojedyncze obwody pokazują co innego. Urządzenie dobrze odtwarza samo siebie, bo średnia energia obwodu
zmienia się między dniami o mniej niż 0{,}01, ale każdy symulator w każdym scenariuszu chybia o mniej więcej
dziesięć razy więcej (rysunek~\ref{fig:h3}, tabela~\ref{tab:h3}). Przy $p = 1$ urządzenie wypada lepiej niż
symulacja bez szumu: ruch optymalny pojawia się w 79 z 84 histogramów, wobec 46 w S1, a jego próbki zawierają
mniej jedynek. Ruch optymalny ustawia na jeden niewiele zmiennych, więc dryf kubitów w stronę stanu
podstawowego mu sprzyja, a taki skutek miałaby relaksacja kubitów w bezczynności. Przy $p = 2$ scenariusze S2
i S3 trafiają średnią energię wszystkich obwodów, ale spłaszczają różnice między obwodami, które widać na
urządzeniu.

### Dyskusja i wnioski

W badaniu takim jak to każdy symulator trajektoriowy przewiduje wynik optymalizacji tak dobrze jak drugi
przebieg na urządzeniu, pod warunkiem że jego model szumu zawiera relaksację, a Qulacs robi to najtaniej.
Zgodność w dużej mierze zawdzięczamy odczytowi w przeszukiwaniu, który zachowuje najlepszą z 4096 próbek i
nie zależy od kształtu rozkładu. Te same symulatory zawodzą, gdy pytanie dotyczy samego rozkładu, na przykład
prawdopodobieństwa ruchu optymalnego albo wpływu głębokości, i tam urządzenie zachowuje się w sposób, którego
nie przewiduje żaden scenariusz. Najbardziej prawdopodobnym brakującym składnikiem jest szum kubitów w
bezczynności: w procesorze gwiazdowym każda bramka dwukubitowa przechodzi przez rezonator, więc większość
kubitów czeka przez większość obwodu.

Te wnioski mają granice. Modele szumu korzystają z jednej migawki kalibracji i traktują rezonator jak kubit,
dane na poziomie obwodów obejmują jedną klasę instancji przy dwóch głębokościach, a porównanie przebiegów
powyżej $n = 20$ opiera się na \tofill{etap 3}. Kolejnym krokiem jest scenariusz z szumem bezczynności
zbudowany z harmonogramu skompilowanych obwodów i sprawdzony na tych samych histogramach.
