# Osobny artykuł (na przyszłość): symulatory wyżarzania kwantowego a D-Wave

Pomysł z 25.10.2026 — wydzielony z planu ANNSIM decyzją użytkownika („tutaj tylko 1 typ symulatorów, ten 2
w innym artykule”; ANNSIM = symulatory bramkowe). Termin i konferencja: do ustalenia (kandydaci: ICCCI 2027,
konferencja z czasopismem, wiosna 2027).

## Pytanie
Na ile darmowe symulatory wyżarzania (klasyczne SA, kwantowe SQA, dokładna dynamika) przewidują wyniki
prawdziwego wyżarzacza D-Wave w kwantowym przeszukiwaniu sąsiedztwa dla PFSP — przy tych samych QUBO (okna
windowed_qubo) — i ile kosztują.

## Atuty
- **Zero nowych biegów sprzętowych:** wyniki D-Wave już są — `results/experiments/` (31 katalogów kampanii P1–P4,
  windowed_qubo, pełny Taillard n = 20…500), `results/qpu_timing.jsonl` (24 317 rekordów czasów QPU).
- Obliczenia lekkie (SA/SQA na CPU: sekundy–minuty na okno) → pełny Taillard wykonalny.
- Spina linię annealerową doktoratu (windowed_qubo, BPASTS) z walidacją symulatorów.

## Scenariusze (analogia do S1–S3 z ANNSIM)
| | Definicja | Programy |
|---|---|---|
| A1 | stan podstawowy QUBO | ExactSolver / brute force |
| A2 | dynamika wyżarzania bez błędów sterowania (SA, SQA), harmonogram zbliżony do D-Wave | dwave-samplers SA/Tabu, OpenJij SA/SQA; QuTiP na małych oknach |
| A3 | A2 + błędy sterowania ICE (gaussowskie zaburzenia h, J) + zanurzenie z łańcuchami (zrywanie) | SA, SQA |

Ewentualnie punkty odniesienia „inspirowane kwantowo”: symulowana bifurkacja (Toshiba SQBM+, implementacje
otwarte), Fujitsu Digital Annealer, Fixstars Amplify — część komercyjna; sprawdzić dostępność i licencje.

## Plan badań (zgodny z etapami z ANNSIM)
1. Przegląd w praktyce: instalacja, test jednego okna, czas, pamięć, obsługiwane scenariusze.
2. Taillard n = 20 na wszystkich: ILS z liczbą wywołań jak w kampaniach D-Wave; PRD vs D-Wave.
3. Pełny Taillard na najlepszych (SA/SQA są tanie — prawdopodobnie wszystkie).
4. Opcjonalnie: mały bieg D-Wave z zapisem pełnych zbiorów próbek (zależny od kwoty) → porównanie rozkładów energii.

## Do sprawdzenia
Czy kampanie P1–P4 zapisały próbki (zbiory próbek) czy tylko wyniki ILS; parametry SQA w OpenJij (Trotter,
harmonogram); dostępne σ ICE dla użytego solvera D-Wave; stan kwoty D-Wave.

## Symulatory wyżarzania (dodane 25.10.2026, decyzja użytkownika)

| Program | Co symuluje | Uwagi | Źródło |
|---|---|---|---|
| dimod `ExactSolver` / brute force | dokładny stan podstawowy | punkt odniesienia A1; okna do ~20 zmiennych | — |
| D-Wave `dwave-samplers` (SimulatedAnnealingSampler, Tabu) | klasyczne symulowane wyżarzanie / tabu | `dwave-neal` wycofany na rzecz `dwave-samplers` | https://sdk-docs.readthedocs.io/en/latest/docs_neal/reference/sampler.html |
| **OpenJij** (SASampler, **SQASampler**) | **symulowane wyżarzanie kwantowe** (Monte Carlo po całkach po trajektoriach), elastyczne harmonogramy na CPU | GPU-SQA nierozwijany; parametry Trottera i harmonogramu do sprawdzenia w API | https://dev.to/heavyfermion/1-an-introduction-to-openjij-26pe |
| QuTiP | dokładna dynamika (równania master) z harmonogramem wyżarzania | tylko bardzo małe okna (~≤ 8–10 kubitów) | — |

Model błędów sprzętu D-Wave do scenariusza A3: **ICE** (integrated control errors) — gaussowskie zaburzenia
h i J, część systematyczna i losowa ( https://docs.dwavesys.com/docs/latest/c_qpu_ice.html ); plus zanurzenie
(łańcuchy, zrywanie łańcuchów).

Dane ze sprzętu D-Wave już są: `results/experiments/` (31 katalogów kampanii P1–P4, windowed_qubo, pełny
Taillard), `results/qpu_timing.jsonl` (24 317 rekordów). Próbek (rozkładów) prawdopodobnie nie zapisywano —
porównanie na poziomie ILS/PRD; ewentualny mały bieg D-Wave z zapisem próbek zależy od kwoty.
