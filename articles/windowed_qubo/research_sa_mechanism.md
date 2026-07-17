# Research: mechanizm "pełny skan → argmax → Metropolis" (nasze "SA")

Data: 2026-07-17. Pipeline: agent szukający (25 pozycji) → agent recenzujący
(weryfikacja Crossref wszystkich kluczowych, ocena B+) → falsyfikacja tez
(agent padł na limitach; domknięta przez orkiestratora na zebranym materiale).
Wszystkie DOI poniżej zweryfikowane w Crossref w trakcie researchu.

## Nasz algorytm (src/algorithms/sa.py)
Pełna ewaluacja otoczenia (DP / QPU zwraca najlepszy ruch złożony) → argmax
→ akceptacja Metropolisa → po odrzuceniu stan bez zmian i deterministyczna
re-propozycja TEGO SAMEGO ruchu (przy niższym T); chłodzenie geometryczne
z podłogą; reheat + double-bridge kick na stagnację.

## Kluczowe pozycje (zweryfikowane)

### Rodowód mechanizmu
- **Ishibuchi, Misaki, Tanaka 1995**, Modified simulated annealing algorithms
  for the flow shop sequencing problem, EJOR 81(2):388–398,
  10.1016/0377-2217(93)E0235-P — najlepszy z PRÓBKI k sąsiadów → Metropolis,
  NA PFSP; uodparnia SA na wybór harmonogramu chłodzenia. (+ wersja IJCNN 1991.)
- **Defersha, Obimuyiwa, Yimer 2020**, Multiple-Trial/Best-Move Simulated
  Annealing..., ISCMI 2020, s. 61–67, 10.1109/ISCMI51676.2020.9311570 —
  jedyna praca z "best-move SA" w NAZWIE; best-of-sample, FJSP.
  (+ wersja czasopismowa CIE 171:108487, 2022 — mechanizm niezweryfikowany.)
- **Alizamir, Rebennack, Pardalos 2008**, IntechOpen (10.5772/5571) —
  optimal stopping do decyzji, ilu sąsiadów próbkować, potem Metropolis.
- **Kalender, Kheiri, Özcan, Burke 2013**, A greedy gradient-simulated
  annealing selection hyper-heuristic, Soft Computing 17(12):2279–2292,
  10.1007/s00500-013-1096-5 — greedy selekcja (wszyscy kandydaci, bierz
  najlepszego) × akceptacja SA, na poziomie hiper-heurystyk.
- **Franzin, Stützle 2019**, Revisiting simulated annealing: A component-based
  analysis, C&OR, 10.1016/j.cor.2018.12.015 — katalog komponentu
  "exploration criterion": NE1 losowy sąsiad / NE2 skan sekwencyjny
  (Connolly 1990, EJOR 46(1):93–100, 10.1016/0377-2217(90)90301-Q) /
  NE3 best-of-sample (Ishibuchi) / NE4 first-improvement.
  **BRAK opcji: wyczerpujący argmax — nasz wariant to brakująca komórka.**

### Rodzina rejection-free / sprzęt
- **Bortz, Kalos, Lebowitz 1975** (n-fold way), J. Comput. Phys. 17:10–18,
  10.1016/0021-9991(75)90060-1 — przodek: pełny skan, następca losowany
  z wag akceptacji.
- **Greene, Supowit 1986**, SA Without Rejected Moves, IEEE TCAD 5(1):221–228,
  10.1109/TCAD.1986.1270190.
- **Rosenthal i in. 2021**, Jump Markov chains and rejection-free Metropolis,
  Comput. Stat. 36:2789–2811, 10.1007/s00180-021-01095-2; + Chen i in.
  arXiv:2205.02083 (PNS).
- **Aramon i in. 2019**, Digital Annealer, Front. Phys. 7:48,
  10.3389/fphy.2019.00048 — parallel-trial: WSZYSCY sąsiedzi dostają test
  Metropolisa w każdym kroku + dynamic offset (odpowiednik naszego reheatu).
- **Fukushima-Kimura i in. 2023**, Mathematical Aspects of the Digital
  Annealer's SA, J. Stat. Phys. 190, 10.1007/s10955-023-03179-3 —
  najbliższa istniejąca teoria zbieżności łańcucha z pełnym skanem.

### Teoria propozycji nienlosowych
- **Zanella 2020**, Informed Proposals for Local MCMC in Discrete Spaces,
  JASA 115(530):852–865, 10.1080/01621459.2019.1585255 — greedy limit bez
  korekty Hastingsa przestaje celować w rozkład Boltzmanna.
- **Liu, Liang, Wong 2000**, Multiple-Try Metropolis, JASA 95(449):121–134,
  10.1080/01621459.2000.10473908 — poprawna (detailed balance) wersja
  "wybierz najlepszego z k".
- **Manousiouthakis, Deem 1999**, Strict detailed balance is unnecessary...,
  J. Chem. Phys. 110(6):2753–2756, 10.1063/1.477973 — stacjonarność przy
  słabszym warunku balansu (legitymizuje deterministyczny PORZĄDEK, ale nie
  deterministyczną PROPOZYCJĘ argmax).
- **Moscato, Fontanari 1990**, Phys. Lett. A 146(4):204–208,
  10.1016/0375-9601(90)90166-L — deterministyczny próg zamiast stochastycznej
  akceptacji bez straty jakości; **Franz, Hoffmann, Salamon 2001**, PRL
  86(23):5219–5222 — optymalna strategia szukania minimów jest deterministyczna.
- **Martin, Otto, Felten 1992**, ORL 11(4):219–224, 10.1016/0167-6377(92)90028-2
  + **Martin, Otto 1996**, Ann. OR 63:57–75, 10.1007/BF02601639 — large-step
  Markov chains: rodowód naszego double-bridge kicka przy akceptacji Metropolisa.
- **Layden i in. 2023**, Quantum-enhanced MCMC, Nature 619:282–287,
  10.1038/s41586-023-06095-4; **Arai, Kadowaki 2025**, Sci. Rep. 15,
  10.1038/s41598-025-07293-y — kwantowy proponent + klasyczny MH:
  najbliższa architektura do naszego QPU-w-pętli.

## Werdykty falsyfikacji tez

- **T1** (brak ustalonej nazwy dla dokładnie naszego wariantu): **UTRZYMANA,
  osłabiona** — okolica jest gęsta (Defersha "best-move SA", GGSA, parallel-trial),
  ale żadna nazwa nie pokrywa wszystkich osi: wyczerpujący argmax po otoczeniu
  ruchów + re-propozycja po odrzuceniu.
- **T2** (nasz wariant = graniczny przypadek Ishibuchiego): **UTRZYMANA
  Z POPRAWKĄ** — granica jest jakościowo inna: u Ishibuchiego po odrzuceniu
  losuje się świeżą próbkę (łańcuch nieredukowalny przy stałym T), u nas
  deterministyczna re-propozycja (redukowalność/2-cykl). W artykule pisać
  "limit case" TYLKO z tym zastrzeżeniem.
- **T3** (rejection-free najbliższą rodziną): **UTRZYMANA jako "jedna z
  najbliższych"** — równie blisko: parallel-trial DA (inna oś: Metropolis
  per-sąsiad) i quantum-enhanced MCMC (najbliższa architektura QPU-w-pętli).
- **T4** (2-cykl specyficzny dla determinizmu pełnego skanu; kick strukturalnie
  konieczny): **UTRZYMANA** — nigdzie nie opisano tej patologii (negatywny
  wynik obu agentów). Zastrzeżenie recenzenta: przy STAŁYM T re-propozycja
  = akceptacja argmaxu po geometrycznym opóźnieniu; ale przy chłodzeniu
  p_akc maleje z każdą próbą (oczekiwany czas ucieczki rozbiega), a koszt
  iteracji jest zdominowany przez oraculum (2.1 s sieci przy QPU) — kick
  uzasadniony strukturalnie i ekonomicznie.
- **T5** (złamany detailed balance ⇒ brak gwarancji): **CZĘŚCIOWO
  SFALSYFIKOWANA W SFORMUŁOWANIU** — detailed balance per se nie jest
  konieczny (Manousiouthakis–Deem), a dla pokrewnych łańcuchów pełnego skanu
  istnieje teoria (Fukushima-Kimura). Poprawne sformułowanie: "żadna
  opublikowana analiza zbieżności nie obejmuje tego łańcucha; argumenty
  balansowe nie stosują się, bo propozycja jest deterministyczna
  (nieredukowalność pada przy stałym T)".
- **T6** (zostawić nazwę "SA" + precyzyjny akapit mechanizmu): **UTRZYMANA
  I WZMOCNIONA** — konwencja pola to "SA + kwalifikator" (modified SA,
  multiple-trial/best-move SA, greedy gradient-SA), a taksonomia
  Franzina–Stützle daje gotową ramę: nasz wariant = brakująca opcja
  komponentu exploration criterion.

## Rekomendowany akapit do artykułu (do decyzji użytkownika)
SA zostaje; w Metaheuristics dopisać: mechanizm = exploration criterion
"best of the entire neighborhood" — rozszerzenie NE3 Ishibuchiego do pełnego
otoczenia (z zastrzeżeniem o jakościowej zmianie przy granicy), wymuszone
przez QPU (zwraca najlepszy ruch, nie losowy); rodzina: rejection-free /
parallel-trial (Digital Annealer) i quantum-enhanced MCMC; patologia 2-cyklu
jako konsekwencja determinizmu i kick jako strukturalna odpowiedź
(rodowód: Martin–Otto). Kandydaci do bibliografii: ishibuchi95, franzin19,
defersha20, rosenthal21, aramon19 (+ ew. zanella20, martin92).
