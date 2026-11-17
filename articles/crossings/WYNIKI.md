# Wyniki kampanii (Mac M4 Pro, 12–16.11.2026)

Dane: `results/crossings/{main,ablation,safe,anatomy}` (poza gitem), tabele: `tables/` ze skryptu
`scripts/analyze_crossing.py` (pełny wydruk: `tables/summary.txt`). Protokół jak w PLAN.md v3: ILS 10 s, start
z permutacji tożsamościowej, tabu 10, mushroom 10, 12 plików Taillarda × 10 instancji × 5 ziaren. Punkt końcowy:
RPD do dolnego ograniczenia po 10 s. Bloki to instancje (średnia z ziaren), więc w klasie jest 10 par.

Komplet: main 3600, ablation 1200, safe 3000 przebiegów, anatomia 110 instancji (rdzenie P, 15.11). Żaden przebieg
nie przekroczył budżetu o więcej niż 2 s ani o więcej niż trzy średnie iteracje, więc uśpienie Maca nie zaburzyło
pomiarów.

## Testy rozstrzygające (PLAN 7.4a, v3)

| Test | Wynik | Werdykt |
|---|---|---|
| P1 (H3): inw. vs N₀, Holm po 12 klasach | inwolucje gorsze w 12/12 klas, w rozstrzygających (n ≤ 100) 9/9; A12 ≈ 0 | H3 ✅ |
| H3 przy równej liczbie iteracji (`move_log`) | to samo, 12/12 | spadek nie wynika z wolniejszych iteracji |
| Page wzdłuż Dyn < N₀ < N₁ < P₁ < inw. | istotny w każdej klasie, łącznie p ≈ 10⁻⁵³ | trend potwierdzony |
| P2 (H9′): Dyn vs N₀, dwustronnie | Dyn lepszy w 7/9 klas rozstrzygających; tai20_5 i tai20_20 bez różnicy | H9′ (część 1) ✅ |

Spadek wzdłuż osi nie jest równomierny. N₁ i P₁ nie różnią się istotnie od N₀ (pojedyncze klasy przy n = 20).
Duże skoki są dwa: z Dynasearcha do N₀ (zagnieżdżanie) i z P₁ do inwolucji (dowolne krzyżowania).

## Analiza eksploracyjna

- **Dyn vs Swap (H9′, część 2):** przewaga Dynasearcha jest mała (0,1–0,6 pp przy n ≤ 100) i po Holmie istotna
  tylko przy n ≥ 200. W rankingu Friedmana Dyn i Swap zajmują dwa pierwsze miejsca w każdej klasie z n ≤ 200.
- **Kontrola implementacji:** Dyn z nowego kodu i k-bonacci z k = n z kampanii k-bonacciego dają te same wyniki
  (różnice ≤ 0,06 pp, żadna istotna). Przy n ≥ 200 małe k (k = 8) jest wyraźnie lepsze od Dynasearcha, bo w 10 s
  robi więcej ruchów.
- **E5, bezpiecznik:** sprowadza wszystkie rodziny do poziomu Dynasearcha. Dla N₀, N₁, P₁ i inwolucji jest istotnie
  lepszy od wersji bez bezpiecznika w prawie wszystkich klasach n ≥ 50, a N₀ z bezpiecznikiem nie różni się od
  Dynasearcha. Inwolucje z bezpiecznikiem odstają przy n ≥ 200, bo tam działa limit top-K (35–100% ruchów).
- **E3, dokładny filtr ścieżki krytycznej:** nie pomaga ani N₀, ani inwolucjom. Przy m = 5 (tai50_5, tai100_5)
  pogarsza wynik o 2–4,5 pp.
- **Budżety krótsze niż 10 s:** kolejność wariantów jest ta sama. Przy n = 200 do 1 s nie kończy się nawet pierwszy
  ruch, a n = 500 to jeden ruch w całym budżecie. Z tych klas nie wyciągamy wniosków o c (PLAN 7.6).

## E1: anatomia (stany z przebiegów bez bezpiecznika)

- **Ruch złożony gorszy od najlepszej pojedynczej zamiany:** Dyn 11–44%, N₀/N₁/P₁ 48–71%, inwolucje 74–91% stanów.
- **Zysk realny wobec przewidywanego:** Dyn 0,46–0,88; N₀/N₁/P₁ 0,03–0,21; inwolucje są ujemne, czyli ruch
  średnio pogarsza C_max.
- **H2 (addytywność par):** pary rozłączne sumują się istotnie lepiej niż nachodzące w 11/11 klas. Zagnieżdżone
  wypadają nie gorzej od krzyżujących się, przy n ≥ 50 prawie na równi. H2 ✅
- **H1 (nasycenie):** wyszło słabsze niż w pilocie. N₀ daje ≥ 60% przewidywanego zysku inwolucji w 8/11 klas,
  a przy tai100_20 i n = 200 tylko 50–59%. P₁ daje ≥ 85% tylko przy n = 20. Najmniejsze c z OPT_c = OPT_inw.
  wynosi co najmniej 2 w 65–100% stanów. H1 ⚠️: przewidywany zysk rośnie z c, ale bez nasycenia na c ≤ 1.
  Razem z H3 daje to spójny obraz: model addytywny obiecuje przy krzyżowaniach coraz więcej, a sprzęt C_max
  daje coraz mniej.

## Czego jeszcze nie ma

- c₂ w anatomii (DP dla c = 2 i MILP dla |C| ≤ 60): wymaga obliczeń, po kampanii SA k-bonacciego.
- Wiersze starego Motzkina i Dynasearcha: stare kampanie (`results/experiments/2026070*`) szły innym protokołem
  (warianty `quantum_*` przez sampler), więc nie wstawiam ich bez decyzji (D4).
- E4 (kwanty), SA, model drugiego rzędu (E3 MAX).
