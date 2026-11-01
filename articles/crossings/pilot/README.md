# Pilot: sąsiedztwa z krzyżowaniami (01.11.2026)

Prototypy i piloty do `../PLAN.md`. Kod badawczy, nie produkcyjny: nowe tryby są podpięte do ILS z repo w skrypcie
(podmiana `get_neighbor`), bez zmian w `src/`. Docelowy moduł to `src/neighborhoods/classical/crossing.py` (Etap 0 planu).

Uruchamiać z tego katalogu (skrypty importują się nawzajem po nazwie, repo dodają do `sys.path` same).
Zależności spoza `requirements.txt`: `networkx`, `scipy` (MILP/HiGHS, Wilcoxon), dla części kwantowej `minorminer` (z dwave-system) i `qiskit`.

| Skrypt | Co robi | Czas (chmura, 4 rdzenie) |
|---|---|---|
| `counts.py` | brute force (n ≤ 11) vs J-ułamek; N_c, P₁, 3-nieprzecinające, bez zagnieżdżeń | ~10 s |
| `counts_asymptotics.py` | liczności do n = 2000, pokrycie inwolucji, prawo Poissona, stała wzrostu P₁ | ~4 s |
| `crossing_select.py` | dokładny wybór: c₀, c₁, P₁ (DP), inwolucje (networkx), dowolne c (MILP); `python crossing_select.py` = walidacja z brute force | ~20 s |
| `pilot_anatomy.py anatomy` | 192 stany (start + trajektoria Motzkina): zysk przewidywany i realny, krzyżowania, pary wg geometrii, sprzężenia QUBO | ~1 min |
| `pilot_anatomy.py descent` | zejście do optimum lokalnego Swap dla c₀, c₁, P₁, c₂, inwolucji (72 przebiegi) | ~1 min |
| `pilot_pairq.py` | model drugiego rzędu (poprawki par, MILP) vs wybór addytywny, 151 stanów | ~1 min |
| `pilot_ils.py <ms> [tryby]` | ILS z repo (bez zmian) z trybami `cr_*`; tai20_5 (ta001–ta004), tai20_10 (ta011–ta014), lcg50_10; 4 instancje × 3 ziarna | 1 s: ~2 min; 5 s: ~6 min |
| `pilot_qubo_structure.py` | QUBO Motzkina / inwolucji / z poprawkami par: sprzężenia, zanurzenie na Pegasusie P16, CZ w QAOA na siatce 4×5 | ~3 min |
| `timing_deltas.py` | czas delt dla wszystkich par (punkt odniesienia dla kosztu wyboru) | ~2 s |
| `taillard_lb.py` | maszynowe dolne ograniczenie i nasiona Taillarda | <1 s |
| `analyze_pilots.py [anatomy\|descent\|pairq\|ils\|qubo\|all]` | tabele cytowane w planie, z `results/*.json` | ~2 s |

Tryby w `pilot_ils.py`: `cr_c0`, `cr_c1`, `cr_pa1`, `cr_inv` (wybór addytywny), sufiks `_safe` (bezpiecznik: lepszy z ruchu złożonego
i najlepszej pojedynczej zamiany wg rzeczywistego C_max), `cr_swap` (sama najlepsza zamiana), `cr_pq<K>_c0` / `cr_pq<K>_inv`
(model drugiego rzędu na K najlepszych kandydatach) oraz stare `motzkin` i `dynasearch` z repo.

Uwagi:
- Czasy w chmurze są zaszumione; tabele do artykułu liczymy na Macu (plan, §7).
- Piloty liczą delty dla wszystkich par (bez filtra blokowego, czyli bezstratnie); kampania ma używać tego samego filtra co wcześniejsze prace (plan, D6).
- „lcg50_10” to instancje z generatora Taillarda z nasionami, których nie potwierdziłem jako ta041–ta044; RPD liczone do maszynowego LB.
