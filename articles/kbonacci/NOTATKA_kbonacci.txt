# Notatka: artykuł o sąsiedztwach k-bonacci (PFSP)

Stan: 01.11.2026 · gałąź `kbonacci-icaps` · repo RemigiuszWoj/PhD

## 1. W jednym akapicie

Badamy rodzinę sąsiedztw **k-bonacci** dla permutacyjnego problemu przepływowego (PFSP, kryterium C_max). Ruch to zbiór zamian końców na rozłącznych
przedziałach, przy odległości zamiany co najwyżej k−1 (środek przedziału stoi). Rodzina wypełnia lukę między sąsiedztwem Fibonacciego (k = 2)
a Dynasearch (k = n). Pytanie badawcze: **jak daleko powinna sięgać zamiana?** Eksperyment ma identyczny protokół jak poprzednie artykuły
(te same ILS i SA, budżety, ziarna, miara), więc nowe wyniki da się wprost zestawić ze starymi. Nowa jest tylko klasa sąsiedztw.

## 2. Definicja i najważniejsze fakty

| Sąsiedztwo | Odległości zamian | k | Liczba ruchów (n = 10) | Wzrost |
|---|---|---|---|---|
| Fibonacci | 1 | 2 | 89 | 1,618ⁿ |
| tribonacci | 1, 2 | 3 | 274 | 1,839ⁿ |
| tetranacci | 1, 2, 3 | 4 | 401 | 1,928ⁿ |
| k-bonacci | 1 … k−1 | k | rekurencja k-bonacciego | → 2ⁿ |
| Dynasearch | 1 … n−1 | n | 512 | 2ⁿ⁻¹ |

- **Liczność:** ruch to pokrycie pozycji kafelkami długości 1 lub 2…k, więc a_p = a_{p−1} + … + a_{p−k}.
- **Pokrycie Dynasearch:** przy k ≈ log₂n + 2 rodzina zawiera ok. 80% wszystkich ruchów Dynasearch (prawo najdłuższej serii). Przy stałym k ułamek maleje wykładniczo.
- **Koszt:** delty O(mnk²) zamiast O(mn³), pozycyjne DP O(nk). Implementacja jest równoważna Dynasearch z `L_max = k`, a wkładem jest rodzina i studium k.
- **Addytywność:** DP sumuje delty niezależnych zamian, a w PFSP makespan nie jest addytywny. Na pilocie sumowanie zawyżało poprawę o ok. 10% (k = 2) do ok. 50% (k ≥ 8).

## 3. Wyniki pilota (ILS, 1 s, RPD do UB, stary filtr, mała próba)

| | 20×5 | 20×10 | 50×10 |
|---|---|---|---|
| Adjacent | 1,59 | 3,26 | 11,73 |
| k = 2 (Fibonacci) | 1,31 | 2,46 | 8,37 |
| k = 3 (tribonacci) | 0,92 | 1,84 | 5,94 |
| k = 6 | 0,74 | 1,09 | 4,85 |
| k = n | **0,55** | **0,99** | **3,62** |
| stary Dynasearch | 0,94 | 1,49 | 5,30 |

Wnioski wstępne: jakość rośnie z k, a przy n ≤ 50 i 1 s wygrywa k = n. Wewnętrzne optimum k szukamy dla n ≥ 100 i krótkich budżetów.
Nowe DP robi z Dynasearch 2–3× więcej iteracji. To tylko pilot, a pełne wyniki da kampania.

## 4. Decyzje wiążące

- Protokół jak w poprzednich pracach: ILS i SA bez zmian, start z permutacji identycznościowej, budżety 0,1 / 0,5 / 1 / 2 / 5 / 10 s, ziarna 0–4, RPD do LB, test Wilcoxona.
- Filtr blokowy bez zmian (dokładny filtr tylko w ablacji `kbonacci_k3_cp`, `kbonacci_k8_cp`).
- Jeden przebieg 10 s na konfigurację, mniejsze budżety wyciągamy z logu ruchów (z automatyczną walidacją).
- **Najpierw tylko ILS** (~23 h na Macu), potem ewentualnie SA.
- Dodane, bo nie zmieniają obliczeń: korekta Holma, Friedman + Nemenyi, RPD do UB-1993 i do BKS.
- Odłożone: start z NEH, budżety t·n·m/2, Iterated Greedy, adaptacyjne k.
- Kwanty etapami MIN → OPT → MAX, po klasyce (D-Wave Advantage; IQM Sirius i Garnet).

## 5. Mapa plików (gałąź `kbonacci-icaps`)

| Plik | Co to |
|---|---|
| `articles/kbonacci/PLAN.md` | pełny plan: hipotezy H0–H7, eksperymenty, ryzyka; **sekcja A.0 i A obowiązują** |
| `articles/kbonacci/RUN_LOCAL.md` | instrukcja uruchomienia na Macu (nadrzędna dla lokalnego modelu) |
| `articles/kbonacci/main.tex`, `wersja_pl.tex` | szkice artykułu EN i PL (PDF-y poza repo, bo `.gitignore`) |
| `articles/kbonacci/pilot/` | surowe wyniki pilota |
| `src/neighborhoods/classical/kbonacci.py` | rdzeń: kandydaci, DP, tryby `kbonacci_k<K>`, `kbonacci_kn`, `*_cp` |
| `src/neighborhoods/classical/insert.py` | Insert (linia odniesienia) |
| `src/neighborhoods/quantum_qubo/kbonacci.py`, `gate_qaoa/kbonacci.py` | wersje D-Wave i IQM |
| `scripts/run_kbonacci_campaign.py` | kampania klasyczna (etapy validate, replicate, main, insert, ablation) |
| `scripts/extract_budgets.py` | wyciąganie budżetów z logu i walidacja |
| `scripts/analyze_kbonacci.py` | tabele, Wilcoxon, Friedman, BKS |
| `scripts/run_quantum_tiers.py` | kwanty poziomami z kontrolą budżetu |
| `data/taillard_pfsp_bks.json` | aktualne najlepsze znane makespany (źródło i kontrole w pliku) |

## 6. Plan działania

1. Mac: etapy `validate` + `replicate` (~2,2 h). Werdykt walidacji i porównanie ze starą kampanią.
2. Mac: `main` + `insert` + `ablation` (~20,5 h, ILS).
3. Analiza i tabele do artykułu, decyzja o SA.
4. Kwanty: MIN, potem OPT i MAX, jeśli starczy budżetu (D-Wave ≤ 225 s QPU łącznie, IQM ok. 200 kredytów).
5. Wypełnienie szkicu i **checkpoint ~22.11:** ICAPS (abstrakt ~7.12, artykuł ~14.12) albo IJCAI (~styczeń). Terminy niepotwierdzone, bo oficjalne strony są zablokowane w sesji.

## 7. Sprawy otwarte

- Prace promotora (Sci Rep 2026, IFAC 2025): sprawdzić, czy nie zawierają już tej hierarchii.
- Potwierdzić listę BKS z promotorem przed cytowaniem.
- Kompletność starych wyników (audyt wskazywał braki: Motzkin n = 500, nieważny Dynasearch n = 500).
- Dokładne daty i limity stron z oficjalnych CFP.
- Znany problem z poprzednich prac: filtr blokowy sprawdza tylko dwie ostatnie maszyny i odrzuca 2–7% zamian poprawiających (`PLAN.md` §14). Decyzja z promotorem.
- SA nie jest powtarzalny nawet przy tym samym ziarnie (reheat zależy od zegara): wartość do opisania w artykule.

## 8. Prompt startowy dla lokalnego modelu (wklej na Macu)

Model: **Sonnet 5.5, średni wysiłek.** Haiku najwyżej do późniejszych krótkich sesji „status”.

```
Jesteś na moim Macu w repozytorium PhD. Przejmujesz uruchomienie kampanii obliczeniowej do artykułu o sąsiedztwach k-bonacci (PFSP), przygotowanej i przetestowanej w chmurze na gałęzi kbonacci-icaps.

1. Najpierw git status. Jeśli są lokalne zmiany, zatrzymaj się i pokaż mi je (nie stashuj, nie kasuj). Potem: git fetch origin && git checkout kbonacci-icaps && git pull. Przeczytaj w całości articles/kbonacci/RUN_LOCAL.md oraz sekcję A.0 w articles/kbonacci/PLAN.md. RUN_LOCAL.md jest nadrzędny.

2. Zasady bezwzględne: nie zmieniaj config.yaml, src/algorithms/*, filtra blokowego, ziaren, budżetów ani instancji (porównywalność z poprzednimi artykułami). Nie aktualizuj pakietów Pythona, doinstaluj tylko brakujące. Liczymy tylko ILS. SA i obliczenia kwantowe wyłącznie na moje wyraźne polecenie. Nie commituj results/ ani .env i nie usuwaj results/.

3. Sekcja 0 RUN_LOCAL.md (środowisko, testy, verify_taillard, dry-run). Jeśli którykolwiek warunek nie jest spełniony, zatrzymaj się i pokaż mi dokładny wynik.

4. Sekcja 1: uruchom w tle etapy validate i replicate (PYTHONUNBUFFERED=1 nohup caffeinate …). Sprawdzaj postęp co ok. 10 minut. Po zakończeniu:
   a) werdykt walidacji: przy „VALIDATION GATE: FAIL” zatrzymaj się i pokaż tabelę;
   b) kontrola replikacji: znajdź katalog starej kampanii klasycznej, uruchom porównanie z RUN_LOCAL.md i oceń kolumny iteracji (różnica ≤ ~10%) oraz RPD (≤ ~0,3 pp przy 10 s). Przy większych różnicach zatrzymaj się i pokaż tabelę.
   Obie tabele zapisz do articles/kbonacci/validation.txt.

5. Jeśli punkt 4 przeszedł, uruchom w tle sekcję 2 (main, insert, ablation) i podaj mi PID, przewidywany czas zakończenia oraz komendy do sprawdzania postępu. Nie musisz czekać do końca. Gdy wrócę i napiszę „status”, sprawdź postęp i warunki stopu z RUN_LOCAL.md.

6. Po zakończeniu kampanii: sekcja 3 (kompletność starych wyników, analiza z --bks data/taillard_pfsp_bks.json), sekcja 4 (raport) i sekcja 5 (commit i push na kbonacci-icaps). Nie interpretuj ponad dane i nie podejmuj decyzji o SA ani o kwantach.

Odpowiadaj po polsku, zwięźle, z liczbami.
```

## 9. Jakość przygotowania

163 testy (Python 3.11 w czystym środowisku i 3.13), pięć iteracji przeglądu krytycznego (dziennik w `PLAN.md`), testy dymne wszystkich skryptów, automatyczny werdykt walidacji i kontrola replikacji starej kampanii.
