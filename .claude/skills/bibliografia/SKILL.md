---
name: bibliografia
description: Bibliografia do artykułu doktoratu – wspólna pula bibliography/pula.bib (z wszystkich references.bib i eksportu z cytownika), weryfikacja każdego DOI w Crossref/DataCite, dobór ~20–30 pozycji, autocytowania w 3. osobie przy ślepej recenzji, wyciąg references.bib dla artykułu. Użyj przy dodawaniu źródeł, przeglądzie literatury, sprawdzaniu DOI.
---

# Bibliografia

## Źródła pozycji

1. **Pula repo:** `bibliography/pula.bib` – budowana z wszystkich `articles/**/references.bib`:
   ```bash
   python tools/bib_pool.py build                       # z repo
   python tools/bib_pool.py build --extra <cytownik>.bib  # + eksport z cytownika
   ```
   Skrypt wypisuje aliasy (ta sama praca pod dwoma kluczami – w nowych tekstach używaj wskazanego klucza)
   i konflikty (ten sam klucz, inna treść – rozstrzygnij ręcznie). Kluczy istniejących wpisów nie zmieniamy.
2. **Cytownik właściciela** – eksport do `.bib` i `--extra`. Ścieżkę zapisz w tym pliku, gdy będzie znana
   (`CYTOWNIK=`…), żeby kolejne sesje nie pytały.
3. **Nowe pozycje** – z Crossref/DataCite/arXiv/link.springer.com, nigdy z pamięci modelu.

## Dobór (wzorzec z ICAISC 2027, ~20 pozycji na 12 stron LNCS; czasopismo 30–50)

| Grupa | Co | Ile |
|---|---|---|
| Nasze prace | SOCO 2026, ICAISC 2026, rozdział EXIT 2025, BPASTS, windowed QUBO (preprint SSRN), GCMM, ACIIDS… | 5–7 |
| PFSP i metaheurystyki | taillard1993, lourenco2003 (ILS), nawaz1983 (NEH), congram2002dynasearch, nowicki1996… | 2–5 |
| QUBO / annealing / QAOA | lucas2014ising, farhi2014qaoa, zhou2020, … | 3–6 |
| Tematyczne (szum, sprzęt, symulatory) | wg pytania badawczego | 2–6 |
| Grupa promotora | bozejko2026new (Fibonacci), bozejko2024cinti, prace z katedry | 2–4 |

Pilnuj proporcji autocytowań (recenzenci to wytykają; w GCMM trzeba było poszerzyć do 22 pozycji).

## Weryfikacja (obowiązkowo przed wysyłką)

```bash
python tools/bib_pool.py check articles/<slug>/references.bib   # Crossref, potem DataCite
```
- Każdy DOI musi istnieć i tytuł się zgadzać. Prace „in press” bez DOI w Crossref → `note = {in press}`.
- Tomy Springera (LNCS/LNAI/CCIS): seria, numer tomu, strony z link.springer.com.
- Podkreślniki w DOI w LaTeX-u: `\detokenize{...}` albo pakiet `doi`/`url` – inaczej Overleaf się wysypie.
- W nagłówku `references.bib` komentarz: skąd i kiedy zweryfikowane (`% DOI zweryfikowane w Crossref 23.10.2026`).

## Ślepa recenzja

Własne prace cytujemy w 3. osobie („Wojewódzki and Bożejko [5] showed…” → „It was shown [5]…”), bez „our previous
work”. Prace „under review” – albo anonimowo (`anon2026a` …), albo z nazwiskami w 3. osobie; decyzję zapisz
w `STATUS.md`. W wersji z nazwiskami/finalnej wracają normalne cytowania.

## Wyciąg dla artykułu

```bash
python tools/bib_pool.py extract articles/<slug>/main.tex -o articles/<slug>/references.bib
```
Brakujące klucze dodaj ręcznie do `references.bib` artykułu (po weryfikacji), potem `build`, żeby trafiły do puli.
Commit puli razem z artykułem: `bibliography: add 6 verified references for <slug>`.
