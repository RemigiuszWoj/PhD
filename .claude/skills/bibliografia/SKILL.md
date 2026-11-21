---
name: bibliografia
description: Bibliografia do artykułu doktoratu – cytownik katedry (Google Sheets), wspólna pula bibliography/pula.bib (wszystkie references.bib + cytownik), weryfikacja każdego DOI w Crossref/DataCite, dobór ~20–30 pozycji, autocytowania w 3. osobie przy ślepej recenzji, wyciąg references.bib dla artykułu. Użyj przy dodawaniu źródeł, przeglądzie literatury, sprawdzaniu DOI.
---

# Bibliografia

**Baza wiedzy** (`~/Desktop/Doktorat/PhD-wiki/literatura/`): `kluczowe-zrodla.md` (najczęściej cytowane źródła i gdzie je cytujemy),
`grupa-bozejki.md` (prace grupy z cytownika) i strony pojedynczych prac. Nowe ważne źródło → strona w `literatura/`
(`/wiki`, dodanie źródła).

## Źródła pozycji

1. **Cytownik katedry** (Google Sheets, prowadzi katedra – prace Bożejki, Uchrońskiego, Wodeckiego, Gnatowskiego,
   Idzikowskiego, Kucharskiego i innych):
   arkusz prywatny – linku nie zapisywać w repo; identyfikator w zmiennej `CYTOWNIK_ID` w `.env`.
   Kolumny: ID, Rok, Autor, Tytuł, Tematyka (Optymalizacja / Kompilacja / ML / RNG / HHL), MNISW (punkty), IF,
   Bibtex, Bibitem, Cytowania. Jedna praca ma tyle wierszy, ilu autorów z katedry – liczy się ID.
   ```bash
   python tools/bib_pool.py cytownik     # pobiera arkusz -> bibliography/cytownik.bib (lokalnie, poza gitem)
   ```
   Pełna synchronizacja (raport zmian, przebudowa puli, brakujące nasze prace): skill `/cytownik`.
   **Stąd bierzemy prace grupy do cytowania** (wymóg promotora: cytować katedrę) i punkty MNiSW/IF.
   BibTeX w cytowniku pochodzi z Google Scholar: bez DOI, często bez tomu, z małymi literami w tytule, czasem
   ten sam klucz dla dwóch prac (skrypt dodaje wtedy literę `b`). Przed cytowaniem uzupełnij DOI, tom i strony
   z Crossref i wpisz pełny wpis do `references.bib` artykułu – wtedy pula bierze wersję zweryfikowaną.
   Prace bez BibTeX-a w arkuszu skrypt wypisuje (na 15.11.2026: ID 10, 16, 26) – szukać ręcznie.
   Gdy nasza praca wyjdzie, dopisać ją do cytownika (wiersz dla Wojewódzkiego i dla Bożejki) – przypomnieć
   właścicielowi w `/domkniecie`.
2. **Pula repo:** `bibliography/pula.bib` – wszystkie `articles/**/references.bib` + `bibliography/cytownik.bib`:
   ```bash
   python tools/bib_pool.py build
   python tools/bib_pool.py build --extra inny.bib     # dodatkowy plik
   ```
   Skrypt wypisuje: prace z cytownika, które pula już ma pod zweryfikowanym kluczem (używaj klucza z puli,
   np. `bozejko2024cinti`, nie `bozejko2024hybrid`), aliasy (ta sama praca pod dwoma kluczami) i konflikty
   (ten sam klucz, inna treść – rozstrzygnij ręcznie). Kluczy istniejących wpisów nie zmieniamy.
3. **Nowe pozycje** – z Crossref/DataCite/arXiv/link.springer.com, nigdy z pamięci modelu.

## Dobór (wzorzec z ICAISC 2027, ~20 pozycji na 12 stron LNCS; czasopismo 30–50)

| Grupa | Co | Ile |
|---|---|---|
| Nasze prace | SOCO 2026, ICAISC 2026, rozdział EXIT 2025, BPASTS, windowed QUBO (preprint SSRN), GCMM, ACIIDS… | 5–7 |
| PFSP i metaheurystyki | taillard1993, lourenco2003 (ILS), nawaz1983 (NEH), congram2002dynasearch, nowicki1996… | 2–5 |
| QUBO / annealing / QAOA | lucas2014ising, farhi2014qaoa, zhou2020, … | 3–6 |
| Tematyczne (szum, sprzęt, symulatory) | wg pytania badawczego | 2–6 |
| Grupa promotora (z cytownika) | bozejko2026new (Fibonacci), bozejko2024cinti, dudkiewicz2026depth, gnatowski2026routing… | 2–4 |

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
