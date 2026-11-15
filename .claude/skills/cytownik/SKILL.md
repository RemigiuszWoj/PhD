---
name: cytownik
description: Synchronizacja z cytownikiem katedry (prywatny arkusz Google z pracami grupy, BibTeX-em, punktami MNiSW i IF) – pobranie arkusza do lokalnego bibliography/cytownik.bib (poza gitem), raport zmian, przebudowa puli bibliography/pula.bib, lista naszych opublikowanych prac, których w arkuszu brakuje. Użyj, gdy właściciel mówi „zsynchronizuj cytownik”, „odśwież cytownik”, „co nowego w cytowniku”, przed doborem bibliografii do nowego artykułu i po publikacji naszej pracy.
---

# Synchronizacja cytownika katedry

Arkusz prowadzi katedra i jest **prywatny**. W repo (publicznym) nie ma ani linku, ani eksportu:

- identyfikator arkusza: `CYTOWNIK_ID=` w `.env` (skrypt czyta go sam; nie wypisuj go w czacie ani w plikach repo),
- eksport: `bibliography/cytownik.bib` – plik lokalny, w `.gitignore`; zawiera punktację MNiSW.

Brak `CYTOWNIK_ID` w `.env` → poproś właściciela o link do arkusza i dopisz sam identyfikator do `.env`.

## Arkusz → komputer

1. Zachowaj poprzedni eksport do porównania (jeśli jest):
   ```bash
   cp bibliography/cytownik.bib /tmp/cytownik_przed.bib 2>/dev/null
   ```
2. Pobierz arkusz:
   ```bash
   python tools/bib_pool.py cytownik
   ```
   Skrypt zapisuje jeden wpis na pracę (po ID) i wypisuje prace bez BibTeX-a w arkuszu.
3. Porównaj po wierszach `% cytownik ID …` i po kluczach `@…{klucz,` – powiedz właścicielowi krótko:
   nowe prace (ID, rok, pierwszy autor, tytuł), prace usunięte, zmieniona punktacja MNiSW, prace nadal bez BibTeX-a.
   Bez zmian → jedno zdanie.
4. Przebuduj pulę:
   ```bash
   python tools/bib_pool.py build
   ```
   Konflikty kluczy i aliasy rozstrzygaj jak w `/bibliografia`. `bibliography/pula.bib` jest w repo: zmianę pokaż
   właścicielowi przed commitem. Do puli idą same dane bibliograficzne, bez punktacji i bez linku.

## Komputer → arkusz

Arkusz edytuje człowiek, skill tylko przygotowuje wiersze.

1. Weź prace ze stanem „opublikowany” (aktywny DOI) z `PUBLICATIONS.md` i sprawdź po tytule, czy są
   w `bibliography/cytownik.bib`.
2. Dla każdej brakującej przygotuj komplet do wklejenia: Rok, Autor, Tytuł, Tematyka, MNISW, IF, Bibtex, Bibitem.
   Jeden wiersz na każdego autora z katedry. BibTeX z Crossref (DOI zweryfikowany), punkty z aktualnego wykazu
   MNiSW – niezweryfikowaną liczbę oznacz znakiem zapytania.
3. Po dopisaniu przez właściciela powtórz „Arkusz → komputer”, żeby sprawdzić, czy praca jest w eksporcie.

Prac w recenzji i przyjętych bez DOI do arkusza nie zgłaszamy.

## Czego nie robić

- Nie commitować `bibliography/cytownik.bib`, linku ani identyfikatora arkusza; przed każdym commitem na tej gałęzi
  `git grep -n 'docs.google.com/spreadsheets'` ma nie zwracać nic.
- Nie edytować arkusza przez przeglądarkę bez wyraźnej prośby właściciela.
- Nie cytować wpisu prosto z cytownika: BibTeX pochodzi z Google Scholar (bez DOI, często bez tomu i stron) –
  najpierw `/bibliografia`.
