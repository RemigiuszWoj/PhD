---
name: domkniecie
description: Aktualizacja stanu artykułu doktoratu wszędzie naraz i domknięcie po publikacji – PUBLICATIONS.md, STATUS.md, notatka DOKTORAT (tabela publikacji i konferencji), kalendarz PWR, pamięć projektu, ORCID i Google Scholar po DOI, offprint, scalenie gałęzi, przypomnienie o druku. Użyj po każdej zmianie stanu (wysłany, decyzja, przyjęty, korekta, DOI) i gdy właściciel mówi „zaktualizuj tabelki/kalendarz”.
---

# Domknięcie i aktualizacja stanu

Stany artykułu: `w przygotowaniu → wysłany/w recenzji → decyzja (accept/minor/major/reject) → rewizja wgrana →
przyjęty → korekta (proofs) → opublikowany (DOI) → wydrukowany`.

## Przy każdej zmianie stanu – lista miejsc (odhacz wszystkie, powiedz, czego nie dało się zrobić)

1. **`articles/<slug>/STATUS.md`** – stan, daty, numer pracy, następny krok.
2. **`PUBLICATIONS.md`** – sekcja wg stanu (Opublikowane / W druku / W recenzji / W przygotowaniu), pełne dane
   bibliograficzne; data „Stan na”.
3. **Notatka „DOKTORAT”** (Notatki Apple): 
   - tabela publikacji – kolumna stanu (np. „przyjęty, rewizja wgrana 13.11”, „w recenzji (zgł. 23.10, nr 316)”),
     link (DOI / Zenodo), daty konferencji;
   - tabela konferencji – usuń wiersz, gdy praca została wysłana (przechodzi do tabeli publikacji); popraw terminy
     i kolumnę „Artykuł”.
   - Edycja przez AppleScript albo computer use. **Przed edycją** zanotuj liczbę załączników/zdjęć, **po edycji**
     sprawdź, że są (tabele w notatce bywają załącznikami, jest zdjęcie tablicy). Uważaj, żeby tekst nie trafił
     do listy notatek albo innej notatki. Notatki same zmieniają pierwszą literę na wielką – to w porządku.
4. **Kalendarz „PWR”** – nowe terminy (decyzja, rewizja, wersja finalna, korekta, konferencja całodniowo z alarmem
   tydzień i dzień przed; w opisie nr pracy, sesja, link). Sprawdź, czy nie ma już takiego wpisu. Stare zadania
   typu „po decyzji dodać do ORCID” zostaw, dopóki nie są zrobione.
5. **Pamięć projektu** (`project_<slug>.md`, `project_konferencje_plan.md`) – stan, gałęzie, commity, decyzje.
6. **Gałęzie** – po wysyłce/akceptacji scalić do `main` (zapytać), push (zapytać).

## Po publikacji (DOI działa)

- DOI, tom, strony do `PUBLICATIONS.md`, `references.bib` wszystkich artykułów, które tę pracę cytują
  (zamiast „in press”), i do puli (`python tools/bib_pool.py build`).
- Offprint/wersja wydawcy do `articles/<slug>/` (jak SOCO), jeśli licencja pozwala.
- **Cytownik katedry** (arkusz Google, prywatny; link poza repo) – dopisać pracę z BibTeX-em, punktami MNiSW i IF
  (wiersz dla każdego autora z katedry); potem `python tools/bib_pool.py cytownik && python tools/bib_pool.py build`.
- **ORCID** i **Google Scholar** – dodaj pracę (właściciel loguje się sam; prowadź krok po kroku).
- Zenodo – powiąż rekord kodu z DOI artykułu („is supplement to”).
- Rozliczenia uczelni (sprawozdanie doktoranta, punkty) – przypomnij, jeśli właściciel prowadzi ich listę.
- **Przypomnienie o druku:** gdy DOI jest, wersja wydawcy w repo, a foldery uporządkowane (`STATUS.md` kompletny)
  – zapytaj „drukujemy egzemplarz?” i przejdź do `/druk`. Jeśli właściciel odkłada, dodaj zadanie w kalendarzu PWR
  „Druk: <tytuł skrócony>” na najbliższy piątek i zaznacz w `STATUS.md` „druk: odłożony”.

## Raport dla właściciela

Krótko: co zaktualizowane (lista miejsc), czego nie i dlaczego, co zostało po jego stronie (np. wysłać mail,
podpisać LtP, dodać do ORCID).
