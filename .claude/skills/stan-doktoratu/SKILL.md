---
name: stan-doktoratu
description: Przegląd stanu całego doktoratu – wszystkie artykuły i ich etapy, najbliższe terminy, rozjazdy między PUBLICATIONS.md, STATUS.md, notatką DOKTORAT i kalendarzem PWR, prace czekające na druk, gałęzie do scalenia. Użyj, gdy właściciel pyta „co mamy”, „jakie terminy”, „co dalej z doktoratem”, albo na początku tygodnia.
---

# Stan doktoratu

1. Zbierz stan z repo: `PUBLICATIONS.md`, `articles/*/STATUS.md`, `articles/*/submission_*/README.md`,
   `ROADMAP.md`, gałęzie (`git for-each-ref --sort=-committerdate refs/remotes`) i niescalone gałęzie
   (`git branch -r --no-merged origin/main`).
2. Porównaj z pamięcią projektu (`project_konferencje_plan.md`), notatką DOKTORAT i kalendarzem PWR
   (najbliższe 60 dni).
3. Odpowiedz krótko, jedną tabelą: praca | miejsce | stan | następny krok | termin. Pod nią:
   - terminy w ciągu 14 dni (pogrubione),
   - rozjazdy między źródłami (co gdzie poprawić – zaproponuj `/domkniecie`),
   - prace gotowe do druku (DOI jest, druku nie było) → przypomnij o `/druk`,
   - konflikty grafiku (dwa terminy w tym samym tygodniu, długie kampanie na Macu nakładające się na siebie).
4. Nic nie zmieniaj bez zgody – to przegląd.
