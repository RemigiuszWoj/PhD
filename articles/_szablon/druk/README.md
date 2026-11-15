# Druk – <slug>

1. `okladka.tex` → `okladka.pdf`
2. Złożenie: okładka + artykuł (offprint lub finalny PDF) + puste strony do wielokrotności 4:
   `qpdf --empty --pages okladka.pdf ../<artykul>.pdf -- do_druku.pdf`, potem dopełnienie pustymi stronami.
   Skalowanie do A5: `pdfjam --paper a5paper --outfile do_druku_a5.pdf do_druku.pdf`.
3. Podgląd, zamówienie (skill `/druk`), wpis w `../STATUS.md`.
