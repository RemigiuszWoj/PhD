# Szablon nowego artykułu

`cp -r articles/_szablon articles/<slug>` (patrz skill `/nowe-badanie`). Potem dołóż klasę LaTeX konferencji
i utwórz `figures/`, `tables/`, `measurements/` w miarę potrzeby.

| Plik | Etap | Skill |
|---|---|---|
| `STATUS.md` | cały cykl – stan, terminy, lista kontrolna | wszystkie |
| `PLAN_BADAN.md` | badania | `/badania` |
| `PLAN_ARTYKULU.md` | pisanie | `/artykul` |
| `WYNIKI.md` | wyniki etapów | `/badania` |
| `main.tex` | szkielet EN (Springer llncs) – PL: skopiuj do `main_pl.tex` | `/artykul` |
| `submission/README.md` | zgłoszenie → przemianuj folder na `submission_<KONF>` | `/zgloszenie` |
| `response/` | rewizja | `/rewizja` |
| `druk/okladka.tex` | druk egzemplarza | `/druk` |
