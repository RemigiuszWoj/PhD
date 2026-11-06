# Zgłoszenie ICAISC 2027 — co wpisać w systemie

**Gdzie:** https://icaiscsystem.icaisc.eu/ („Submit a Paper” na icaisc.eu)
**Termin:** 15.11.2026 (strona podaje „Paper submission deadline”); **wysyłka planowana do 13.11.2026**.
Powiadomienie 28.02.2027, wersja finalna 25.03.2027 (tylko LaTeX).

**STAN 06.11.2026 (wieczór): NIE WYSŁANE.** Po 8 rundach krytyka (EN/PL, wersja PL napisana od nowa
zgodnie z EN) tekst wprowadzony; `abstrakt.txt` odświeżony (249 słów, limit LNCS 150–250);
`overleaf_source.zip` przebudowany. 13 stron (strona prosi o ok. 12, do skrócenia przy formatowaniu),
pdfinfo puste, fonty osadzone, bez Type 3.

## Wymagania ze strony ICAISC (sprawdzone 01.11.2026)
- pełny artykuł Springer LNAI (llncs), „preferably around 12 pages” — mamy 13 (do skrócenia)
- recenzja **dwustronnie ślepa**: anonimowy PDF + źródła **z nazwiskami**
- strona prosi, by nie ujawniać autorów w podziękowaniach, linkach i bibliografii („self-references where
  possible”) — u nas własne prace w 3. osobie; trzy pozycje „under review” z nazwiskami (decyzja 24.10, wariant b)
- zasad dotyczących narzędzi AI strona nie podaje; oświadczenie jest w Acknowledgements (bez nazwisk)
- jedna opłata rejestracyjna = jeden artykuł

## Autorzy (do formularza i do źródeł z nazwiskami — jak w ACIIDS)

| | Autor 1 (corresponding) | Autor 2 |
|---|---|---|
| First name | Remigiusz | Wojciech |
| Last name | Wojewódzki | Bożejko |
| Email | remigiusz.wojewodzki@pwr.edu.pl | wojciech.bozejko@pwr.edu.pl |
| Country | Poland | Poland |
| Affiliation | Wrocław University of Science and Technology | Wrocław University of Science and Technology |

Blok autorów do wersji z nazwiskami (zamiast `\author{Anonymous}` / `\institute{Anonymous}`):
```
\author{Remigiusz Wojew\'{o}dzki \and Wojciech Bo\.{z}ejko}
\authorrunning{R. Wojew\'{o}dzki and W. Bo\.{z}ejko}
\institute{Department of Control and Quantum Computing,
  Faculty of Information and Communication Technology,
  Wroc\l{}aw University of Science and Technology, Wroc\l{}aw, Poland\\
  \email{remigiusz.wojewodzki@pwr.edu.pl, wojciech.bozejko@pwr.edu.pl}}
```

## Pola tekstowe (czysty tekst, bez LaTeX-a)
- **Title:** `tytul.txt`
- **Abstract:** `abstrakt.txt`
- **Keywords:** `slowa_kluczowe.txt` (5)

## Pliki do wgrania (przygotować po H2)
1. anonimowy PDF z `main.tex` (autor „Anonymous”, puste metadane)
2. źródła z nazwiskami: main.tex z blokiem autorów jak wyżej + references.bib, llncs.cls, splncs04.bst,
   figures/*_en.pdf, tables/*.tex (bez wersji PL)

## Kontrola przed wysyłką (01.11.2026, commit 8b3dd49)
- liczby w tekście i tabelach zgodne z danymi ✔
- 0 wspólnych fraz 8-wyrazowych z ACIIDS, GCMM, windowed, SOCO, ICAISC 2026, BPASTS poza fragmentem
  tytułu i oświadczeniem o LLM ✔
- pdfinfo: puste Title/Author/Subject/Keywords ✔; pdffonts: wszystko osadzone, brak Type 3 ✔
- DOI wszystkich 20 cytowanych prac sprawdzone (Crossref / DataCite) ✔ (06.11: +congram2002, DOI OK, razem 21; wieczorem +nowicki1996, barkoutsos2020cvar, galda2021,
  akshay2020, hadfield2019, kurowski2023, amaro2022, wszystkie DOI OK, razem 28); ICAISC 2026 jeszcze bez DOI
  w Crossref → „in press”
- po H2 powtórzyć: kompilacja, pdfinfo/pdffonts, liczba stron, liczba słów abstraktu
