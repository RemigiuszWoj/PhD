# Zgłoszenie ICAISC 2027 — co wpisać w systemie

**Gdzie:** https://icaiscsystem.icaisc.eu/ („Submit a Paper” na icaisc.eu)
**Termin:** 15.11.2026 (strona podaje „Paper submission deadline”); **wysyłka planowana do 13.11.2026**.
Powiadomienie 28.02.2027, wersja finalna 25.03.2027 (tylko LaTeX).

**WYSŁANE 09.11.2026 wieczorem: Paper 12373** (system icaiscsystem.icaisc.eu; kategoria „Various Problems of Artificial Intelligence”, słowo kluczowe „Artificial intelligence in scheduling and optimization”, LLM: writing aid + other; pliki: ICAISC2027_anonymous.pdf + zrodla_z_nazwiskami.zip z main.pdf). Decyzja od 28.02.2027.

**Wcześniej, 06.11.2026 (wieczór): NIE WYSŁANE.** Po 8 rundach krytyka (EN/PL, wersja PL napisana od nowa
zgodnie z EN) tekst wprowadzony; `abstrakt.txt` odświeżony (09.11: 117 słów / 697 znaków; formularz ICAISC przyjmuje max 700 znaków);
`overleaf_source.zip` przebudowany. 12 stron EN (06.11: 6 skrótów; 07.11: wzory wydzielone, „quantum processor” zamiast „device”, 12 kolejnych skrótów, rysunki bez zmian; PL wyrównana treściowo z EN, 13 stron; wersja z nazwiskami 13 przez blok autorów; 09.11: uwagi promotora — prostszy abstrakt 216 słów, PRD „· 100%”, „1% do 26%”, łagodniejsze oświadczenie o AI; fizyczny komputer IQM Sirius wyeksponowany, także w tytule „…, Tested on a Real Quantum Computer” i słowie kluczowym „quantum hardware”; 4 drobne skróty, EN nadal 12 stron; „Disclosure of Interests” w credits, Rys. 4 niższy z legendą 6 pt (min. Springera), EN 12 stron; abstrakt skrócony do 697 znaków = limit pola formularza 700 znaków, ten sam w artykule i formularzu; średnik w oświadczeniu AI zamieniony na kropkę),
pdfinfo puste, fonty osadzone, bez Type 3.

## Wymagania ze strony ICAISC (sprawdzone 01.11.2026)
- pełny artykuł Springer LNAI (llncs), „preferably around 12 pages” — mamy 12 (EN)
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
