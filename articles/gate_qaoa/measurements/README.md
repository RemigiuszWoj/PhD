# Pomiary na sprzecie kwantowym — linia gate_qaoa

PFSP Taillard `tai20_5`, n=20, m=5, instancje 0-9, seed 0 (start identycznosciowy),
p=1, 4096 shotow, okno 6 dla sasiedztw gestych, dokladnie jeden ruch na przebieg.

## Srednie PRD [%] (odchylenie populacyjne, jak w Tabeli 2 artykulu)

| Sasiedztwo | ibm_fez ILS | ibm_fez SA | sirius ILS | garnet ILS |
|---|---|---|---|---|
| adjacent | 25.04 ±9.99 (n=10) | 25.04 ±9.99 (n=10) | 25.79 ±10.26 (n=9) | 12.82 (n=1) |
| fibonacci | 23.41 ±9.21 (n=10) | 23.48 ±9.02 (n=10) | 24.39 ±8.99 (n=9) | 12.01 (n=1) |
| dynasearch | 22.12 ±8.04 (n=10) | 22.38 ±8.72 (n=10) | — | 15.10 (n=1) |
| motzkin | 23.12 ±8.00 (n=10) | 23.59 ±8.12 (n=10) | — | 14.85 (n=1) |

## Stan pokrycia

- **ibm_fez**: 80 przebiegow — 10 instancji x 4 sasiedztwa x ILS i SA, komplet
- **sirius**: 18 przebiegow — 9 instancji x adjacent i fibonacci x ILS (instancja 6 nie miesci sie w 16 kubitach: QUBO ma 18 zmiennych)
- **garnet**: 4 przebiegi — instancja 0 x 4 sasiedztwa x ILS; dane surowe utracone, wartosci odtworzone z NOTES.md
- **emerald**: brak pomiarow

## Do domkniecia

- sirius: dynasearch i motzkin na 9 instancjach — ~51 kredytow
- garnet: pozostale 9 instancji x 4 sasiedztwa — ~97 kredytow

Stawki zmierzone: garnet 0,25 + 0,70 na obwod; sirius 0,25 + 0,43. 1 kredyt = 2 s QPU ~ 0,60 USD.
