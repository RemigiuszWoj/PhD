---
name: badania
description: Prowadzenie badań do artykułu doktoratu (PFSP, sąsiedztwa, QUBO, D-Wave, QAOA/IQM/IBM) – plan badań z pytaniami Q1…Qn i etapami E0…E5, pilot, pomiar czasu, kampanie w tle na Macu, dane w measurements/, statystyka, raporty etapów. Użyj przy planowaniu eksperymentów, uruchamianiu kampanii, analizie wyników.
---

# Badania

**Baza wiedzy** (`~/Desktop/Doktorat/PhD-wiki`, skill `/wiki`): na start przeczytaj stronę artykułu (`publikacje/`), pojęcia, których dotyczy
plan (`pojecia/`), oraz `kod/kampanie-i-dane.md` i `kod/backendy-kwantowe.md`. Po każdym etapie dopisz wnioski
słownie do sekcji „Metoda” i „Wyniki” strony artykułu.

## Plan badań (`articles/<slug>/PLAN_BADAN.md`, szablon w `articles/_szablon/`)

Układ sprawdzony w ICAISC 2027 i ANNSIM 2027:
1. **Nagłówek:** stan na datę, gałąź, termin, cel wysyłki, format (strony, ślepa recenzja).
2. **Pytanie badawcze** rozbite na Q1…Qn; każde mierzalne (co, na czym, jaką miarą).
   Na końcu wszystko sprowadzamy do **Cmax i PRD w ILS** – tak jak w poprzednich pracach.
3. **Co już mamy (sprawdzone <data>)** – z odwołaniem do plików i commitów. Nie z pamięci: sprawdź `git grep`,
   policz z danych.
4. **Usterki i braki wykryte przy planowaniu** – numerowane, z decyzją i datą (np. backend bez modelu szumu,
   za mały zbiór kalibracyjny, stały `maxiter`).
5. **Zakres** – tabela: instancje (Taillard: tai20_5/10/20, …, tai500_20), sąsiedztwa, parametry, shoty, ziarna,
   budżety czasu; uzasadnienie, czego nie robimy.
6. **Etapy E0…E5** – każdy z wejściem, wyjściem, kosztem (czas Maca, QPU/kredyty), kryterium „gotowe”.
7. **Ryzyka i decyzje** – D1…Dn z datą; otwarte pytania do promotora.

## Zasady prowadzenia

- **Najpierw pilot**, potem kampania. Zmierz czas na małym wycinku i podaj godzinę końca.
- Kampanie liczą się w tle na Macu łańcuchem etapów (jeden po drugim, wznowienie od przebiegu N). Raportuj postęp
  krótko: „1175 z 1200, koniec ok. 11:25”.
- Czas QPU jest drogi i ograniczony (D-Wave quota, IQM/IBM kredyty): budżetuj transzami, licz koszt przed biegiem,
  zapisuj fallbacki. Tokeny tylko z `.env`.
- Dane surowe: `articles/<slug>/measurements/` (albo `data/`), nazwy z ustawieniami
  (`algo=ils__neigh=...__file=tai20_5__inst=0__tl=10000ms__seed=0`). Do repo idą dane stojące za tabelami
  i figurami (potem snapshot w Zenodo).
- Tabele i figury **tylko ze skryptu** czytającego dane (`tables/*.tex`, `figures/*_en.pdf`, `*_pl.pdf`).
  Liczby w tekście muszą się zgadzać z tabelami – sprawdzane przed wysyłką.
- Statystyka: średnie ± odchylenie, test Wilcoxona na parach; uwaga na zależność ziaren tej samej instancji
  (zastrzeżenie w tekście). Punkty odniesienia: losowanie / p = 0, klasyczne DP, Insert, BKS lub dolne ograniczenie.
- `WYNIKI.md`: wyniki każdego etapu słownie + tabelka + co to znaczy dla tezy. Wstępne liczby oznaczaj jako wstępne.
- Raport etapu (np. `ETAP1_RAPORT.md`) z decyzjami do potwierdzenia, numerowanymi; zaproponuj wersję audio.
- Nie zmieniaj po cichu kodu użytego w opublikowanych wynikach – nowe zachowanie za flagą, domyślne bez zmian.
- Commit po każdym etapie: `<slug>: stage E2 results (...)`.

## Gdy wyniki są gotowe

Oceń z właścicielem, czy teza jest ostra (wystarczy na docelowe miejsce) – jeśli nie, zaproponuj zmianę miejsca
zawczasu. Potem `/bibliografia` i `/artykul`.
