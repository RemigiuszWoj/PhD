---
name: przeglad-krytyczny
description: Rundy krytycznego przeglądu artykułu doktoratu przed oddaniem promotorowi i przed wysyłką – niezależni krytycy (matematyka, wyniki od zera z danych, tezy vs dowody, recenzent docelowej konferencji, PL vs EN), humanizer, formatowanie, runda promotora. Użyj, gdy szkic jest gotowy albo właściciel prosi o „rundę krytyka”, „przegląd”, „humanizer”.
---

# Przegląd krytyczny

Wzorzec: ANNSIM 5 rund, ACIIDS 10 rund, ICAISC 2027 8 rund, k-bonacci 5 rund + przeglądy na najmocniejszym modelu.

**Baza wiedzy:** listę kontrolną `~/Desktop/Doktorat/PhD-wiki/proces/uwagi-promotora.md` i zasady `proces/styl-pisania.md` daj krytykom
stylu. Istotne zarzuty, które zostają w tekście jako ograniczenia, dopisz do strony artykułu w `publikacje/`.

## Jedna runda

Uruchom równolegle niezależnych krytyków (subagenci, każdy dostaje ścieżki plików i jedno zadanie, raport
z numerowanymi uwagami: miejsce, problem, propozycja poprawki, waga):

1. **Matematyka** – wzór po wzorze, definicje, twierdzenia, kontrprzykłady, sprawdzenie na małych przypadkach.
2. **Wyniki od zera** – przelicz liczby z `measurements/` i tabel, także innymi miarami; czy tekst mówi to,
   co pokazują dane; czy statystyka uprawnia do wniosku.
3. **Recenzent docelowego miejsca** – co zaatakuje (nowość, porównanie ze stanem wiedzy, dopasowanie do
   konferencji, wielkość części kwantowej), czego brakuje do akceptacji.
4. **Styl i forma** – zasady z `/artykul` (proza, nagłówki, pełne zdania, zakazane sformułowania, akronimy,
   spójność pisowni, limit stron).
5. **PL vs EN** – braki treści, tabele tylko w jednej wersji, bibliografia, kalki (tańszy model wystarczy).

Merytoryczne przeglądy (1–3) na najmocniejszym modelu, 4–5 na tańszym. Potem:
- Zbierz uwagi, odrzuć błędne (z jednym zdaniem dlaczego), resztę nanieś w EN i PL.
- Krótko właścicielowi: co zmieniło tezy, co zostało otwarte, co wymaga decyzji. Bez wklejania całego tekstu.
- Commit po rundzie: `<slug>: critical review round 3 (EN/PL)`.
- Kończ, gdy runda nie przynosi uwag o wadze wyższej niż kosmetyczna.

## Humanizer (po merytoryce)

Przejście językowe: usuń szablonowe frazy („delve”, „notably”, „it is worth noting”, „crucial”), nadmiar
myślników i średników, powtarzalne początki zdań, wypunktowania w prozie. Zachowaj terminologię i liczby.
Commit: `<slug>: humanizer pass (EN/PL)`.

## Formatowanie (na samym końcu)

Floaty na właściwych stronach, przepełnione linie, wdowy, kolejność tabel, legenda ≥ 6 pt, liczba stron,
kompilacja bez ostrzeżeń o brakujących odwołaniach.

## Runda promotora

PDF + Overleaf. Uwagi promotora spisz jako listę z rozstrzygnięciem każdej; zmiany z Overleafa osobnym commitem.
Typowe uwagi z przeszłości: prostszy, opisowy abstrakt; mniej wypunktowań; podpisy 2-liniowe; wyeksponować
fizyczny sprzęt; łagodniejsze oświadczenie o AI; średnia ± odchylenie w tabelach.

## Kontrola nakładania z naszymi pracami

Przed wysyłką: brak wspólnych fraz 8-wyrazowych z poprzednimi artykułami (poza tytułem i oświadczeniem o AI);
iThenticate < 20% tam, gdzie konferencja go wymaga.
