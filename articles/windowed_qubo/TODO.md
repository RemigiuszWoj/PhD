# TODO — windowed_qubo (Computers & Operations Research)

Stan: 2026-07-13. Tabele rpd_n20 i rpd_scaling w 100% wypełnione realnymi
danymi QPU (kampanie P1/P2/T3 lipiec + P4 11–13.07). Wilcoxon policzony
i wpisany. Wnioski/abstract/highlights zaktualizowane o wyniki.

## Przed wysyłką do promotora
- [ ] /humanizer pass na całym tekście (stały wymóg przed finalizacją)
- [ ] Ostatnie czytanie PDF strona po stronie (formatowanie, overfull boxy,
      podpisy figur) — formatowanie robimy NA KOŃCU

## Przed submisją (decyzje podjęte, pilnować spójności)
- [ ] BTJL: ablacja vs random restart POMINIĘTA (decyzja 13.07 — lista jest
      zaimplementowana i używana, nie robimy osobnego pomiaru).
      Tekst już złagodzony: "ablation ... left for future work".
      NIE wpisywać z powrotem twierdzeń o porównaniu bez danych.
- [ ] Siatka 6 budżetów dla quantum: tabele jej NIE wymagają; Setup
      doprecyzowany (klasyczne 6 budżetów, QPU: tabele tl=1000/5000,
      konwergencja 100–5000 enh). Opcjonalne doliczenie warstw
      tl=100/500/2000/10000 (P1 orig 1525 runów + P2 enh 914 runów,
      ~170 s quoty = 4–5 transz, ~30–40 h wall) TYLKO jeśli recenzent
      poprosi o pełne krzywe konwergencji dla klasy oryginalnej.

## Badania na przyszłość (sierpniowa quota lub później)
- [ ] Czułość na overlap ρ: dyn/motz enhanced, ρ ∈ {0.25, 0.75}
      (ρ=0.5 zmierzone), n=20 i n=50, ILS tl=5000 → ~400 runów,
      ~25–30 s quoty (1 transza), ~1 noc wall. Kod: przepuścić
      overlap_ratio przez config/runner (parametr już istnieje
      w funkcjach okienkowania). Po pomiarze wymienić zdanie
      "sensitivity to ρ remains untested".
- [ ] Ewentualna ablacja BTJL, jeśli wróci w recenzji: flaga
      diversification=mushroom|random_restart w ils.py (~20 linii),
      600 runów klasycznych (~3 h CPU, zero quoty), Wilcoxon par.
- [ ] Future work z artykułu: adaptive window sizing, QAOA/Willow,
      cross-window interactions, hybrid flow shop.

## Administracyjne przy submisji
- [ ] Data availability: upublicznić repo kodu i wyników (obiecane
      "upon acceptance")
- [ ] Sprawdzić dane afiliacji/ORCID w elsarticle
- [ ] Highlights: limit Elsevier 85 znaków/bullet — zweryfikować długości

## Zasoby / fakty
- Budżet QPU lipca: zużyte 194.5 s / ~400 s (P1 40 + P2 40 + T3 40 +
  P4T1 40 + P4T2 34.5); reszta ~205 s.
- Katalogi wyników: 20260703_083952 (P1 orig m=5),
  20260707_203922 (P2+T3 enh m=5), 20260711_094938 (P4 scaling m=10/20).
- Skrypt kampanii: scripts/run_qpu_p4.py (--resume, --limit).
