#!/usr/bin/env python3
"""Wspólna pula bibliografii doktoratu.

Trzy polecenia (tylko biblioteka standardowa):

  build    zbiera wszystkie articles/**/*.bib (i pliki podane w --extra, np. eksport
           z cytownika) do bibliography/pula.bib; duplikaty po kluczu i po DOI,
           konflikty wypisuje zamiast zgadywać.
  check    sprawdza DOI z pliku .bib w Crossref (a gdy brak, w DataCite) i porównuje
           tytuł; wypisuje rozjazdy. Wymaga sieci (api.crossref.org, api.datacite.org).
  extract  bierze klucze z \\cite{...} w pliku .tex i zapisuje podzbiór puli do
           references.bib artykułu (z kolejnością jak w puli).

Przykłady:
  python tools/bib_pool.py build
  python tools/bib_pool.py build --extra ~/Documents/cytownik.bib
  python tools/bib_pool.py check bibliography/pula.bib
  python tools/bib_pool.py extract articles/nowy/main.tex -o articles/nowy/references.bib
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POOL = REPO / "bibliography" / "pula.bib"


def parse_bib(text: str) -> list[dict]:
    """Prosty parser: dzieli plik na wpisy @typ{klucz, ...} liczac nawiasy."""
    entries = []
    i = 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.S).search(text, i)
        if not m:
            break
        etype, key = m.group(1).lower(), m.group(2)
        if etype in ("comment", "preamble", "string"):
            i = m.end()
            continue
        depth, j = 1, text.index("{", m.start()) + 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        raw = text[m.start():j]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", raw):
            name = fm.group(1).lower()
            k = fm.end()
            if k < len(raw) and raw[k] == "{":
                d, s = 1, k + 1
                k += 1
                while k < len(raw) and d:
                    d += {"{": 1, "}": -1}.get(raw[k], 0)
                    k += 1
                fields.setdefault(name, raw[s:k - 1])
            elif k < len(raw) and raw[k] == '"':
                e = raw.index('"', k + 1)
                fields.setdefault(name, raw[k + 1:e])
            else:
                e = re.compile(r"[,}\n]").search(raw, k)
                fields.setdefault(name, raw[k:e.start() if e else len(raw)].strip())
        entries.append({"type": etype, "key": key, "raw": raw.strip(), "fields": fields})
        i = j
    return entries


def norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.replace("\\_", "_")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\.|[{}]", "", t.lower()))


def cmd_build(args) -> int:
    sources = sorted((REPO / "articles").rglob("*.bib"))
    sources = [p for p in sources if "submitted_v" not in str(p) and "_szablon" not in str(p)]
    sources += [Path(p).expanduser() for p in (args.extra or [])]
    by_key: dict[str, dict] = {}
    by_doi: dict[str, str] = {}
    conflicts, aliases = [], []
    for src in sources:
        for e in parse_bib(src.read_text(encoding="utf-8", errors="replace")):
            e["src"] = str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)
            doi = norm_doi(e["fields"].get("doi"))
            if e["key"] in by_key:
                old = by_key[e["key"]]
                if norm_doi(old["fields"].get("doi")) != doi or \
                        norm_title(old["fields"].get("title", "")) != norm_title(e["fields"].get("title", "")):
                    conflicts.append(f"klucz {e['key']}: {old['src']} vs {e['src']}")
                continue
            if doi and doi in by_doi:
                # ta sama praca pod innym kluczem: zostaje (artykuly sie do niego odwoluja),
                # ale nowe artykuly maja uzywac pierwszego klucza
                aliases.append(f"{e['key']} = {by_doi[doi]} (DOI {doi})")
            by_key[e["key"]] = e
            if doi:
                by_doi.setdefault(doi, e["key"])
    POOL.parent.mkdir(exist_ok=True)
    out = [
        "% Wspolna pula bibliografii doktoratu. NIE edytowac recznie kluczy istniejacych wpisow,",
        "% bo artykuly sie do nich odwoluja. Odtwarzana: python tools/bib_pool.py build",
        f"% Zrodla: {len(sources)} plikow, {len(by_key)} wpisow.",
        "",
    ]
    for e in sorted(by_key.values(), key=lambda x: x["key"].lower()):
        out += [f"% z: {e['src']}", e["raw"], ""]
    POOL.write_text("\n".join(out), encoding="utf-8")
    print(f"Zapisano {POOL.relative_to(REPO)}: {len(by_key)} wpisow z {len(sources)} plikow.")
    no_doi = [k for k, e in by_key.items() if not e["fields"].get("doi")]
    if no_doi:
        print(f"Bez DOI ({len(no_doi)}): {', '.join(sorted(no_doi))}")
    if aliases:
        print(f"\nALIASY ({len(aliases)}) – ta sama praca pod dwoma kluczami; w nowych tekstach uzywaj drugiego:")
        for a in aliases:
            print("  -", a)
    if conflicts:
        print(f"\nKONFLIKTY ({len(conflicts)}) – zostawiony pierwszy wpis, sprawdz recznie:")
        for c in conflicts:
            print("  -", c)
    return 0


def fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "PhD-bib-check/1.0 (mailto:remigiusz.wojewodzki@pwr.edu.pl)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def cmd_check(args) -> int:
    entries = parse_bib(Path(args.bib).read_text(encoding="utf-8"))
    bad = 0
    for e in entries:
        doi = norm_doi(e["fields"].get("doi"))
        if not doi:
            print(f"[BRAK DOI] {e['key']}")
            continue
        q = urllib.parse.quote(doi)
        title, where = None, None
        d = fetch_json(f"https://api.crossref.org/works/{q}")
        if d and d.get("message", {}).get("title"):
            title, where = d["message"]["title"][0], "Crossref"
        else:
            d = fetch_json(f"https://api.datacite.org/dois/{q}")
            if d and d.get("data"):
                ts = d["data"]["attributes"].get("titles") or [{}]
                title, where = ts[0].get("title"), "DataCite"
        if not title:
            print(f"[DOI NIE ISTNIEJE] {e['key']}: {doi}")
            bad += 1
        elif norm_title(title)[:40] != norm_title(e["fields"].get("title", ""))[:40]:
            print(f"[INNY TYTUL] {e['key']} ({where}):\n    bib:  {e['fields'].get('title')}\n    {where}: {title}")
            bad += 1
        else:
            print(f"[OK] {e['key']} ({where})")
        time.sleep(0.2)
    print(f"\n{len(entries)} wpisow, problemow: {bad}")
    return 1 if bad else 0


def cmd_extract(args) -> int:
    tex = Path(args.tex).read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    keys = []
    for m in re.finditer(r"\\(?:no)?cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    pool = {e["key"]: e for e in parse_bib(POOL.read_text(encoding="utf-8"))}
    missing = [k for k in keys if k not in pool]
    found = [pool[k] for k in keys if k in pool]
    Path(args.output).write_text(
        f"% Wyciag z bibliography/pula.bib dla {args.tex} ({len(found)} pozycji).\n\n"
        + "\n\n".join(e["raw"] for e in found) + "\n", encoding="utf-8")
    print(f"Zapisano {args.output}: {len(found)} pozycji.")
    if missing:
        print(f"BRAK W PULI ({len(missing)}): {', '.join(missing)} – dodaj je do references.bib artykulu "
              "i po weryfikacji DOI przebuduj pule (build).")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--extra", nargs="*", help="dodatkowe pliki .bib (np. eksport z cytownika)")
    c = sub.add_parser("check")
    c.add_argument("bib")
    x = sub.add_parser("extract")
    x.add_argument("tex")
    x.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    return {"build": cmd_build, "check": cmd_check, "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
