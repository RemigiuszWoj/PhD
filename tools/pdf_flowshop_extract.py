"""Extract keyword occurrences (page numbers + line snippets) from Smutnicki PDF.

Keywords tuned for Flow Shop context.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, List

try:
    import PyPDF2  # type: ignore
except ImportError:  # pragma: no cover
    print("PyPDF2 not installed. pip install PyPDF2", file=sys.stderr)
    sys.exit(1)

PDF_PATH = pathlib.Path(__file__).resolve().parent.parent / "docs" / "smutnicki_algorytmy.pdf"
KEYWORDS = [
    "flow shop",
    "permutacyjny",
    "permutation",
    "NEH",
    "tabu",
    "sąsiedzt",  # prefix: sąsiedztwo
    "bli\u017csz",  # prefix for Polish stems like 'bliższe'
]


def normalize(txt: str) -> str:
    return txt.lower()


def extract(pdf_path: pathlib.Path, keywords: List[str]):
    reader = PyPDF2.PdfReader(str(pdf_path))
    hits: Dict[str, List[Dict]] = {k: [] for k in keywords}
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            continue
        low = normalize(raw)
        for kw in keywords:
            if kw in low:
                # capture lines containing kw
                lines = [ln for ln in raw.splitlines() if kw in ln.lower()]
                snippet = " | ".join(line_text.strip()[:160] for line_text in lines[:5])
                hits[kw].append({"page": i + 1, "snippet": snippet})
    return hits


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        sys.exit(1)
    hits = extract(PDF_PATH, KEYWORDS)
    for kw, lst in hits.items():
        if not lst:
            continue
        print(f"=== {kw} ===")
        pages = ", ".join(str(x["page"]) for x in lst)
        print(f"Pages: {pages}")
        for rec in lst[:5]:
            print(f"  p{rec['page']}: {rec['snippet']}")
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
