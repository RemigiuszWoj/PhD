"""Check that data/tai*.txt are the genuine Taillard PFSP instances (regenerated from their seeds).

Taillard (1993) generates p_ij with a Lehmer LCG (a = 16807, m = 2^31 - 1) from the
seed printed in each instance header. All 120 instances must match exactly.

Usage: python scripts/verify_taillard.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.parser import parser  # noqa: E402

FILES = ["tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20", "tai100_5",
         "tai100_10", "tai100_20", "tai200_10", "tai200_20", "tai500_20"]


def generate(seed: int, n: int, m: int):
    a, b, c, M = 16807, 127773, 2836, 2**31 - 1
    s, out = seed, [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            k = s // b
            s = a * (s % b) - k * c
            if s < 0:
                s += M
            out[i][j] = 1 + int(s / M * 99)
    return out


def main() -> None:
    present = {os.path.basename(p)[:-4] for p in glob.glob("data/tai*.txt")}
    missing = [f for f in FILES if f not in present]
    bad = 0
    for f in FILES:
        if f in missing:
            continue
        for t in range(10):
            d = parser(f"data/{f}.txt", t)
            i = d["info"]
            if generate(i["seed"], i["jobs"], i["machines"]) != d["processing_times"]:
                bad += 1
                print(f"MISMATCH {f} instance {t}")
    print(f"files present: {len(FILES) - len(missing)}/12, missing: {missing or 'none'}, mismatches: {bad}")
    sys.exit(1 if (missing or bad) else 0)


if __name__ == "__main__":
    main()
