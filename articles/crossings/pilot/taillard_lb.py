"""Taillard-style machine lower bound and the first Taillard seeds (ta001-ta004, ta011-ta014 reproduce the published LBs)."""
def lb(p):
    m, n = len(p), len(p[0])
    best = max(sum(p[i][j] for i in range(m)) for j in range(n))
    for i in range(m):
        head = min(sum(p[r][j] for r in range(i)) for j in range(n)) if i > 0 else 0
        tail = min(sum(p[r][j] for r in range(i + 1, m)) for j in range(n)) if i < m - 1 else 0
        best = max(best, head + sum(p[i]) + tail)
    return best
S = {"tai20_5": [873654221, 379008056, 1866992158, 216771124, 495070989, 402959317, 1369363414, 2021925980, 573109518, 88325120],
     "tai20_10": [587595453, 1401007982, 873136276, 268827376, 1634173168, 691823909, 73807235, 1273398721, 2065119309, 1672900551],
     "tai50_10": [1958948863, 575633267, 655816003, 1977864101, 93805469, 1803345551, 49612559, 1899802599, 2013025619, 578962478]}
DIMS = {"tai20_5": (20, 5), "tai20_10": (20, 10), "tai50_10": (50, 10)}
if __name__ == "__main__":
    from pilot_anatomy import lcg_instance
    for f, seeds in S.items():
        n, m = DIMS[f]
        print(f, [lb(lcg_instance(s, n, m)) for s in seeds])
