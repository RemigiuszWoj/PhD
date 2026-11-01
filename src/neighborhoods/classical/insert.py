"""Insert neighborhood - best single insertion (reference neighborhood).

Move: remove the job at position i and re-insert it at position t of the
remaining sequence ((n-1)^2 distinct neighbors). Classically the strongest
single-move neighborhood for the PFSP (Taillard 1990). Used in the k-bonacci
paper only as a reference row; it is not part of the previous papers.

Evaluation: Taillard's acceleration. For each removed job the heads e and
tails q of the remaining sequence are computed once (O(mn)); the makespan of
inserting the job before position t is then max_r (f[r][t] + q[r][t]) with
f[r][t] = max(f[r-1][t], e[r][t-1]) + p[r][job], i.e. O(m) per position.
Total O(m n^2) per neighborhood evaluation.

Like Adjacent, the best neighbor is returned even if it does not improve
(the metaheuristic decides acceptance).
"""

from typing import List, Tuple

from src.permutation_procesing import c_max


def _heads_tails(seq: List[int], pt: List[List[int]]):
    m, L = len(pt), len(seq)
    e = [[0] * L for _ in range(m)]
    q = [[0] * L for _ in range(m)]
    for r in range(m):
        row, prev_row = pt[r], e[r - 1] if r else None
        er = e[r]
        for t in range(L):
            left = er[t - 1] if t else 0
            up = prev_row[t] if prev_row is not None else 0
            er[t] = (left if left > up else up) + row[seq[t]]
    for r in range(m - 1, -1, -1):
        row, next_row = pt[r], q[r + 1] if r < m - 1 else None
        qr = q[r]
        for t in range(L - 1, -1, -1):
            right = qr[t + 1] if t < L - 1 else 0
            down = next_row[t] if next_row is not None else 0
            qr[t] = (right if right > down else down) + row[seq[t]]
    return e, q


def insert_neighborhood(
    pi: List[int], processing_times: List[List[int]]
) -> Tuple[List[int], int, Tuple[int, int, int]]:
    """Best insertion move. Returns (new_pi, new_cmax, move_attribute=(job, min(i,t), max(i,t)))."""
    n, m = len(pi), len(processing_times)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (pi[0] if pi else -1, 0, 0)
    best_c, best_move = None, (0, 0)
    for i in range(n):
        job = pi[i]
        seq = pi[:i] + pi[i + 1 :]
        e, q = _heads_tails(seq, processing_times)
        L = n - 1
        for t in range(L + 1):
            if t == i:  # re-inserting at the same place = identity
                continue
            f_prev = 0
            cmax = 0
            for r in range(m):
                left = e[r][t - 1] if t else 0
                f = (f_prev if f_prev > left else left) + processing_times[r][job]
                tail = q[r][t] if t < L else 0
                if f + tail > cmax:
                    cmax = f + tail
                f_prev = f
            if best_c is None or cmax < best_c:
                best_c, best_move = cmax, (i, t)
    i, t = best_move
    seq = pi[:i] + pi[i + 1 :]
    new_pi = seq[:t] + [pi[i]] + seq[t:]
    # Tabu attribute: (job, {i, t}). The reverse move takes the same job from
    # position t back to position i, so it carries the same attribute and the
    # tabu list blocks the immediate undo, as (i, i+1) does for Adjacent and the
    # swap set does for the k-bonacci family. With (i, t) the undo would be (t, i),
    # never tabu, and ILS could cycle between two solutions.
    return new_pi, best_c, (pi[i], min(i, t), max(i, t))
