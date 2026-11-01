"""Large-n counts of the crossing family: ratios, coverage, Poisson law, per-arc growth constant."""
import math, sys
from counts import jfrac_counts, involution_numbers, motzkin

N = 500; C = 6
J = jfrac_counts(N, C)
I = involution_numbers(N); M = motzkin(N)

def le(n, c): return sum(J[n][:c+1])

def sci(x):
    if x == 0: return "0"
    e = math.floor(math.log10(x)); m = x / 10**e
    return f"{m:.1f}e{e}"

print("== |N_c(n)| (<= c crossings) and ratio to Motzkin ==")
for n in [10, 20, 50, 100, 200, 500]:
    row = [f"n={n}"]
    for c in range(0, 5):
        row.append(f"c<={c}: {sci(le(n,c))} (x{le(n,c)/M[n]:.3g})")
    print("  ".join(row))
print()
print("== growth check: ratio_c(n) = |N_c|/M_n vs n^(3c/2); show ratio/n^(3c/2) ==")
for c in range(1, 5):
    vals = []
    for n in [50, 100, 200, 400, 500]:
        r = le(n, c) / M[n]
        vals.append(f"n={n}: {r / n**(1.5*c):.4g}")
    print(f"c={c}: " + ", ".join(vals))
print()
print("== exact-c share: J[n][c]/M_n / n^(3c/2) ==")
for c in range(1, 5):
    vals = []
    for n in [100, 200, 400, 500]:
        vals.append(f"{J[n][c]/M[n]/n**(1.5*c):.4g}")
    print(f"c={c}: " + ", ".join(vals))
print()
print("== coverage of involutions |N_c|/I(n) ==")
for n in [10, 12, 14, 16, 20, 30, 50, 100]:
    print(f"n={n}: " + ", ".join(f"c<={c}: {le(n,c)/I[n]:.3g}" for c in range(0, 5)) + f"  I(n)={sci(I[n])}  M_n={sci(M[n])}")
print()
# log growth of involutions
for n in [20, 50, 100, 200, 500]:
    print(f"n={n}: log10 I(n)={math.log10(I[n]):.1f}, log10 M(n)={math.log10(M[n]):.1f}, log10 2^(n-1)={(n-1)*math.log10(2):.1f}")

# per-arc <=1 GF: S = 1 + x S + x^2 S^2 + x^4 S^4
def perarc(Nmax):
    S = [1] + [0]*Nmax
    for it in range(Nmax+1):
        # compute powers
        def mul(a, b):
            r = [0]*(Nmax+1)
            for i, ai in enumerate(a):
                if ai:
                    for j in range(Nmax+1-i):
                        r[i+j] += ai*b[j]
            return r
        S2 = mul(S, S); S4 = mul(S2, S2)
        new = [0]*(Nmax+1); new[0] = 1
        for n in range(1, Nmax+1):
            new[n] = S[n-1] + (S2[n-2] if n >= 2 else 0) + (S4[n-4] if n >= 4 else 0)
        if new == S: break
        S = new
    return S
P = perarc(60)
print("per-arc<=1 via S=1+xS+x^2S^2+x^4S^4:", P[:12])
for n in [20, 40, 60]:
    print(f"  n={n}: perarc={sci(P[n])} vs c<=1 {sci(le(n,1))}, M={sci(M[n])}, ratio^(1/n) trend: {(P[n]/P[n-1]):.4f}")
# random involution: expected arcs and crossings (exact via J-fraction derivative is heavy) -> Monte Carlo
import random
def rand_inv(n, rng):
    # sample uniform involution via recursion I(n)=I(n-1)+(n-1)I(n-2)
    free = list(range(n)); arcs = []
    while free:
        k = len(free)
        a = free.pop(0)
        if rng.random() < I[k-1]/I[k]:
            continue
        idx = rng.randrange(len(free)); b = free.pop(idx); arcs.append((a, b))
    return arcs
rng = random.Random(1)
for n in [20, 50, 100]:
    tot_cr = 0; tot_a = 0; T = 300
    for _ in range(T):
        arcs = rand_inv(n, rng); tot_a += len(arcs)
        arcs.sort()
        cr = 0
        for i in range(len(arcs)):
            a, b = arcs[i]
            for j in range(i+1, len(arcs)):
                c, d = arcs[j]
                if c < b < d: cr += 1
        tot_cr += cr
    print(f"random involution n={n}: mean arcs={tot_a/T:.1f} (n/2-sqrt(n)/2={n/2-math.sqrt(n)/2:.1f}), mean crossings={tot_cr/T:.1f} (C(k,2)/3 with k=mean arcs: {math.comb(round(tot_a/T),2)/3:.1f})")

# ---- Poisson law check and per-arc growth constant ----
N = 2000; C = 5
J = jfrac_counts(N, C); M = motzkin(N)
print("exact-c / M_n divided by (n/3)^c / c!   (-> 1 ?)")
for c in range(1, C + 1):
    print(f"c={c}: " + "  ".join(f"n={n}: {J[n][c] / M[n] / ((n/3)**c / math.factorial(c)):.4f}" for n in (50, 100, 200, 500, 1000, 2000)))
print("<=c / M_n divided by (n/3)^c / c!")
for c in range(1, C + 1):
    print(f"c={c}: " + "  ".join(f"n={n}: {sum(J[n][:c+1]) / M[n] / ((n/3)**c / math.factorial(c)):.4f}" for n in (50, 100, 200, 500, 1000, 2000)))
# per-arc<=1 growth: singularity of S = 1 + xS + x^2 S^2 + x^4 S^4
def F(S, x): return 1 + x*S + x*x*S*S + x**4 * S**4 - S
def FS(S, x): return x + 2*x*x*S + 4*x**4*S**3 - 1
# Newton on the system
S, x = 2.0, 0.28
for _ in range(100):
    f1, f2 = F(S, x), FS(S, x)
    # partials
    Fx = S + 2*x*S*S + 4*x**3*S**4
    FSS = 2*x*x + 12*x**4*S*S
    FSx = 1 + 4*x*S + 16*x**3*S**3
    det = FS(S, x)*FSx - Fx*FSS  # Jacobian [[FS, Fx],[FSS, FSx]]
    dS = (f1*FSx - Fx*f2) / det; dx = (FS(S, x)*f2 - FSS*f1) / det
    S -= dS; x -= dx
print(f"per-arc<=1: singularity x={x:.6f}, growth 1/x={1/x:.4f}, S={S:.4f}")
