"""Figure for the crossing scale (plan v5): RPD difference to N_0 against F, the allowed share of the
involution move's crossings (F = 0 is the exact N_0, F = 100 the involution), mean over the classes of each n.
Writes articles/crossings/figures/cf_scale.pdf."""

import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_crossing import ROOT, inst_means
from analyze_crossing_v4 import load_any

F = list(range(0, 101, 10))
VAR = ["c0"] + [f"inv_cf{f}" for f in range(10, 100, 10)] + ["inv"]
GROUPS = {20: ["tai20_5", "tai20_10", "tai20_20"], 50: ["tai50_5", "tai50_10", "tai50_20"],
          100: ["tai100_5", "tai100_10", "tai100_20"], 200: ["tai200_10", "tai200_20"]}


def main():
    runs = load_any(["main", "safe", "cf", "cf0"])
    rpd = lambda v, f: statistics.mean(inst_means(runs, v, f, 10000))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for (n, files), mk, col in zip(GROUPS.items(), "osD^", ["C0", "C1", "C2", "C3"]):
        y = [statistics.mean(rpd(v, f) - rpd("c0", f) for f in files) for v in VAR]
        ln, = ax.plot(F, y, marker=mk, ms=3.5, lw=1.2, color=col, ls="--" if n == 200 else "-", label=f"$n={n}$")
        if n >= 100:   # control: greedy repair down to zero crossings (stage cf0)
            ax.plot([0], [statistics.mean(rpd("inv_cf0", f) - rpd("c0", f) for f in files)], marker=mk, ms=4.5,
                    mfc="none", mec=ln.get_color(), ls="none")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("allowed crossings $F$ [% of the involution move]")
    ax.set_ylabel("RPD $-$ RPD($\\mathcal{N}_0$) [pp]")
    ax.set_xticks(F[::2])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    fig.tight_layout(pad=0.3)
    out = ROOT / "articles" / "crossings" / "figures" / "cf_scale.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(out)


if __name__ == "__main__":
    main()
