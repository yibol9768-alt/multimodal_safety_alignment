"""Generate Fig 1 — 3-panel mechanism figure for the RDT/RDT+ paper.

Panel A: per-layer diagnostic (proj AUC at last_text vs first_action) — data from v3c/v4
Panel B: Δ_rand functional effect across α (refusal vs random zero_mass) — data from v3c
Panel C: rank-k normalized sweep (unnormalized vs √k-normalized) — data from v4

All numbers are the exact v3c / v4 numbers that appear in paper/sections/04_experiments.tex.
Keeping a single source of truth here so the figure regenerates deterministically.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35))

# -----------------------------------------------------------------------
# Panel A — per-layer projection AUC (v3c/v4 sanity_check Step 3)
# -----------------------------------------------------------------------
ax = axes[0]
layers = np.array([8, 10, 12, 14, 16, 18, 20])
auc_text  = np.array([0.603, 0.637, 0.488, 0.359, 0.292, 0.231, 0.331])
auc_act   = np.array([0.434, 0.633, 0.519, 0.485, 0.401, 0.331, 0.396])

ax.axhline(0.5, color="grey", lw=0.6, ls="--", alpha=0.7)
ax.plot(layers, auc_text, "o-", color="#d62728", lw=1.2, ms=4,
        label="last text token")
ax.plot(layers, auc_act,  "s-", color="#1f77b4", lw=1.2, ms=4,
        label="first action token")
# Highlight the chosen intervention layer
ax.axvline(10, color="black", lw=0.5, ls=":", alpha=0.5)
ax.annotate(r"$L^{\star}=10$", xy=(10, 0.68), xytext=(11.5, 0.68),
            fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))

ax.set_xlabel("layer $L$")
ax.set_ylabel("projection AUC (harm vs. benign)")
ax.set_title("(a) diagnostic: refusal axis across layers")
ax.set_ylim(0.15, 0.75)
ax.set_xticks(layers)
ax.legend(loc="upper right", frameon=False, handlelength=1.5)
ax.grid(True, lw=0.3, alpha=0.5)

# -----------------------------------------------------------------------
# Panel B — Δ_rand: refusal vs random across α (v3c Step 4 / 4b)
# -----------------------------------------------------------------------
ax = axes[1]
alphas      = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
refusal_act = np.array([0.2274, 0.2351, 0.2441, 0.2468, 0.2415])
# random values: baseline + random @ α∈{1,2,5}; use refusal α=0 as common baseline
random_act  = np.array([0.2274, np.nan, 0.2284, 0.2322, 0.2620])

ax.axhline(0.2274, color="grey", lw=0.6, ls="--", alpha=0.7,
           label="no-defense baseline")
ax.plot(alphas, refusal_act, "o-", color="#d62728", lw=1.2, ms=4,
        label="refusal direction")
# mask NaN for plotting random
mask = ~np.isnan(random_act)
ax.plot(alphas[mask], random_act[mask], "s-", color="#1f77b4", lw=1.2, ms=4,
        label="random unit vector")

# shade the effective α range
ax.axvspan(0.5, 2.0, alpha=0.08, color="green")
ax.text(1.25, 0.212, "effective\n$\\alpha \\in [0.5, 2]$",
        fontsize=7, ha="center", color="#2ca02c")

ax.set_xlabel(r"injection scale $\alpha$")
ax.set_ylabel("zero-motion mass")
ax.set_title(r"(b) $\Delta_{\mathrm{rand}}$: refusal geometry is active")
ax.set_xticks(alphas)
ax.set_ylim(0.205, 0.275)
ax.legend(loc="upper left", frameon=False, handlelength=1.8)
ax.grid(True, lw=0.3, alpha=0.5)

# -----------------------------------------------------------------------
# Panel C — rank-k: unnormalized vs √k-normalized (v3c + v4 Step 4d)
# -----------------------------------------------------------------------
ax = axes[2]
ks       = np.array([1, 3, 5, 10])
unnorm   = np.array([0.2249, 0.2110, 0.2209, 0.2455])
sqrtknorm = np.array([0.2249, 0.2328, 0.2358, 0.2291])

ax.axhline(0.2274, color="grey", lw=0.6, ls="--", alpha=0.7,
           label="no-defense baseline")
ax.plot(ks, unnorm,    "s--", color="#7f7f7f", lw=1.2, ms=4,
        label=r"unnorm. $\sum v_j$")
ax.plot(ks, sqrtknorm, "o-",  color="#d62728", lw=1.2, ms=4,
        label=r"$\sqrt{k}$-normalized")

# Highlight k=5 peak in normalized
ax.annotate(r"$k^{\star}=5$",
            xy=(5, 0.2358), xytext=(6.3, 0.242),
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.5, color="black"))

ax.set_xlabel(r"subspace rank $k$")
ax.set_ylabel("zero-motion mass")
ax.set_title(r"(c) rank-$k$: small but real multi-dim gain")
ax.set_xticks(ks)
ax.set_ylim(0.205, 0.255)
ax.legend(loc="lower right", frameon=False, handlelength=1.8)
ax.grid(True, lw=0.3, alpha=0.5)

plt.tight_layout(w_pad=1.2)
out_pdf = Path(__file__).parent / "fig1_mechanism.pdf"
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
print(f"Wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
