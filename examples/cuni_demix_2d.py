"""
The payoff, made visible (2D illustration): what the retrained potential does.

A clean 2D regular-solution Monte-Carlo demixing driven by MACE's own Cu-Ni
mixing interaction. At 270 K --- above MACE-MP-0's (wrong) consolute, below the
tuned (correct) one --- conserved-composition (Kawasaki) MC on a square lattice
with H = (Omega/z) * (unlike nearest-neighbour bonds). MACE-MP-0's Omega (42
meV) keeps the alloy a random solid solution; the tuned Omega (108 meV)
phase-separates into Cu-rich and Ni-rich domains. Same temperature, same lattice;
only the retrained weights differ. (A 2D lattice for a clean microstructure read;
the thermodynamics is the same regular-solution model used for the Cu-Ni gap, and
mean-field T_c = Omega/2k_B is lattice-independent.)

Run:  python examples/cuni_demix_2d.py
Out:  examples/output/cuni_demix_2d.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
K_B = 8.617333e-5
Z = 4                       # square-lattice nearest neighbours

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=120)       # lattice side
ap.add_argument("--T", type=float, default=270.0)
ap.add_argument("--sweeps", type=int, default=3000)
ap.add_argument("--om-pre", type=float, default=42.4)
ap.add_argument("--om-tuned", type=float, default=107.6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
N = args.N
Tc = lambda om: om / 1e3 / (2 * K_B)


def run_mc(omega_meV):
    w = (omega_meV / 1e3) / Z
    s = np.zeros((N, N), dtype=np.int8); s[:, N // 2:] = 1   # prepared Cu|Ni interface
    beta = 1.0 / (K_B * args.T)
    for _ in range(args.sweeps * N * N):
        i, j = rng.integers(N), rng.integers(N)
        nbrs = (((i+1) % N, j), ((i-1) % N, j), (i, (j+1) % N), (i, (j-1) % N))
        i2, j2 = nbrs[rng.integers(4)]
        if s[i, j] == s[i2, j2]:
            continue
        def unlike(a, b, ex):          # unlike bonds of site (a,b) if its value were `ex`
            return (int(s[(a+1) % N, b] != ex) + int(s[(a-1) % N, b] != ex)
                    + int(s[a, (b+1) % N] != ex) + int(s[a, (b-1) % N] != ex))
        old = unlike(i, j, s[i, j]) + unlike(i2, j2, s[i2, j2])
        new = unlike(i, j, s[i2, j2]) + unlike(i2, j2, s[i, j])
        dE = w * (new - old)
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            s[i, j], s[i2, j2] = s[i2, j2], s[i, j]
    nb = ((s != np.roll(s, 1, 0)).sum() + (s != np.roll(s, 1, 1)).sum()) / (2 * N * N)
    return s, nb


print(f"2D square lattice {N}x{N}, {args.sweeps} sweeps, T={args.T:.0f} K")
print("running MACE-MP-0 (pretrained Omega)…"); s_pre, u_pre = run_mc(args.om_pre)
print("running tuned Omega…");                  s_tun, u_tun = run_mc(args.om_tuned)
print(f"  pretrained Omega {args.om_pre:.0f} meV (T_c {Tc(args.om_pre):.0f} K): unlike {u_pre:.3f}")
print(f"  tuned      Omega {args.om_tuned:.0f} meV (T_c {Tc(args.om_tuned):.0f} K): unlike {u_tun:.3f}")

cmap = ListedColormap(["#d98032", "#9aa7b2"])      # Cu orange, Ni silver
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4))
for ax, s, u, om, lab in [(axes[0], s_pre, u_pre, args.om_pre, "MACE-MP-0 (pretrained)"),
                          (axes[1], s_tun, u_tun, args.om_tuned, "tuned")]:
    ax.imshow(s, cmap=cmap, interpolation="nearest")
    ax.set_title(f"{lab}:  $\\Omega$={om:.0f} meV, $T_c$={Tc(om):.0f} K\n"
                 f"unlike-bond fraction {u:.2f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color="#d98032", label="Cu"), Patch(color="#9aa7b2", label="Ni")],
               loc="upper right", fontsize=9, framealpha=0.9)
fig.suptitle(f"A prepared Cu$|$Ni interface at {args.T:.0f} K: it dissolves under "
             f"MACE-MP-0, persists under the retrained potential", fontsize=12)
fig.tight_layout()
out = OUT / "cuni_demix_2d.png"; fig.savefig(out, dpi=150)
json.dump({"T": args.T, "om_pre": args.om_pre, "om_tuned": args.om_tuned,
           "unlike_pre": u_pre, "unlike_tuned": u_tun}, open(OUT / "cuni_demix_2d.json", "w"), indent=1)
print(f"\nwrote {out}")
