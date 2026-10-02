"""
The payoff, made visible: what the tuned potential DOES that MACE-MP-0 cannot.

We changed the weights so MACE's Cu-Ni mixing interaction went Omega 42 -> 108
meV/atom, lifting the consolute temperature 246 -> 624 K onto the assessed value.
Here is what that buys in a simulation. At 450 K --- above MACE-MP-0's (wrong)
consolute, below the tuned (correct) one --- we run conserved-composition
(Kawasaki) Monte Carlo on an fcc lattice with the regular-solution Hamiltonian
H = (Omega/z) * (number of unlike nearest-neighbour bonds), using each
potential's own Omega. MACE-MP-0 keeps the alloy a random solid solution; the
tuned potential phase-separates into Cu-rich and Ni-rich domains --- the
experimentally correct behaviour. Same temperature, same lattice; the only
difference is the weights we retrained.

Run:  python examples/cuni_demix_mc.py
Out:  examples/output/cuni_demix_mc.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
K_B = 8.617333e-5
Z = 12                     # fcc nearest neighbours

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=12)        # conventional fcc cells per side
ap.add_argument("--T", type=float, default=450.0)   # K: between the two consolutes
ap.add_argument("--sweeps", type=int, default=4000)
ap.add_argument("--om-pre", type=float, default=42.4)    # meV, MACE-MP-0 Cu-Ni Omega
ap.add_argument("--om-tuned", type=float, default=107.6) # meV, tuned
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# ---- fcc lattice sites + nearest-neighbour table (periodic) ------------------
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
cells = np.array([[i, j, k] for i in range(args.L) for j in range(args.L) for k in range(args.L)])
pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)   # fractional, units of a
N = len(pos)
L = args.L
# 12 fcc NN offsets (in units of a): permutations of (±0.5,±0.5,0)
offs = []
for a in (0.5, -0.5):
    for b in (0.5, -0.5):
        offs += [[a, b, 0], [a, 0, b], [0, a, b]]
offs = np.unique(np.array(offs), axis=0)
# hash site positions to indices for O(1) neighbour lookup (periodic)
key = lambda p: tuple(np.round((p % L) * 2).astype(int))
index = {key(p): i for i, p in enumerate(pos)}
nbr = np.empty((N, Z), dtype=np.int64)
for i, p in enumerate(pos):
    for m, o in enumerate(offs):
        nbr[i, m] = index[key(p + o)]
print(f"fcc lattice: {N} sites, {Z} neighbours each (L={L}, {args.sweeps} sweeps, T={args.T:.0f} K)")


def run_mc(omega_meV):
    """Kawasaki (swap) MC; returns the final occupancy (0=Cu,1=Ni) and unlike-bond fraction."""
    w = (omega_meV / 1e3) / Z                          # eV per unlike bond
    s = np.zeros(N, dtype=np.int8); s[rng.permutation(N)[: N // 2]] = 1   # 50/50
    beta = 1.0 / (K_B * args.T)
    for sweep in range(args.sweeps):
        for _ in range(N):
            i = rng.integers(N); j = nbr[i, rng.integers(Z)]
            if s[i] == s[j]:
                continue
            # dE of swapping i<->j = w * (new unlike bonds - old), over their neighbours
            ni = s[nbr[i]]; nj = s[nbr[j]]
            old = np.count_nonzero(ni != s[i]) + np.count_nonzero(nj != s[j])
            new = np.count_nonzero(ni != s[j]) + np.count_nonzero(nj != s[i])
            dE = w * (new - old)
            if dE <= 0 or rng.random() < np.exp(-beta * dE):
                s[i], s[j] = s[j], s[i]
    unlike = sum(np.count_nonzero(s[nbr[i]] != s[i]) for i in range(N)) / (N * Z)
    return s, unlike


print("running MACE-MP-0 (pretrained Omega)…")
s_pre, u_pre = run_mc(args.om_pre)
print("running tuned Omega…")
s_tun, u_tun = run_mc(args.om_tuned)
Tc = lambda om: om / 1e3 / (2 * K_B)
print(f"  pretrained: Omega {args.om_pre:.0f} meV (T_c {Tc(args.om_pre):.0f} K), "
      f"unlike-bond fraction {u_pre:.3f}")
print(f"  tuned:      Omega {args.om_tuned:.0f} meV (T_c {Tc(args.om_tuned):.0f} K), "
      f"unlike-bond fraction {u_tun:.3f}  (lower = demixed)")

np.save(OUT / "cuni_demix_cfg.npy", {"pos": pos, "L": L, "s_pre": s_pre,
        "s_tun": s_tun, "u_pre": u_pre, "u_tun": u_tun, "args": vars(args)},
        allow_pickle=True)


# ---- render: 3D block-averaged Ni fraction, mid-z slab (domains read cleanly)-
def slab_map(s, nb=8):
    """Average onto nb^3 blocks; return the mid-z slab (nb x nb)."""
    b = np.clip((pos / L * nb).astype(int), 0, nb - 1)
    tot = np.zeros((nb, nb, nb)); cnt = np.zeros((nb, nb, nb))
    np.add.at(tot, (b[:, 1], b[:, 0], b[:, 2]), s)
    np.add.at(cnt, (b[:, 1], b[:, 0], b[:, 2]), 1.0)
    comp = tot / np.maximum(cnt, 1)
    return comp[:, :, nb // 2]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
for ax, s, u, om, lab in [(axes[0], s_pre, u_pre, args.om_pre, "MACE-MP-0 (pretrained)"),
                          (axes[1], s_tun, u_tun, args.om_tuned, "tuned")]:
    im = ax.imshow(slab_map(s), origin="lower", cmap="RdBu", vmin=0.25, vmax=0.75,
                   interpolation="bilinear")
    ax.set_title(f"{lab}:  $\\Omega$={om:.0f} meV, $T_c$={Tc(om):.0f} K\n"
                 f"unlike-bond fraction {u:.2f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("local Ni fraction  (red = Cu-rich, blue = Ni-rich)", fontsize=8)
fig.suptitle(f"Cu-Ni at {args.T:.0f} K: the retrained potential phase-separates where "
             f"MACE-MP-0 stays mixed", fontsize=12)
fig.tight_layout()
out = OUT / "cuni_demix_mc.png"
fig.savefig(out, dpi=150)
json.dump({"T": args.T, "om_pre": args.om_pre, "om_tuned": args.om_tuned,
           "unlike_pre": u_pre, "unlike_tuned": u_tun,
           "Tc_pre": Tc(args.om_pre), "Tc_tuned": Tc(args.om_tuned)},
          open(OUT / "cuni_demix_mc.json", "w"), indent=1)
print(f"\nwrote {out}")
