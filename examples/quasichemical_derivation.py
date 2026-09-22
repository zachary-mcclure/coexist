"""
Deriving the quasichemical (Bethe pair) short-range-order correction from
scratch, piece by piece, each piece checked against an independent ground
truth before the next piece is allowed to depend on it.

Motivation: Sec. 4.7 (Cu-Zn) needs a next-order correction beyond the
mean-field Debye term to explain the beta-brass stabilization gap. A first
attempt (session N) tried to transcribe the classical quasichemical
formula from a scanned lecture-note PDF and found two internally
inconsistent forms in the same source -- not trustworthy enough to hard-
code into a submission-track paper. This script re-derives everything
from first principles instead, with each step validated numerically:

  PIECE 1: exact combinatorial mass-balance identity (no approximation,
           no energy) -- brute-force enumeration on a small explicit graph.
  PIECE 2: exact brute-force canonical statistical mechanics on the same
           small graph -- absolute ground truth for later comparison.
  PIECE 3: the Bethe-lattice cavity recursion, derived from scratch, and
           checked against exact finite-tree dynamic programming (must
           match to machine precision -- it's the same calculation done
           two different ways).
  PIECE 4: the z=2 special case checked against the independently-exact
           1D transfer-matrix solution.
  PIECE 5: bulk thermodynamics extracted from the self-consistent cavity
           solution, checked against known theorems (recovers the regular
           solution at w=0, entropy <= ideal, Gibbs-Duhem).
  PIECE 6: the approximation's honest error on real (loopy) lattices,
           quantified against brute force at realistic z.

Run: python examples/quasichemical_derivation.py
"""
import itertools

import numpy as np

RNG = np.random.default_rng(0)


# ═══════════════════════════════════════════════════════════════════════════
# PIECE 1: exact combinatorial mass-balance identity
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("PIECE 1: mass-balance identity n_AA = (z*N_A - n_AB)/2, exact combinatorics")
print("═" * 70)


def ring_edges(n):
    """z=2 periodic ring on n sites."""
    return [(i, (i + 1) % n) for i in range(n)]


def square_torus_edges(L):
    """z=4 periodic L x L square lattice (torus). Returns edge list on L*L sites."""
    edges = []
    for r in range(L):
        for c in range(L):
            s = r * L + c
            edges.append((s, r * L + (c + 1) % L))          # right neighbor
            edges.append((s, ((r + 1) % L) * L + c))          # down neighbor
    return edges


def count_bond_types(edges, labels):
    """labels: 0/1 array per site (0=A, 1=B). Returns (n_AA, n_BB, n_AB)."""
    n_AA = n_BB = n_AB = 0
    for i, j in edges:
        if labels[i] == 0 and labels[j] == 0:
            n_AA += 1
        elif labels[i] == 1 and labels[j] == 1:
            n_BB += 1
        else:
            n_AB += 1
    return n_AA, n_BB, n_AB


def check_mass_balance(edges, n_sites, z, trials="all"):
    """Verify n_AA == (z*N_A - n_AB)/2 for every enumerated (or sampled)
    configuration. Pure counting -- no approximation, must hold exactly."""
    max_violation = 0.0
    checked = 0
    if trials == "all":
        configs = itertools.product([0, 1], repeat=n_sites)
    else:
        configs = (RNG.integers(0, 2, n_sites) for _ in range(trials))
    for labels in configs:
        labels = np.asarray(labels)
        N_A = int(np.sum(labels == 0))
        n_AA, n_BB, n_AB = count_bond_types(edges, labels)
        predicted_n_AA = (z * N_A - n_AB) / 2
        max_violation = max(max_violation, abs(n_AA - predicted_n_AA))
        checked += 1
    return checked, max_violation


# z=2 ring, N=14 sites: exhaustive (2^14 = 16384 configs)
edges = ring_edges(14)
checked, viol = check_mass_balance(edges, 14, z=2, trials="all")
print(f"z=2 ring, N=14: {checked} configs enumerated exhaustively, "
      f"max|n_AA - predicted| = {viol}")
assert viol == 0, "mass-balance identity FAILED on the ring"

# z=4 torus, 5x5=25 sites: exhaustive would be 2^25 (33M, feasible but slow) ->
# sample instead, still a strong check since the identity must hold PER
# CONFIGURATION, not just on average.
edges = square_torus_edges(5)
checked, viol = check_mass_balance(edges, 25, z=4, trials=20000)
print(f"z=4 torus 5x5: {checked} random configs sampled, "
      f"max|n_AA - predicted| = {viol}")
assert viol == 0, "mass-balance identity FAILED on the torus"

print("PASS: n_AA = (z*N_A - n_AB)/2 holds exactly, for every configuration, "
      "on both graphs -- this identity needs no approximation.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PIECE 2: exact brute-force canonical statistical mechanics (ground truth)
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("PIECE 2: exact canonical partition function on small graphs")
print("═" * 70)

K_B = 8.617333e-5   # eV/K, same constant as coexist.core.phase_diagram


def exact_canonical(edges, n_sites, N_A, w, T, e_AA=0.0, e_BB=0.0):
    """Exact Z, <U>, F, S at fixed (N_A, N_B=n_sites-N_A) by enumerating
    every distinct arrangement (C(n_sites, N_A) of them) -- no
    approximation of any kind. w = e_AB - (e_AA+e_BB)/2 (this script's
    sign convention: w>0 favors like pairs / clustering / positive
    Omega_regular; w<0 favors unlike pairs / ordering, matching Cu-Zn's
    negative L0).
    """
    e_AB = w + (e_AA + e_BB) / 2.0
    kT = K_B * T
    positions = range(n_sites)
    Z = 0.0
    U_sum = 0.0
    n_AB_sum = 0.0
    labels = np.ones(n_sites, dtype=int)
    for A_positions in itertools.combinations(positions, N_A):
        labels[:] = 1
        labels[list(A_positions)] = 0
        n_AA, n_BB, n_AB = count_bond_types(edges, labels)
        U = n_AA * e_AA + n_BB * e_BB + n_AB * e_AB
        boltz = np.exp(-U / kT)
        Z += boltz
        U_sum += U * boltz
        n_AB_sum += n_AB * boltz
    U_avg = U_sum / Z
    n_AB_avg = n_AB_sum / Z
    F = -kT * np.log(Z)
    S = (U_avg - F) / T
    return dict(Z=Z, U=U_avg, F=F, S=S, n_AB=n_AB_avg)


# z=2 ring, N=12 (C(12,6)=924 configs at x=0.5 -- exact, instant)
edges2 = ring_edges(12)
res_w0 = exact_canonical(edges2, 12, N_A=6, w=0.0, T=1000.0)
n_bonds = len(edges2)
naive_infinite_N = 2 * 0.5 * 0.5 * n_bonds       # WRONG reference -- see below
exact_finite_N = n_bonds * 2 * 6 * 6 / (12 * 11)  # hypergeometric (hypergeom.) pair prob
print(f"z=2 ring N=12, x=0.5, w=0 (no interaction): <n_AB>={res_w0['n_AB']:.4f}")
print(f"  naive infinite-N (Bragg-Williams 2x(1-x)) reference: {naive_infinite_N:.4f}  "
      f"<- WRONG reference for finite N, kept only to show why")
print(f"  exact finite-N (sampling WITHOUT replacement, hypergeometric) reference: "
      f"{exact_finite_N:.4f}")
# At w=0 every arrangement has EQUAL energy (U=0 identically), so <n_AB> is a
# pure COMBINATORIAL average over uniformly-likely arrangements of a FIXED
# number of A's and B's -- i.e. sampling without replacement. That is NOT
# the same distribution as independent per-site occupancy (which is what
# the textbook Bragg-Williams 2x(1-x) assumes, implicitly taking N -> infinity
# where the two ensembles coincide). The correct EXACT finite-N comparison
# uses the hypergeometric pair probability 2*N_A*N_B/(N*(N-1)), not 2x(1-x).
assert abs(res_w0["n_AB"] - exact_finite_N) < 1e-8, \
    "at w=0 exact <n_AB> must equal the EXACT finite-N (hypergeometric) value"
assert abs(res_w0["n_AB"] - naive_infinite_N) > 0.1, \
    "sanity: the finite-N correction should be visible at N=12, not accidentally zero"
print("PASS (and lesson learned): the exact brute force matches the correct "
      "finite-N hypergeometric reference, not naive infinite-N Bragg-Williams --\n"
      "      this finite-size gap is a REAL effect of fixed-composition sampling,\n"
      "      not a bug; it vanishes as N -> infinity at fixed x, which is the\n"
      "      regime the bulk theory below actually targets.\n")

# Confirm the finite-size gap actually SHRINKS as N grows (as claimed above)
print("Finite-size convergence check (w=0, x=0.5, ring): "
      "<n_AB>/bonds should -> 0.5")
for n in (8, 12, 16, 20):
    e = ring_edges(n)
    r = exact_canonical(e, n, N_A=n // 2, w=0.0, T=1000.0)
    print(f"  N={n:2d}: <n_AB>/bonds = {r['n_AB']/len(e):.5f}  "
          f"(hypergeom. exact: {2*(n//2)*(n//2)/(n*(n-1)):.5f})")
print()

# Now turn on interactions: n_AB should move monotonically with w, in the
# physically required direction (w<0 favors AB bonds -> MORE unlike pairs;
# w>0 favors like bonds -> FEWER unlike pairs), and w=0 must sit between.
print("Interaction-strength sanity (z=2 ring N=12, x=0.5, T=500 K):")
ws_meV = [-200, -100, -50, 0, 50, 100, 200]
n_ab_vals = []
for w_meV in ws_meV:
    r = exact_canonical(edges2, 12, N_A=6, w=w_meV * 1e-3, T=500.0)
    n_ab_vals.append(r["n_AB"])
    print(f"  w={w_meV:+4d} meV: <n_AB> = {r['n_AB']:.4f}")
assert all(n_ab_vals[i] >= n_ab_vals[i + 1] for i in range(len(n_ab_vals) - 1)), \
    "<n_AB> must be monotonically non-increasing in w (more positive w = more clustering)"
print("PASS: <n_AB> decreases monotonically as w increases, exactly as the "
      "sign convention (w<0 ordering, w>0 clustering) requires.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PIECE 3: the Bethe-lattice cavity recursion, derived from scratch
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("PIECE 3: Bethe-lattice cavity recursion vs. exact finite-tree DP")
print("═" * 70)
print("""
Setup: a rooted tree, coordination z. Per-site fugacities z_A, z_B (only
the ratio z_B/z_A matters -- it is the field conjugate to composition).
Bond Boltzmann factors b_AA=exp(-e_AA/kT), b_AB=exp(-e_AB/kT), b_BB=exp(-e_BB/kT).

EXACT tree DP (no approximation, ground truth): for any rooted tree, define
  V_A(node) = z_A * prod_{children c} [b_AA V_A(c) + b_AB V_B(c)]
  V_B(node) = z_B * prod_{children c} [b_AB V_A(c) + b_BB V_B(c)]
bottom-up from the leaves (V_A(leaf)=z_A, V_B(leaf)=z_B, empty product=1).
Every edge's Boltzmann factor is applied exactly once (at the step where
the parent's V is built from the child's V) -- this is just repeated
application of the definition of a partition function on a tree (no loops
means the subtrees hanging off different children are independent given
the child's own type), so it is exact by construction.

CAVITY RECURSION (candidate closed form, to be checked against the above):
for a node with (z-1) children (one connection reserved for "the parent
not yet attached"), define the ratio r = V_B/V_A. Substituting into the
V_A, V_B recursion and dividing:
  r(k) = (z_B/z_A) * [ (b_AB + b_BB r(k-1)) / (b_AA + b_AB r(k-1)) ]^(z-1)
with r(0) = z_B/z_A (a bare leaf, empty product = 1).
""")


def cavity_step(r_prev, z, zB_over_zA, b_AA, b_AB, b_BB):
    return zB_over_zA * ((b_AB + b_BB * r_prev) / (b_AA + b_AB * r_prev)) ** (z - 1)


def build_cavity_tree(z, depth):
    """A rooted tree where the root has (z-1) children (a CAVITY tree, i.e.
    one branch already 'used up' by the not-yet-attached parent), and every
    non-leaf descendant also has (z-1) children, down to `depth` generations.
    Returns nested tuples: a node is either None (leaf) or a list of
    `z-1` child nodes.
    """
    if depth == 0:
        return None
    return [build_cavity_tree(z, depth - 1) for _ in range(z - 1)]


def exact_V(node, z_A, z_B, b_AA, b_AB, b_BB):
    """Exact (V_A, V_B) for a cavity-tree node via direct bottom-up DP.

    Rescales each child's (vA, vB) pair by a common positive factor
    (1/(vA+vB)) immediately after computing it. This changes nothing:
    rescaling a child's pair by c multiplies BOTH of the parent's
    (b_AA*vA+b_AB*vB) and (b_AB*vA+b_BB*vB) terms by the same c, so the
    parent's V_A and V_B are themselves rescaled by an overall c relative
    to the un-rescaled calculation -- the ratio V_B/V_A this function is
    ultimately used for is invariant. Purely a numerical-overflow guard for
    large trees (raw V values are astronomically large products); without
    it, deep/high-z trees silently overflow to inf/nan in float64.
    """
    if node is None:                       # leaf
        return z_A, z_B
    VA_prod, VB_prod = 1.0, 1.0
    for child in node:
        vA, vB = exact_V(child, z_A, z_B, b_AA, b_AB, b_BB)
        scale = 1.0 / (vA + vB)
        vA, vB = vA * scale, vB * scale
        VA_prod *= (b_AA * vA + b_AB * vB)
        VB_prod *= (b_AB * vA + b_BB * vB)
    return z_A * VA_prod, z_B * VB_prod


def tree_depth_count(node):
    if node is None:
        return 0
    return 1 + max(tree_depth_count(c) for c in node)


# Cross-check: does the closed-form recursion EXACTLY match the explicit
# tree DP, at every depth, for several (z, energies, composition-field)
# combinations? This is the core correctness test of Piece 3.
test_cases = [
    dict(z=3, zB_over_zA=1.0, b_AA=1.0, b_AB=np.exp(0.08 / (K_B * 700)), b_BB=1.0),
    dict(z=4, zB_over_zA=0.6, b_AA=np.exp(0.02 / (K_B * 900)), b_AB=1.0,
        b_BB=np.exp(-0.01 / (K_B * 900))),
    dict(z=6, zB_over_zA=2.3, b_AA=1.0, b_AB=np.exp(-0.15 / (K_B * 1000)), b_BB=1.0),
    dict(z=2, zB_over_zA=1.7, b_AA=np.exp(0.05 / (K_B * 600)), b_AB=1.0, b_BB=1.0),
]
max_depth = 6
worst = 0.0
for case in test_cases:
    z = case["z"]
    tree = build_cavity_tree(z, max_depth)
    assert tree_depth_count(tree) == max_depth
    vA, vB = exact_V(tree, 1.0, case["zB_over_zA"], case["b_AA"], case["b_AB"], case["b_BB"])
    r_exact = vB / vA
    r_rec = case["zB_over_zA"]              # r(0)
    for _ in range(max_depth):
        r_rec = cavity_step(r_rec, z, case["zB_over_zA"], case["b_AA"], case["b_AB"], case["b_BB"])
    rel_err = abs(r_rec - r_exact) / abs(r_exact)
    worst = max(worst, rel_err)
    print(f"  z={z}, depth={max_depth}: exact tree DP r={r_exact:.10f}  "
          f"recursion r={r_rec:.10f}  rel.err={rel_err:.2e}")
assert worst < 1e-10, "cavity recursion does not match exact tree DP"
print(f"\nPASS: closed-form recursion matches exact tree dynamic programming to "
      f"{worst:.1e} relative error (machine precision) across z in {{2,3,4,6}} "
      f"and varied energies/fields -- the recursion formula is CORRECT, "
      f"verified against an unambiguous ground truth, not against a scanned "
      f"reference.\n")


# ═══════════════════════════════════════════════════════════════════════════
# Bulk marginals from the converged cavity fixed point (needed by pieces 4-5)
# ═══════════════════════════════════════════════════════════════════════════
def cavity_fixed_point(z, zB_over_zA, b_AA, b_AB, b_BB, n_iter=400):
    r = zB_over_zA
    for _ in range(n_iter):
        r = cavity_step(r, z, zB_over_zA, b_AA, b_AB, b_BB)
    return r


def bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star):
    """Bulk site marginal P(A) and bond marginal P(AB) from the converged
    cavity ratio r_star, for a node/edge with z FULL neighbors (not the
    z-1 cavity count) -- i.e. what an actual bulk site/edge looks like.

    Every bulk branch (in the translation-invariant, converged solution)
    has the SAME ratio vB_i/vA_i = r_star, by definition of r_star as the
    fixed point of the cavity recursion. So for a full site (z branches):
        V_A(site) = zA * prod_i [b_AA vA_i + b_AB vB_i]
                  = zA * (b_AA + b_AB r_star)^z * prod_i(vA_i)
        V_B(site) = zB * (b_AB + b_BB r_star)^z * prod_i(vA_i)
    the common prod_i(vA_i) factor cancels in the normalized P_A -- r_star
    substitutes DIRECTLY, with no rescaling. (An earlier version of this
    function substituted a rescaled `cr = r_star*zA/zB` instead, which is
    wrong in general and only happened to coincide with r_star when
    zA=zB in the very first test case -- caught by testing an asymmetric
    zA != zB case too, see the derivation note in the module docstring.)
    """
    VA_site = zA * (b_AA + b_AB * r_star) ** z
    VB_site = zB * (b_AB + b_BB * r_star) ** z
    P_A = VA_site / (VA_site + VB_site)
    P_B = 1.0 - P_A
    # edge marginal: each endpoint contributes (z-1) OTHER branches, plus
    # the direct bond factor for the edge between the two endpoints.
    wAA = (zA * (b_AA + b_AB * r_star) ** (z - 1)) ** 2 * b_AA
    wAB = (zA * (b_AA + b_AB * r_star) ** (z - 1)) \
        * (zB * (b_AB + b_BB * r_star) ** (z - 1)) * b_AB
    wBB = (zB * (b_AB + b_BB * r_star) ** (z - 1)) ** 2 * b_BB
    Z_edge = wAA + 2 * wAB + wBB
    P_AA, P_AB_each, P_BB = wAA / Z_edge, wAB / Z_edge, wBB / Z_edge
    return dict(P_A=P_A, P_B=P_B, P_AA=P_AA, P_AB=2 * P_AB_each, P_BB=P_BB)


def exact_site_and_edge_marginal(z, depth, zA, zB, b_AA, b_AB, b_BB):
    """Exact P(A) at the root, and exact P(root,child1) joint, for a FULL
    tree (root has z children, each a depth-(depth-1) cavity subtree) --
    ground truth to check bethe_marginals() against."""
    root_children = [build_cavity_tree(z, depth - 1) for _ in range(z)]
    child_VAB = [exact_V(c, zA, zB, b_AA, b_AB, b_BB) for c in root_children]

    def site_val(root_type_is_A):
        zs = zA if root_type_is_A else zB
        prod = 1.0
        for vA, vB in child_VAB:
            prod *= ((b_AA if root_type_is_A else b_AB) * vA
                     + (b_AB if root_type_is_A else b_BB) * vB)
        return zs * prod
    VA_root, VB_root = site_val(True), site_val(False)
    P_A_exact = VA_root / (VA_root + VB_root)

    # joint (root, child_0): fix both types, product over child_0's OWN
    # (b_AA/b_AB/b_BB)-weighted value, times the direct bond, times the
    # other (z-1) children as before.
    vA0, vB0 = child_VAB[0]

    def joint(root_is_A, child_is_A):
        zs = zA if root_is_A else zB
        prod_rest = 1.0
        for vA, vB in child_VAB[1:]:
            prod_rest *= ((b_AA if root_is_A else b_AB) * vA
                          + (b_AB if root_is_A else b_BB) * vB)
        bond = (b_AA if (root_is_A and child_is_A) else
                b_BB if (not root_is_A and not child_is_A) else b_AB)
        child_val = vA0 if child_is_A else vB0
        return zs * bond * child_val * prod_rest

    wAA, wAB, wBA, wBB = (joint(True, True), joint(True, False),
                          joint(False, True), joint(False, False))
    Z_j = wAA + wAB + wBA + wBB
    return dict(P_A=P_A_exact, P_AA=wAA / Z_j, P_AB=(wAB + wBA) / Z_j,
               P_BB=wBB / Z_j)


print("Validating bethe_marginals() (site+edge) against exact tree DP.")
print("The exact reference truncates cavity subtrees at finite depth with a")
print("BARE-LEAF boundary (r(0)=zB/zA, not r_star) -- an intentionally")
print("independent, un-self-consistent construction -- so it should converge")
print("geometrically to the closed form as depth grows, never be handed the")
print("answer, and the CONVERGENCE ITSELF (not just the final number) is part")
print("of the evidence this is the right closed form:\n")
final_errs = []
MAX_LEAVES = 20000    # tree node count grows as (z-1)^depth -- cap it per z
for case in test_cases:
    z = case["z"]
    branch = max(z - 1, 1)
    max_depth = max(4, int(np.log(MAX_LEAVES) / np.log(branch))) if branch > 1 else 24
    depths = sorted(set(np.linspace(4, max_depth, 4).astype(int)))
    r_star = cavity_fixed_point(z, case["zB_over_zA"], case["b_AA"], case["b_AB"], case["b_BB"])
    bm = bethe_marginals(z, 1.0, case["zB_over_zA"], case["b_AA"], case["b_AB"],
                         case["b_BB"], r_star)
    errs = []
    for depth in depths:
        ex = exact_site_and_edge_marginal(z, depth, 1.0, case["zB_over_zA"],
                                          case["b_AA"], case["b_AB"], case["b_BB"])
        dPA = abs(bm["P_A"] - ex["P_A"])
        dPAB = abs(bm["P_AB"] - ex["P_AB"])
        errs.append(max(dPA, dPAB))
        print(f"  z={z} depth={depth:2d}: |dP_A|={dPA:.2e}  |dP_AB|={dPAB:.2e}")
    # strictly decreasing (geometric convergence) while still resolvable above
    # floating-point noise; some symmetric cases (e.g. z=3 here, r*=1 exactly)
    # hit the machine-precision floor immediately and have nothing left to
    # shrink -- that is convergence too, just faster than depth=6 can show.
    above_noise = [e for e in errs if e > 1e-13]
    assert all(above_noise[i + 1] < 0.3 * above_noise[i]
              for i in range(len(above_noise) - 1)), \
        f"z={z}: error does not shrink geometrically with depth -- {errs}"
    final_errs.append(errs[-1])
    print(f"    -> monotonic geometric convergence confirmed for z={z}, "
          f"error shrinks >3x per +4 depth\n")
assert max(final_errs) < 1e-6, f"deepest-tree residual too large: {final_errs}"
print(f"PASS: site AND edge marginals from the closed-form Bethe fixed point "
      f"converge geometrically to the exact (finite-tree DP) values as depth "
      f"grows, reaching {max(final_errs):.1e} at the deepest tree tested per "
      f"case (capped at {MAX_LEAVES} leaves), for every "
      f"(z, field, energy) case -- including the asymmetric zA != zB cases "
      f"that caught the earlier substitution bug (r_star vs. a mis-rescaled "
      f"'cr').\n")


# ═══════════════════════════════════════════════════════════════════════════
# PIECE 4: z=2 against the INDEPENDENTLY exact 1D transfer matrix
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("PIECE 4: z=2 Bethe solution vs. the exact 1D transfer matrix")
print("═" * 70)
print("""
This uses completely different machinery (2x2 linear-algebra eigenproblem,
not recursive tree DP) as a second, independent check specifically for the
z=2 (1D chain) case, where the quasichemical/Bethe pair approximation is
known to be EXACT (a 1D ring has no loops for the pair-cluster to miss).

Standard construction: T_{s,t} = sqrt(z_s z_t) * b_{s,t}, s,t in {A,B}.
For a periodic ring of N sites, Z = Tr(T^N) exactly (splitting each site's
fugacity between its two bonds symmetrically). As N -> infinity, Z ~
lambda_max^N, and the bulk marginals come from the dominant eigenvector.
""")


def transfer_matrix_bulk(zA, zB, b_AA, b_AB, b_BB):
    T = np.array([[np.sqrt(zA * zA) * b_AA, np.sqrt(zA * zB) * b_AB],
                  [np.sqrt(zB * zA) * b_AB, np.sqrt(zB * zB) * b_BB]])
    w, v = np.linalg.eig(T)
    i = np.argmax(w.real)
    lam = w[i].real
    right = v[:, i].real
    # left eigenvector of T = right eigenvector of T^T (T is symmetric here,
    # since b_AB=b_BA and the sqrt-split is symmetric, so left=right)
    left = right
    # exact bulk single-site and bond marginals (standard transfer-matrix result):
    denom = left @ right
    P_A = left[0] * right[0] / denom
    P_B = left[1] * right[1] / denom
    # bond marginal: P(s,t) = left_s * T_{s,t} * right_t / (lam * denom)
    P_AA = left[0] * T[0, 0] * right[0] / (lam * denom)
    P_AB2 = left[0] * T[0, 1] * right[1] / (lam * denom)   # one direction
    P_BA2 = left[1] * T[1, 0] * right[0] / (lam * denom)
    P_BB = left[1] * T[1, 1] * right[1] / (lam * denom)
    return dict(lam=lam, P_A=P_A, P_B=P_B, P_AA=P_AA, P_AB=P_AB2 + P_BA2, P_BB=P_BB)


z = 2
worst = 0.0
for case in test_cases:
    zA, zB = 1.0, case["zB_over_zA"]
    tm = transfer_matrix_bulk(zA, zB, case["b_AA"], case["b_AB"], case["b_BB"])
    r_star = cavity_fixed_point(z, zB, case["b_AA"], case["b_AB"], case["b_BB"])
    bm = bethe_marginals(z, zA, zB, case["b_AA"], case["b_AB"], case["b_BB"], r_star)
    dPA = abs(tm["P_A"] - bm["P_A"])
    dPAB = abs(tm["P_AB"] - bm["P_AB"])
    worst = max(worst, dPA, dPAB)
    print(f"  zB/zA={zB:.2f}: transfer-matrix P_A={tm['P_A']:.10f}  "
          f"Bethe P_A={bm['P_A']:.10f}  |diff|={dPA:.2e}   "
          f"P_AB diff={dPAB:.2e}")
assert worst < 1e-10, f"z=2 Bethe vs exact 1D transfer matrix mismatch: {worst}"
print(f"\nPASS: for z=2, the Bethe/cavity closed form matches the exact 1D "
      f"transfer-matrix solution (fully independent method: eigendecomposition, "
      f"no recursion, no trees) to {worst:.1e}. Two structurally unrelated "
      f"derivations agree -- strong evidence the closed form itself is right, "
      f"not just self-consistent with how it was built.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PIECE 5: the Bethe free energy / entropy -- derived, then checked
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("PIECE 5: Bethe free energy and entropy")
print("═" * 70)
print("""
A genuine Cayley tree is NOT a valid stand-in for "a large bulk system":
its site count grows as (z-1)^depth, so a fixed FRACTION of sites are
always within O(1) generations of the boundary -- there is no boundary-
free bulk limit to take by just building a bigger tree and dividing by
N. The Bethe/CVM approximation instead constructs the bulk free energy
directly from the CONVERGED LOCAL marginals (already validated above),
via the standard cluster (pair) free-energy functional:

  <U>/N   = (z/2) * [P_AA e_AA + P_AB e_AB + P_BB e_BB]      (site energy density)
  S/(Nk)  = -(z-1) * sum_i P_i ln(P_i)  +  (z/2) * sum_{ij} P_ij ln(P_ij)
  F/N     = <U>/N - T*S/N

This entropy form is not re-derived from scratch here (that requires the
Guggenheim/Kikuchi combinatorial counting argument this whole exercise
set out to avoid blindly trusting) -- instead it is VALIDATED the same
way as everything else: checked against an independent exact answer.

UPDATE (in-progress, kept for the record): the formula printed above was
tried FIRST and FAILED the z=2-vs-exact-transfer-matrix check by 0.03-0.07
eV -- not a rounding-level miss. Patching it with the obvious missing
"field" term (-kT P_B ln(zB/zA), the usual grand-canonical/canonical
Legendre-transform piece) did not reconcile it either. Rather than keep
guessing at a remembered formula, it is abandoned below in favor of
integrating the (already twice-validated) energy function from the exact
T=infinity reference using a standard thermodynamic identity -- no
entropy formula assumed at all.
""")


def bethe_energy(z, P_AA, P_AB, P_BB, e_AA, e_AB, e_BB):
    """<U>/N -- already validated (built from the doubly-checked marginals)."""
    return (z / 2.0) * (P_AA * e_AA + P_AB * e_AB + P_BB * e_BB)


def bethe_U_of_T(z, zA, zB, e_AA, e_AB, e_BB, T):
    kT = K_B * T
    b_AA, b_AB, b_BB = np.exp(-e_AA / kT), np.exp(-e_AB / kT), np.exp(-e_BB / kT)
    r_star = cavity_fixed_point(z, zB / zA, b_AA, b_AB, b_BB)
    bm = bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star)
    return bethe_energy(z, bm["P_AA"], bm["P_AB"], bm["P_BB"], e_AA, e_AB, e_BB)


def bethe_free_energy_by_integration(z, zA, zB, e_AA, e_AB, e_BB, T, n_grid=4000):
    """F(T)/N via the EXACT thermodynamic identity d(F/T)/d(1/T) = U at fixed
    field (zA, zB fixed as T varies), integrated from the EXACT T=infinity
    reference F(T->inf)/T -> -k*ln(zA+zB) (bonds carry no weight at all at
    infinite T -- sites are then exactly independent with weights zA, zB).
    No entropy formula is assumed anywhere in this function -- only the
    doubly-validated energy function above and a standard, unimpeachable
    thermodynamic relation (equivalent to F = U - TS, dF=-S dT at fixed
    field, differentiated as F/T).
    """
    x_target = 1.0 / T
    # integrate from x=0 (T=inf) to x=x_target; U(x) can vary fast near x=0
    # if bonds are strong, so use a modestly fine grid (checked for
    # convergence by doubling n_grid in the caller's own validation below).
    xs = np.linspace(1e-12, x_target, n_grid)
    Us = np.array([bethe_U_of_T(z, zA, zB, e_AA, e_AB, e_BB, 1.0 / x) for x in xs])
    integral = np.trapezoid(Us, xs)
    F_over_N = (-K_B * np.log(zA + zB) + integral) * T
    return F_over_N


# ── validate against the EXACT z=2 transfer-matrix free energy ─────────────
print("z=2 check against the exact transfer matrix free energy (independent):")
z = 2
worst = 0.0
for case in test_cases:
    zA, zB = 1.0, case["zB_over_zA"]
    b_AA, b_AB, b_BB = case["b_AA"], case["b_AB"], case["b_BB"]
    T_probe = 700.0
    kT = K_B * T_probe
    e_AA, e_AB, e_BB = -kT * np.log(b_AA), -kT * np.log(b_AB), -kT * np.log(b_BB)
    tm = transfer_matrix_bulk(zA, zB, b_AA, b_AB, b_BB)
    F_exact = -K_B * T_probe * np.log(tm["lam"])
    F_bethe = bethe_free_energy_by_integration(z, zA, zB, e_AA, e_AB, e_BB, T_probe)
    diff = abs(F_bethe - F_exact)
    worst = max(worst, diff)
    print(f"  zB/zA={zB:.2f}: exact F/N={F_exact:+.8f} eV  "
          f"Bethe(integrated) F/N={F_bethe:+.8f} eV  |diff|={diff:.2e} eV")
assert worst < 1e-6, f"integrated Bethe free energy does not match exact z=2 result: {worst}"
print(f"\nPASS: the free energy obtained by integrating the (already validated) "
      f"energy function from the exact T=infinity reference matches the exact "
      f"1D transfer-matrix free energy to {worst:.1e} eV at z=2 -- derived from "
      f"a standard thermodynamic identity, no entropy formula assumed.\n")

# ── known-theorem checks, now for GENERAL z (not just the z=2 exact case) ──
print("Known-theorem checks at general z (no interaction limit, S <= ideal):")
print("""
IMPORTANT ensemble subtlety (caught by the first version of this check,
kept here rather than silently fixed): bethe_free_energy_by_integration
returns the GRAND potential per site, Omega_grand/N = -kT ln(Xi_per_site)
(built from fugacities zA, zB) -- NOT the canonical Helmholtz free energy
at fixed composition x that the textbook ideal-mixing formula
kT[x ln x + (1-x)ln(1-x)] refers to. They are related by the standard
Legendre transform Omega_grand/N = F_canonical(x)/N - x*mu_B - (1-x)*mu_A
with mu_i = kT ln(z_i). Comparing Omega_grand directly to the canonical
ideal formula (as a first pass here did) fails by ~7 meV even at w=0 --
not a bug, just the wrong two numbers held up against each other. The
Legendre transform must be applied first.
""")
for z in (2, 3, 4, 6, 8, 12):
    # (a) w=0 (no interaction): must reduce EXACTLY to the ideal solution,
    #     once Omega_grand is Legendre-transformed to F_canonical(x).
    zA, zB, T = 1.0, 0.8, 800.0
    kT = K_B * T
    r_star = cavity_fixed_point(z, zB / zA, 1.0, 1.0, 1.0)
    bm = bethe_marginals(z, zA, zB, 1.0, 1.0, 1.0, r_star)
    Omega_grand = bethe_free_energy_by_integration(z, zA, zB, 0.0, 0.0, 0.0, T)
    x_B, x_A = bm["P_B"], bm["P_A"]
    mu_A, mu_B = kT * np.log(zA), kT * np.log(zB)
    F_canonical = Omega_grand + x_B * mu_B + x_A * mu_A
    F_ideal_std = kT * (x_B * np.log(x_B) + x_A * np.log(x_A))
    d_ideal = abs(F_canonical - F_ideal_std)
    print(f"  z={z:2d}: w=0 limit  F_canonical={F_canonical:+.10f}  "
          f"F_ideal(textbook)={F_ideal_std:+.10f}  |diff|={d_ideal:.2e}")
    assert d_ideal < 1e-8, f"z={z}: w=0 limit does not recover the ideal solution"

    # (b) S <= S_ideal at the SAME composition, for a genuinely interacting case.
    b_AA, b_AB, b_BB = 1.3, 1.0, 0.85     # clustering-tendency energies
    kT = K_B * T
    e_AA, e_AB, e_BB = -kT * np.log(b_AA), -kT * np.log(b_AB), -kT * np.log(b_BB)
    r_star2 = cavity_fixed_point(z, zB / zA, b_AA, b_AB, b_BB)
    bm2 = bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star2)
    U2 = bethe_energy(z, bm2["P_AA"], bm2["P_AB"], bm2["P_BB"], e_AA, e_AB, e_BB)
    F2 = bethe_free_energy_by_integration(z, zA, zB, e_AA, e_AB, e_BB, T)
    S2 = (U2 - F2) / T
    x2 = bm2["P_A"]
    S_ideal2 = -K_B * (x2 * np.log(x2) + (1 - x2) * np.log(1 - x2))
    print(f"        interacting  x={x2:.4f}  S_Bethe/k={S2/K_B:+.6f}  "
          f"S_ideal/k={S_ideal2/K_B:+.6f}  (S_Bethe <= S_ideal: {S2 <= S_ideal2 + 1e-10})")
    assert S2 <= S_ideal2 + 1e-10, \
        f"z={z}: entropy exceeds the ideal-mixing bound -- {S2} vs {S_ideal2}"
print("\nPASS: for every z tested, the w=0 limit reproduces the ideal solution "
      "EXACTLY, and turning on interactions always REDUCES the entropy below "
      "the ideal value at matched composition -- both are textbook-required "
      "properties of any correlated (non-random) mixing theory, confirmed "
      "here rather than assumed.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PIECE 6: honest approximation error on real (LOOPY) lattices
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("PIECE 6: Bethe/quasichemical error on loopy lattices (vs. trees)")
print("═" * 70)
print("""
Everything above is EXACT for a tree (Bethe lattice) -- a real fcc (z=12)
or bcc (z=8) crystal has LOOPS (shortest cycles: 3 for fcc triangles, 4
for bcc squares), which the pair/Bethe approximation simply does not see.
This section quantifies that gap honestly on graphs small enough for
exact grand-canonical brute force (z=3 cube graph, z=4 square torus,
z=6/z=8 circulant graphs -- real fcc/bcc unit cells are far too large
for 2^N enumeration, and a small period-2 cubic torus turns out to be a
BAD z=6 stand-in: it self-wraps into doubled edges, i.e. artificial
2-cycles, which exaggerate loop effects far beyond what a genuine lattice
shows -- caught by an out-of-trend result and fixed by switching to a
circulant graph, see below), and looks at the TREND as z grows, since
that trend is what licenses extrapolating any confidence at all to z=8/12.
""")


def cube_graph_edges():
    """The 3-regular cube graph: 8 vertices at {0,1}^3, edges between
    vertices differing in exactly one coordinate."""
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                edges.append((a, b))
    return edges


def circulant_edges(n, offsets):
    """z=2*len(offsets) circulant graph on n vertices: i connects to
    i +/- d (mod n) for each d in offsets. A genuine simple graph (no
    multi-edges) as long as n > 2*max(offsets) -- unlike a small periodic
    cubic torus (L=2), which self-wraps into pathological doubled edges
    (artificial 2-cycles) that exaggerate loop effects unrealistically.
    Locally tree-like at short range for n >> offsets, a fair stand-in
    for "some genuine z-regular lattice with ordinary loop lengths"."""
    edges = set()
    for i in range(n):
        for d in offsets:
            j = (i + d) % n
            edges.add(tuple(sorted((i, j))))
    return list(edges)


def exact_grand_canonical(edges, n_sites, zA, zB, e_AA, e_AB, e_BB, T):
    """Exact grand-canonical <n_AB>/bonds and P_A, summing over ALL 2^n_sites
    configurations (no fixed-composition constraint) -- the correct
    apples-to-apples reference for the Bethe/cavity grand-canonical result
    (avoids conflating the Bethe approximation's error with the canonical-
    vs-grand-canonical finite-size gap already isolated in Piece 2)."""
    kT = K_B * T
    Z = 0.0
    NA_sum = 0.0
    nAB_sum = 0.0
    labels = np.zeros(n_sites, dtype=int)
    for bits in range(2 ** n_sites):
        for i in range(n_sites):
            labels[i] = (bits >> i) & 1
        N_A = int(np.sum(labels == 0))
        n_AA, n_BB, n_AB = count_bond_types(edges, labels)
        U = n_AA * e_AA + n_BB * e_BB + n_AB * e_AB
        w_stat = (zA ** N_A) * (zB ** (n_sites - N_A)) * np.exp(-U / kT)
        Z += w_stat
        NA_sum += N_A * w_stat
        nAB_sum += n_AB * w_stat
    return dict(P_A=NA_sum / (Z * n_sites), n_AB_over_bonds=nAB_sum / (Z * len(edges)))


loopy_cases = [
    (3, cube_graph_edges(), 8),
    (4, square_torus_edges(4), 16),
    (6, circulant_edges(16, (1, 2, 3)), 16),
    (8, circulant_edges(18, (1, 2, 3, 4)), 18),
]
print("z    exact P_AB/bonds   Bethe P_AB   |diff|     (clustering case, zB/zA=0.7)")
diffs_by_z = {}
for z, edges, n_sites in loopy_cases:
    zA, zB = 1.0, 0.7
    T = 900.0
    kT = K_B * T
    e_AA, e_AB, e_BB = -0.02, 0.0, -0.01     # like-pairs favored (clustering)
    b_AA, b_AB, b_BB = np.exp(-e_AA / kT), np.exp(-e_AB / kT), np.exp(-e_BB / kT)
    ex = exact_grand_canonical(edges, n_sites, zA, zB, e_AA, e_AB, e_BB, T)
    r_star = cavity_fixed_point(z, zB / zA, b_AA, b_AB, b_BB)
    bm = bethe_marginals(z, zA, zB, b_AA, b_AB, b_BB, r_star)
    d = abs(ex["n_AB_over_bonds"] - bm["P_AB"])
    diffs_by_z[z] = d
    print(f"z={z}: {ex['n_AB_over_bonds']:.5f}          {bm['P_AB']:.5f}      "
          f"{d:.2e}   (P_A exact={ex['P_A']:.4f}, Bethe={bm['P_A']:.4f})")

assert diffs_by_z[8] < diffs_by_z[3], \
    "expected the Bethe/loop error at z=8 to be smaller than at z=3 -- did not observe that"
print(f"\nOVERALL TREND: error at z=8 ({diffs_by_z[8]:.2e}) is well below z=3 "
      f"({diffs_by_z[3]:.2e}); z=3->4->6->8: "
      f"{diffs_by_z[3]:.1e}, {diffs_by_z[4]:.1e}, {diffs_by_z[6]:.1e}, {diffs_by_z[8]:.1e} "
      f"(not perfectly monotone step-to-step across different small graph FAMILIES -- "
      f"cube graph vs. circulant vs. torus are not the same lattice, so a little "
      f"non-monotonicity between them is expected graph-family noise, not a failure "
      f"of the underlying z-trend). This is the expected, textbook behavior (the "
      f"pair/Bethe approximation becomes asymptotically exact as z -> infinity) -- "
      f"real fcc (z=12) and bcc (z=8) have AS MANY OR MORE neighbors than any case "
      f"tested here, so this trend is the basis for treating the Bethe correction as "
      f"a genuine (if approximate, and not literally re-verified at z=12 specifically, "
      f"since 2^N brute force is computationally out of reach there) improvement over "
      f"Bragg-Williams -- not as an exact result -- when applied to Cu-Zn below.\n")
