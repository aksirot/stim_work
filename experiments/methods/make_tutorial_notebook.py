"""Generate three_techniques_18_4_4.ipynb (source only; cells are NOT executed here).

A from-scratch, "open the hood" walkthrough of the three fail-fast techniques (arXiv:2511.15177)
as implemented in stim_work/src, on the smallest interesting code: the Kunlun processor's
[[18,4,4]] bivariate-bicycle code (arXiv:2505.09684). Everything is small enough to print and to
brute-force-verify with direct Monte-Carlo.

Run order: just run top-to-bottom in the `qec` kernel. Self-contained (defines the code inline).
"""
import json
from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

# ===========================================================================
md(r"""# The three fail-fast techniques, from the inside — Kunlun **[[18,4,4]]** code

A hands-on walkthrough of *exactly how* each of the three rare-event techniques in
`stim_work/src` works (paper: *Fail fast: techniques to probe rare events in QEC*,
arXiv:2511.15177). For each technique we **call the high-level `src` function AND reproduce its
inner steps inline**, on a code small enough that every intermediate object is printable:

| | technique | `src` module | what it gives |
|---|---|---|---|
| **II** | minimum-weight onset (distance, $L(D)$, onset) | `min_weight.py` | circuit fault distance $D$, onset weight $w_0=\lceil D/2\rceil$, the exact set $L(D)$ |
| **I** | failure-spectrum ansatz (importance sampling) | `importance_sampling.py` | `f(w)`, the fitted ansatz, and `LER(p)` by reweighting |
| **III** | replica-exchange splitting | `splitting.py` | `LER(p)` deep in the rare regime via a tempered ladder |

**Why this code.** The [[18,4,4]] bivariate-bicycle code (demonstrated on the 32-qubit *Kunlun*
processor, arXiv:2505.09684) has just n=18 data qubits, so the detector-error-model is tiny, the
minimum-weight logicals can be enumerated **exactly**, and a **direct Monte-Carlo** ground truth is
cheap. That lets us *check* all three techniques against brute force — impossible for the gross
[[144,12,12]] code.

**One subtlety up front.** The *code* distance is 4, but the *syndrome-extraction circuit* has fault
distance **D=3** — the two-qubit gates spread single faults into weight-2 data errors (*hook errors*),
knocking the circuit distance below the code distance. (Same phenomenon as the gross code: code 12 →
circuit 11.) D=3 is **odd**, so — exactly as in the bb144 report — the onset *weight* `w₀=⌈D/2⌉=2`
is exact but the onset *fraction* `f₀` is **fitted, not pinned** (Proposition 1 needs even D).""")

# ---------------------------------------------------------------------------
md("""## Setup — import the building blocks
`src/` is an editable install (`pip install -e .` in the qec env), so every module imports directly.""")

code('''import numpy as np, matplotlib.pyplot as plt

# --- the code + circuit + decoder ---
from bb_code_sim import (BBCodeParams, build_parity_checks, find_logical_ops,
                         _gf2_rank, BBCodeSimulator, RelayBPDecoder)
from surface_code_sim import ErrorModel
# --- Technique II : distance / onset / logical counts ---
from min_weight import (dem_check_action_matrices, compute_distance,
                        _mw_init, _mw_coset_enum_task, optimal_onset_fraction)
# --- Technique I : importance sampling + ansatz ---
from importance_sampling import (_parse_dem, _expand, _sample_failures_at_weight,
                                 importance_sample, fit_failure_spectrum,
                                 failure_spectrum_ansatz, logical_error_rate_from_ansatz)
# --- Technique III : replica-exchange splitting ---
from splitting import (replica_exchange_estimate, min_weight_logical_seeds,
                       _config_fails, _config_syndrome_truth, _direct_mc_failure_prob)
print("imports OK")''')

# ===========================================================================
md(r"""## §0 — The Kunlun [[18,4,4]] code

A bivariate-bicycle (BB) code is built from two polynomials over $\mathbb{F}_2[x,y]/(x^{\ell}-1,\,y^{m}-1)$,
where $x$ and $y$ are cyclic shifts of the two factors of an $\ell\times m$ torus. The Kunlun
construction (arXiv:2505.09684) uses $\ell=m=3$ with

$$A = x + 1 + y^2, \qquad B = y + 1 + x^2,$$

and the CSS checks $H_X=[A\,|\,B]$, $H_Z=[B^{\top}\,|\,A^{\top}]$. In `bb_code_sim.BBCodeParams` a
polynomial is a list of `(x_exp, y_exp)` monomials, so $A = x^1+y^0+y^2 \to$ `[(1,0),(0,0),(0,2)]`.""")

code('''# Kunlun [[18,4,4]]:  A = x + 1 + y^2 ,  B = y + 1 + x^2   (l = m = 3)
P = BBCodeParams(l=3, m=3,
                 a_exps=[(1, 0), (0, 0), (0, 2)],
                 b_exps=[(0, 1), (0, 0), (2, 0)],
                 distance=4)

H_X, H_Z = build_parity_checks(P)            # H_X=[A|B], H_Z=[B^T|A^T]
n = 2 * P.l * P.m
A_blk, B_blk = H_X[:, :P.l*P.m], H_X[:, P.l*P.m:]
k = n - _gf2_rank(H_X) - _gf2_rank(H_Z)       # CSS: k = n - rank(H_X) - rank(H_Z)
lx, lz = find_logical_ops(H_X, H_Z)

print(f"n = 2·l·m = {n}        H_X shape = {H_X.shape}")
print(f"k = n - rk(H_X) - rk(H_Z) = {k}")
print(f"A == B^T ? {np.array_equal(A_blk % 2, B_blk.T % 2)}   (so H_X == H_Z: {np.array_equal(H_X%2, H_Z%2)})")
print(f"find_logical_ops -> {lx.shape[0]} X-logicals / {lz.shape[0]} Z-logicals")
print("\\nA block (9x9 circulant):\\n", A_blk)''')

md("""Now the **noisy memory circuit** (`build_circuit`, symmetric depolarizing noise) and a
Relay-BP decoder. Everything below operates on this circuit's detector-error-model (DEM).""")

code('''P_REF, ROUNDS = 0.01, 2          # reference rate + syndrome rounds (kept small so objects stay printable)
circuit = BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(P_REF), rounds=ROUNDS)
dem = circuit.detector_error_model(decompose_errors=False)
print(f"circuit: {circuit.num_detectors} detectors, {circuit.num_observables} observables, "
      f"{dem.num_errors} DEM fault mechanisms")''')

# ===========================================================================
md(r"""## §1 — Technique II: the perfect-decoder floor — distance, onset, exact `L(D)`  (`min_weight.py`)

We do this **first** because it sets the *reference a theoretically perfect decoder would achieve* —
the floor that Technique I then measures the **real** decoder against. It is a property of the code +
circuit alone (no decoder, no sampling): a minimum-weight / ML decoder corrects every fault of weight
below the onset `w₀ = ⌈D/2⌉`, so its failure spectrum is exactly zero there. Two `src` calls:

1. **`dem_check_action_matrices(circuit)`** parses the DEM into a check matrix `H` (which detectors a
   fault flips) and an action matrix `A` (which logical observables it flips), plus per-mechanism
   multiplicities `mult` and prior probabilities. Columns are fault mechanisms.
2. **`compute_distance(circuit)`** finds the circuit fault distance `D`: for each logical it appends
   a *forcing row* to `H`, decodes the syndrome that fires only that row → a minimum-weight fault
   that flips that logical. `D` is the minimum over logicals.""")

code('''H, A, mult, priors = dem_check_action_matrices(circuit)
K = A.shape[0]                                   # number of logical observables
print(f"H (checks x mechanisms) = {H.shape}   A (logicals x mechanisms) = {A.shape}")

D = compute_distance(circuit).distance
w0 = -(-D // 2)                                  # ceil(D/2)
print(f"circuit fault distance D = {D}   (code distance = {P.distance}; D < d  -> hook errors, and D is {'EVEN' if D%2==0 else 'ODD'})")
print(f"onset weight  w0 = ceil(D/2) = {w0}   (exact even for odd D; the onset FRACTION f0 is not — needs even D)")''')

md(r"""**The exact $L(D)$ — open the hood.** `find_all_min_weight_logicals` enumerates every
minimum-weight logical by decoding all $2^K-1$ *cosets* of the logical group, then branching within
each coset by column-exclusion (`_mw_coset_enum_task`). It normally parallelises with a process
`Pool`; here we run it **serially** (only $2^4-1 = 15$ cosets) — which both avoids Windows/Jupyter
multiprocessing and lets us see the loop. `_mw_init` loads the matrices into the worker globals.""")

code('''_mw_init(H, A, priors, 10, 200)                  # osd_order=10, max_iter=200
L_D = set()
for mask in range(1, (1 << K)):                  # every nonzero coset of the K-dim logical group
    for support in _mw_coset_enum_task((mask, D, 40)):   # budget 40 decodes/coset
        if len(support) == D:
            L_D.add(frozenset(support))
print(f"exact |L(D)| = {len(L_D)}   (over {(1<<K)-1} cosets; each support is a set of {D} fault mechanisms)")
print("example min-weight logical (mechanism indices):", sorted(next(iter(L_D))))
print(f"\\nTechnique II summary:  D = {D},  w0 = {w0},  |L(D)| = {len(L_D)}  (all EXACT for this tiny code)")''')

md(r"""**What's actually in $L(D)$.** The cell above only counted the configurations and printed one
example. Here we print **all** of them: each row is one minimum-weight logical — the $D=3$ DEM fault
mechanisms it's made of, a check that it fires **no net detectors** (so the decoder can't see it),
and which logical observable(s) it flips. The final tally groups them by which observable pattern
they hit, i.e. across the logical sectors.""")

code('''# --- Spell out every configuration in L(D) ---
# Each config is a set of D=3 DEM fault mechanisms whose combined action flips a logical
# observable while firing NO net detectors (an undetectable, minimum-weight logical fault).
from collections import Counter

L_D_sorted = sorted(tuple(sorted(s)) for s in L_D)   # deterministic order

print(f"The {len(L_D_sorted)} minimum-weight logical configurations L(D)  (D = {D}):\\n")
print(f"  {'#':>2}  {'fault mechanisms':<16}  {'#det fired':>10}  logicals flipped")
for i, sup in enumerate(L_D_sorted):
    cols = list(sup)
    det_fired = int(np.bitwise_xor.reduce(H[:, cols], axis=1).sum())   # 0 => undetectable (a logical)
    flip = np.bitwise_xor.reduce(A[:, cols], axis=1)                   # combined logical action
    obs = [j for j in range(K) if flip[j]]
    print(f"  {i:>2}  {str(cols):<16}  {det_fired:>10}  {obs}")

# How the 23 split across the logical sectors (which observable pattern each flips):
by_obs = Counter(tuple(j for j in range(K)
                       if np.bitwise_xor.reduce(A[:, list(s)], axis=1)[j])
                 for s in L_D_sorted)
print("\\nbreakdown by flipped-observable pattern:")
for pat, c in sorted(by_obs.items()):
    print(f"  observables {list(pat)}: {c} configs")
assert all(int(np.bitwise_xor.reduce(H[:, list(s)], axis=1).sum()) == 0 for s in L_D_sorted), \\
    "every L(D) config must be undetectable (no net detectors)"''')

md(r"""**From a mechanism index to the physical fault.** A mechanism index is a *DEM mechanism* index —
column `j` of `H`/`A` is exactly the `j`-th `error` instruction of the detector-error-model. There are
two levels of "what is it really":

- *Which checks it trips.* The instruction's targets list the detectors (`D#`) and observables (`L#`)
  it flips; each detector carries coordinates `[basis, check, round]` that `build_circuit` stamped on
  it (`basis` 0 = Z-check, 1 = X-check; `check` = stabilizer index 0–8; `round` = SE cycle).
- *Which physical gate caused it.* `circuit.explain_detector_error_model_errors(dem_filter=…)` traces a
  mechanism back to a representative circuit fault — the Pauli, the qubit, the gate, and how many
  `TICK`s into the circuit it occurs.

**A configuration in $L(D)$ is $D=3$ mechanisms that fire _together in a single shot_** — three
simultaneous faults, *not* a time-sequence of bitflips, and (since $D=3$) never four. Individually
each fault lights up some detectors; *combined*, their detector flips cancel pairwise (a silent
syndrome) while their observable flips don't — so the decoder sees a clean syndrome yet the logical
has flipped. The cell below spells out one such combination end to end.""")

code('''import stim
# --- Spell out ONE L(D) configuration as three simultaneous physical faults ---
# Mechanism j == column j of H/A == the j-th DEM `error` instruction. We resolve each to
#   (i)  the detectors/observables it flips  (instruction targets + detector coordinates), and
#   (ii) a representative physical fault      (explain_detector_error_model_errors): Pauli, qubit, gate, tick.
from collections import Counter

mech_insts = [inst for inst in dem.flattened() if inst.type == "error"]   # mechanism j == mech_insts[j]
det_coords = circuit.get_detector_coordinates()                           # {det_idx: [basis, check, round]}
BASIS = {0: "Z-check", 1: "X-check"}

def det_label(d):
    b, s, c = (int(round(v)) for v in det_coords[d])
    return f"D{d} ({BASIS[b]} #{s} @ round {c})"

def describe(j):
    """mechanism index -> (prior p, detectors it flips, observables it flips)."""
    tgt  = mech_insts[j].targets_copy()
    dets = sorted(t.val for t in tgt if t.is_relative_detector_id())
    obs  = sorted(t.val for t in tgt if t.is_logical_observable_id())
    assert dets == sorted(np.flatnonzero(H[:, j]).tolist())   # the targets ARE the H column
    assert obs  == sorted(np.flatnonzero(A[:, j]).tolist())   # ... and the A column
    return mech_insts[j].args_copy()[0], dets, obs

def physical_cause(j):
    """mechanism index -> a representative physical circuit fault (Pauli string, gate, tick)."""
    filt = stim.DetectorErrorModel(); filt.append(mech_insts[j])
    loc = circuit.explain_detector_error_model_errors(
        dem_filter=filt, reduce_to_one_representative_error=True)[0].circuit_error_locations[0]
    pauli = "*".join(("X" if g.gate_target.is_x_target else
                      "Y" if g.gate_target.is_y_target else "Z") + str(g.gate_target.value)
                     for g in loc.flipped_pauli_product)
    it = loc.instruction_targets                                          # n = #data qubits (qubits 0..n-1)
    qs = ", ".join(f"q{t.gate_target.value}({'data' if t.gate_target.value < n else 'ancilla'})"
                   for t in it.targets_in_range)
    return pauli, f"{it.gate} on [{qs}], {loc.tick_offset} TICKs into the circuit"

cfg = L_D_sorted[0]                                    # one specific configuration (exactly 3 mechanisms)
print(f"L(D) config #0 = mechanisms {list(cfg)} -- three faults that occur TOGETHER in one shot:\\n")
det_tally, obs_acc = Counter(), np.zeros(K, int)
for j in cfg:
    p, dets, obs = describe(j)
    pauli, cause = physical_cause(j)
    det_tally.update(dets); obs_acc[obs] += 1
    print(f"  mechanism {j}  (prior p = {p:.2e})")
    print(f"     physical cause   : {pauli}  <-  {cause}")
    print(f"     flips detectors  : {[det_label(d) for d in dets]}")
    print(f"     flips observables: L{obs}\\n")

print("  COMBINED  (the three fire at once -> XOR their flips):")
for d, c in sorted(det_tally.items()):
    print(f"     {det_label(d)}: hit {c}x  ->  {'cancels' if c % 2 == 0 else 'NET'}")
silent  = all(c % 2 == 0 for c in det_tally.values())
net_obs = [int(x) for x in np.flatnonzero(obs_acc % 2)]
print(f"     => syndrome is {'SILENT (every detector cancels)' if silent else 'NOT silent'}, "
      f"but observables L{net_obs} flip -> an undetectable weight-{D} logical error.")''')

md(r"""**Technique II's headline number: the perfect-decoder failure rate.** Distance and `L(D)` are the
*inputs*; the deliverable is the **onset fraction** `f₀* = |F(w₀)| / C(N, w₀)` — the fraction of
weight-`w₀` faults that even an optimal (max-class min-weight ≈ ML) decoder gets wrong. It sets the
**floor** `LER(p) ≈ f₀*·C(N,w₀)·q(p)^{w₀}` that *no* decoder can beat. For even `D` this is Proposition 1;
our circuit distance is odd (`D=3`), so `optimal_onset_fraction` uses the **Appendix A.6** odd-`D` recipe:
every weight-`w₀` restriction of `L(D)` fails (its weight-`(w₀−1)` complement sits in a different class
the decoder prefers), **plus** the losing weight-`w₀` restrictions of the weight-`(D+1)` logicals `L(D+1)`
under the max-class vote. (`L(D+1)` is enumerated exactly by a weight-`(D+1)` meet-in-the-middle.)""")

code('''# Technique II failure RATE -- the perfect-decoder onset fraction (now odd-D capable, paper App. A.6).
# We pass the exact L(D) enumerated above; src finds L(D+1) by half-MITM and applies the A.6 recipe.
from math import comb
onset = optimal_onset_fraction(circuit, logicals=L_D)
f0_star = onset.onset_fraction
print(f"D = {onset.distance} (odd)   onset weight w0 = {onset.onset}")
print(f"|L(D)|   = {onset.n_min_logicals}      (weight-{onset.distance} logicals, from above)")
print(f"|L(D+1)| = {onset.n_min_logicals_Dp1}    (weight-{onset.distance+1} logicals, via half-MITM)")

# The denominator: C(N, w0) = "N choose w0" = number of weight-w0 fault configs (pick w0 of the N
# expanded columns). src computes it via log-gamma (overflow-safe for big N); here N is small so the
# exact integer is identical -- spell it out:
N, w0 = onset.n_expanded, onset.onset
C_N_w0 = comb(N, w0)                               # = N*(N-1)/2 for w0=2
print(f"\\nC(N, w0) = C({N}, {w0}) = {N}*{N-1}/2 = {C_N_w0:,}   (total weight-{w0} fault configurations)")
print(f"|F(w0)|  = {onset.fail_count:,}   (of those, the ones a perfect decoder still gets wrong)")
print(f"\\nperfect-decoder onset fraction  f0* = |F(w0)| / C(N,w0) = {onset.fail_count:,}/{C_N_w0:,} = {onset.fail_count/C_N_w0:.4f}")
print(f"   (matches onset.onset_fraction = {f0_star:.4f})")
print(f"=> floor  LER(p) ~ f0* * C(N,w0) * q(p)^w0  -- the best ANY decoder can do at the onset.")''')

md(r"""**How `|F(w₀)|` splits across `L(D)` and `L(D+1)`.** A weight-`w₀` fault can only fail if some
equal-or-lighter error with the *same syndrome* sits in a *different* logical class — and pairing the
two gives a zero-syndrome, logical-flipping codeword of weight `≤ 2w₀ = D+1`, i.e. an element of `L(D)`
or `L(D+1)`. So the failing weight-`w₀` faults are exactly the weight-`w₀` **restrictions** (sub-supports)
of those two sets:

- **From `L(D)` — all of them fail.** A restriction `r ⊂ ℓ∈L(D)` has complement `ℓ∖r` of weight `w₀−1`
  (*lighter*) in a different class; a min-weight decoder prefers it, which flips the logical. We count
  every one, weighted by its expanded multiplicity `ρ(r)=∏_{j∈r} m_j`.
- **From `L(D+1)` — only the losers.** A restriction `r′ ⊂ ℓ′∈L(D+1)` has complement `ℓ′∖r′` of weight
  `w₀` (a *tie*) in a different class. The optimal decoder picks the more populated class per syndrome;
  the rest fail — the §4.3 `Σ − max` rule. (Remove any `L(D+1)` restriction already in `L(D)`'s set
  first, to avoid double-counting.)

The cell reproduces both contributions inline and checks they sum to the `src` total `|F(w₀)|`.""")

code('''# Open the hood: reproduce |F(w0)| as its L(D) + L(D+1) contributions (mirrors min_weight_fail_count_odd).
import itertools
from min_weight import find_weight_logicals_mitm
rho = lambda cols: int(np.prod(mult[list(cols)]))           # expanded multiplicity of a restriction

# Part 1 -- every weight-w0 restriction of L(D) fails (lighter complement in another class)
R_D = {frozenset(r) for s in L_D for r in itertools.combinations(sorted(s), w0)}
fails_LD = sum(rho(r) for r in R_D)

# Part 2 -- weight-w0 restrictions of L(D+1), minus L(D)'s, that LOSE the per-syndrome max-class vote
L_Dp1 = find_weight_logicals_mitm(H, A, D + 1)             # the weight-(D+1) logicals (half-MITM)
R_Dp1 = {frozenset(r) for s in L_Dp1 for r in itertools.combinations(sorted(s), w0)}
by_sig = {}
for r in (R_Dp1 - R_D):
    cols = sorted(r)
    sig = tuple(np.flatnonzero(np.bitwise_xor.reduce(H[:, cols], axis=1)))   # detectors fired (syndrome)
    act = tuple(np.flatnonzero(np.bitwise_xor.reduce(A[:, cols], axis=1)))   # observables flipped (class)
    by_sig.setdefault(sig, {})
    by_sig[sig][act] = by_sig[sig].get(act, 0) + rho(cols)
fails_LDp1 = sum(sum(cls.values()) - max(cls.values()) for cls in by_sig.values())

print(f"Part 1  all weight-{w0} restrictions of L(D)      : {fails_LD:>7,}   (each fails outright)")
print(f"Part 2  losing restrictions of L(D+1) (max-class) : {fails_LDp1:>7,}")
print(f"sum                                               : {fails_LD + fails_LDp1:>7,}")
print(f"src |F(w0)| (optimal_onset_fraction)              : {onset.fail_count:>7,}   "
      f"-> {'MATCH' if fails_LD + fails_LDp1 == onset.fail_count else 'MISMATCH'}")''')

# ===========================================================================
md(r"""## §2 — Technique I: the failure-spectrum ansatz (`importance_sampling.py`)

Idea (paper §3): the logical error rate is a reweighting of the **failure spectrum** `f(w)` — the
fraction of weight-`w` fault configurations that the decoder gets wrong:

$$\mathrm{LER}(p) \;=\; \sum_w f(w)\,\binom{N}{w}\,q(p)^w\,(1-q(p))^{N-w},\qquad q(p)=q_\text{base}\,p/p_\text{ref}.$$

Unlike Technique II, `f(w)` is a property of the **decoder you actually run** (here Relay-BP). The real
reason to have *both* techniques is the comparison: laying the measured `f(w)` against the
perfect-decoder floor from Technique II reads off the **decoder gap** — how far the real decoder is
from optimal.

**Open the hood — compressed → expanded.** `_parse_dem` gives the per-mechanism (*compressed*) arrays;
`_expand` replicates each mechanism into identical columns so all `N` *expanded* columns share one base
rate `q_base`. Sampling a weight-`w` config = choosing `w` of the `N` columns. (The cell right after
unpacks these two representations explicitly.)""")

code('''probs, det_mat, obs_mat = _parse_dem(circuit)        # det_mat: (mechanisms x detectors), obs_mat: (mechanisms x logicals)
col_to_mech, q_base, _ = _expand(probs, None)        # expanded columns -> source mechanism
N_exp = col_to_mech.shape[0]
print(f"N (expanded columns) = {N_exp}   q_base = {q_base:.5g}   (mechanisms map to {N_exp} equal-rate columns)")

# Reproduce _sample_failures_at_weight for ONE weight, inline, to see the mechanics:
def sample_f_of_w(w, shots, rng):
    M, Kk = det_mat.shape[1], obs_mat.shape[1]
    fails = 0
    dec = RelayBPDecoder(); dec.setup(circuit)
    synd = np.zeros((shots, M), bool); truth = np.zeros((shots, Kk), bool)
    for t in range(shots):
        cols = rng.choice(N_exp, size=w, replace=False)      # a weight-w config
        mech = col_to_mech[cols]                              # back to source mechanisms
        synd[t]  = np.bitwise_xor.reduce(det_mat[mech], axis=0)   # syndrome = XOR of mechanism rows
        truth[t] = np.bitwise_xor.reduce(obs_mat[mech], axis=0)   # true logical flip
    pred = dec.decode_batch(synd)
    return int(np.any(pred != truth, axis=1).sum())

rng = np.random.default_rng(0)
for w in (1, 2, 3):
    F = sample_f_of_w(w, 4000, rng)
    print(f"  weight {w}:  f(w) = {F}/4000 = {F/4000:.4f}" + ("   <- below onset w0=2: no failures" if w < w0 else ""))''')

md(r"""**Compressed vs. expanded — two views of the same DEM.** Worth pausing on, because the three
techniques live in *different* representations:

- **Compressed** (per-mechanism): the DEM as-is — `M` fault mechanisms, each with its **own** prior
  probability `p_j`. This is the space `H`, `A`, the distance, and `L(D)` live in (Technique II); a
  column is one mechanism, and the `L(D)` supports printed above are indices into it.
- **Expanded** (equal-rate columns): each mechanism `j` is replicated into `m_j = round(p_j/q_base)`
  identical columns so that **all** `N = Σ_j m_j` columns share one base rate `q_base = min_j p_j`.
  Technique I needs this — with a single common rate, a weight-`w` configuration is just a choice of
  `w` of the `N` columns and the binomial reweighting $\binom{N}{w}\,q^w(1-q)^{N-w}$ above applies.

`col_to_mech` is the bridge mapping each expanded column back to its source mechanism. Technique III
crosses the same bridge: `min_weight_logical_seeds` lifts an `L(D)` support from compressed mechanism
indices to expanded columns before splitting.""")

code('''# COMPRESSED (per-mechanism) vs EXPANDED (equal-rate columns) -- the SAME DEM, two encodings.
# (mult = per-mechanism multiplicities m_j, returned by dem_check_action_matrices in Technique II.)
M = len(probs)
print(f"COMPRESSED:  M = {M} mechanisms, each with its OWN prior p_j")
print(f"   prior p_j range : {probs.min():.2e} .. {probs.max():.2e}   (q_base = min_j p_j = {q_base:.2e})")
print(f"   m_j = round(p_j/q_base):  min {int(mult.min())}, max {int(mult.max())}, sum {int(mult.sum())}")
print(f"EXPANDED  :  N = sum_j m_j = {N_exp} columns, ALL at the single rate q_base")
print(f"   consistency: N == sum(mult)? {N_exp == int(mult.sum())}   (col_to_mech maps {N_exp} cols -> {M} mechs)")

jmax = int(np.argmax(mult))                              # the highest-probability mechanism
cols_of_jmax = np.flatnonzero(col_to_mech == jmax)
print(f"\\nexample: mechanism {jmax} has p = {probs[jmax]:.2e} = {int(mult[jmax])}x q_base")
print(f"   -> expands to columns {list(cols_of_jmax[:6])}{' ...' if mult[jmax] > 6 else ''}  "
      f"({len(cols_of_jmax)} identical copies)")
print(f"\\nso: Technique II / L(D) speak COMPRESSED indices (config #0 = {list(L_D_sorted[0])});")
print(f"    Technique I samples EXPANDED columns; Technique III lifts L(D) supports compressed -> expanded.")''')

md("""That inline loop **is** `importance_sampling._sample_failures_at_weight`. The library
`importance_sample` does it for a whole band of weights at once and returns a `FailureSpectrum`.""")

code('''is_res = importance_sample(circuit, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                           weights=list(range(1, 11)), shots_per_weight=6000, seed=1)
spec = is_res.spectrum
fw = np.array([F/T for F, T in zip(spec.failures, spec.trials)])
print("w   :", list(spec.weights))
print("f(w):", [round(x, 4) for x in fw])
print(f"\\nf(1)={fw[0]:.3f} (below onset), f(2)={fw[1]:.3f} = the onset fraction; f(w) -> saturation as w grows.")''')

md(r"""**The decoder gap — real (Relay-BP) vs. the perfect-decoder floor.** The payoff of running
Techniques I and II together is to compare the *real* decoder against the optimal floor on **two** axes:

- *Onset weight (the slope).* A perfect decoder corrects every fault below `w₀`, so `f(w)=0` for `w<w₀`.
  Relay-BP measures `f(1)=0` with its first failures at `w₀=2` — it **matches** the optimal onset weight,
  so there is *no slope gap*.
- *Onset fraction (the prefactor).* Here Technique II's exact `f₀*≈0.020` is the floor, and Relay-BP's
  measured `f(w₀)` sits **above** it. That ratio is the **decoder gap** — and it is real even for this
  tiny code.

This refines the usual story: a small code is *not* automatically decoded optimally. Relay-BP nails the
slope but leaves a ~1.5× prefactor gap (a stronger/longer decoder only partly closes it); for the gross
[[144,12,12]] code the gap is far larger.""")

code('''# Compare the REAL decoder (Technique I) to the PERFECT-decoder floor (Technique II) on BOTH axes.
onset_meas = next((w for w, F in zip(spec.weights, spec.failures) if F > 0), None)
f_relay_w0 = fw[w0 - 1]                              # measured Relay-BP f(w0) (spec.weights starts at 1)
print(f"onset WEIGHT  : perfect corrects all w<w0;  Relay-BP first failing weight = {onset_meas}, w0 = {w0}"
      f"   -> {'match (no slope gap)' if onset_meas == w0 else 'GAP'}")
print(f"onset FRACTION: perfect f0* = {f0_star:.4f};  Relay-BP f({w0}) = {f_relay_w0:.4f}"
      f"   -> decoder gap x{f_relay_w0/f0_star:.2f}")
print(f"\\n=> Relay-BP achieves the optimal onset WEIGHT but sits ~{f_relay_w0/f0_star:.1f}x above the optimal")
print(f"   onset FRACTION -- a real decoder gap, notable for a code THIS small.")''')

md(r"""**Fitting the ansatz — Technique I on its own (unpinned).** `fit_failure_spectrum` fits a smooth
3–5 parameter form to the measured `f(w)`. The fit target is the **failure spectrum**, not the LER: it
minimizes χ² on **log f(w)** — residual `= (log f_ansatz(w) − log f̂(w)) / σ_log`, with delta-method
binomial weights `σ_log = se/f̂`, over the weights that had ≥1 observed failure — because `f(w)` spans
orders of magnitude and log-space keeps that dynamic range well conditioned. The `LER(p)` is then a
*downstream reweighting* of the fitted `f(w)`; it never enters the fit objective.

We leave the onset weight `w₀` **free**, so Technique I stays fully independent of Technique II. The f5
objective is **non-convex** — a naive single start can fall into a bad basin (a low-`gamma1` fit that
matches only the onset and undershoots the rest of the spectrum several-fold), so `fit_failure_spectrum`
uses a small **multistart** and keeps the lowest-cost solution. With that, the free-`w₀` fit reliably
lands on the good minimum: it recovers `w₀→2`, tracks `f(w)` across the whole range (see the log-scale
panel below), and its reweighted `LER` matches direct Monte-Carlo — **essentially identical to pinning
`w₀=2` from Technique II** (same cost, same params). So here the techniques stand on their own and still
agree; pinning `w₀` remains a useful stabilizer when data is scarce or you can't afford the multistart.""")

code('''p_grid = np.geomspace(1e-4, 0.013, 40)   # down to p = 1e-4 -- the rare-event regime
fits, LER = {}, {}
for label, model, w0pin in [("f3, free w0", "f3", None),
                            ("f5, free w0", "f5", None),
                            ("f5, w0 pinned=2 (Technique II)", "f5", float(w0))]:
    fit = fit_failure_spectrum(spec, K=K, model=model, w0=w0pin, f0=None)
    fits[label] = fit
    LER[label] = np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid)))
    print(f"  {label:34s}: fitted params { {k: round(v,4) for k,v in fit.params.items()} }")

PRIMARY = "f5, free w0"      # Technique I STANDALONE (no Technique-II pin); used downstream

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
meas = fw > 0
ax[0].plot(np.array(spec.weights)[meas], fw[meas], "o", color="steelblue", label="measured f(w)")
ww = np.linspace(w0, max(spec.weights), 200)
ax[0].plot(ww, failure_spectrum_ansatz(ww, a=fits[PRIMARY].a, model="f5", **fits[PRIMARY].params),
           "-", color="crimson", label="f5 ansatz (w0 free, multistart)")
ax[0].plot([w0], [f0_star], "*", color="purple", ms=16, zorder=6,
           label=f"perfect-decoder f0* (Tech II) = {f0_star:.3f}")   # the optimal onset-fraction floor
ax[0].axvline(w0, ls="--", color="darkorange", label=f"onset w0={w0}")
ax[0].set_yscale("log")   # f(w) spans orders of magnitude (and is the space the fit optimizes)
ax[0].set_xlabel("fault weight w"); ax[0].set_ylabel("f(w)"); ax[0].set_title("failure spectrum (log y): measured vs. f5 ansatz"); ax[0].legend(fontsize=8)
for label, ler in LER.items():
    ax[1].plot(p_grid, ler, "-" if label == PRIMARY else "--", lw=2 if label == PRIMARY else 1, label=label)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("p"); ax[1].set_ylabel("LER")
ax[1].set_title("ansatz LER(p): free vs pinned w0"); ax[1].legend(fontsize=8); plt.tight_layout(); plt.show()''')

# ===========================================================================
md(r"""## §3 — Technique III: replica-exchange splitting (`splitting.py`)

Direct Monte-Carlo can't reach very low $p$ (failures become too rare). Splitting estimates
$P(q_\text{low})$ by telescoping a **ladder** of rates,
$P(q_n)=P(q_0)\prod_i P(q_{i+1})/P(q_i)$, where each ratio is a reweighting expectation under a
**failure-restricted** distribution sampled by Metropolis moves. `replica_exchange_estimate` runs
one walker per rate and swaps adjacent rates so configurations diffuse across the ladder.

**Open the hood — the ingredients.** Seeds come from (a) the Technique-II min-weight logicals,
lifted to failing expanded-column configs by `min_weight_logical_seeds`, and (b) typical failures
from a direct-MC **anchor** at the top rate (`_direct_mc_failure_prob`).""")

code('''dec = RelayBPDecoder(); dec.setup_from_matrices(det_mat.T.astype(np.uint8),
                                                probs, obs_mat.T.astype(np.uint8))
# (a) lift the exact L(D) supports to failing seeds (each is a set of expanded columns that fails)
seeds = min_weight_logical_seeds(circuit, col_to_mech, det_mat, obs_mat, dec, supports=list(L_D))
print(f"min-weight seeds that fail under the decoder: {len(seeds)} / {len(L_D)}")
s0 = next(iter(seeds))
synd, truth = _config_syndrome_truth(det_mat, obs_mat, col_to_mech, s0)
print(f"  example seed weight={len(s0)}, fails? {_config_fails(det_mat, obs_mat, col_to_mech, s0, dec)}")

# (b) the direct-MC anchor at the top of the ladder
q_high = q_base * (0.015 / P_REF)
P_anchor, se, mc_seeds = _direct_mc_failure_prob(det_mat, obs_mat, col_to_mech, dec, q_high, 4000, np.random.default_rng(0))
print(f"anchor P(q_high) = {P_anchor:.3e} +/- {se:.1e}  ({len(mc_seeds)} typical failing seeds harvested)")''')

md("""Now the full estimator. For this code (only 4 logical qubits) the single-flip chains
**mix well** — watch the swap-acceptance and the mean-weight ladder.""")

code('''temper, diag = replica_exchange_estimate(
    circuit, RelayBPDecoder(), p_ref=P_REF, p_high=0.015, p_low=1e-4, n_levels=16,
    n_walkers=8, local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000,
    distance=D, seed=2, single_sector=False, mw_supports=list(L_D), verbose=False)
sp = np.asarray(temper.p_ladder)[::-1]; sP = np.asarray(temper.P_logical)[::-1]   # ascending in p
print(f"swap-accept (adjacent rungs): {min(diag['swap_accept']):.2f} .. {max(diag['swap_accept']):.2f}  (healthy ~0.2-0.9)")
print(f"mean weight  hi-q -> lo-q   : {diag['mean_weight'][0]:.1f} -> {diag['mean_weight'][-1]:.1f}  (collapses toward the onset w0={w0})")
print(f"ladder: p {sp[0]:.4f} .. {sp[-1]:.4f}   P {sP[0]:.3e} .. {sP[-1]:.3e}")''')

md(r"""**Where do failures come from? — mean fault weight vs. `p`.** A compact way to *see* the
rare-event principle. At each `p` the logical failures have a weight distribution
`∝ f(w)·C(N,w)·q(p)^w(1-q(p))^{N-w}`; its **mean** is the typical number of simultaneous faults behind
a logical error. As `p→0` it collapses to the onset `w₀=⌈D/2⌉` — failures are dominated by the
fewest-fault events (*fail fast*) — and it climbs as `p` rises and heavier errors start to contribute.
We get it two independent ways and they agree: analytically from the Technique-I ansatz `f(w)`, and for
free from Technique III (the mean weight of the failure-restricted walkers at each ladder rung — the
diagnostic the cell above already reported collapsing toward `w₀`).""")

code('''from scipy.special import gammaln
def mean_failure_weight(fit, p_values, wmax=40):
    """Mean fault weight of logical FAILURES at each p: <w> over f(w)*C(N,w)*q^w*(1-q)^(N-w)."""
    N = fit.n_expanded
    w = np.arange(int(fit.params["w0"]), wmax + 1)             # failures start at the onset w0
    fwa = failure_spectrum_ansatz(w, a=fit.a, model=fit.model, **fit.params)
    log_binom = gammaln(N + 1) - gammaln(w + 1) - gammaln(N - w + 1)
    out = []
    for p in np.atleast_1d(p_values):
        q = np.clip(fit.q_base * p / fit.p_ref, 1e-300, 1 - 1e-15)
        g = fwa * np.exp(log_binom + w * np.log(q) + (N - w) * np.log1p(-q))
        out.append((w * g).sum() / g.sum() if g.sum() > 0 else float(fit.params["w0"]))
    return np.array(out)

mw_ansatz = mean_failure_weight(fits[PRIMARY], p_grid)                          # Technique I
pl, mw_split = np.asarray(temper.p_ladder), np.asarray(diag["mean_weight"])     # Technique III
print(f"mean failure weight:  p=1e-4 -> {mw_ansatz[0]:.2f} (onset w0={w0}),  p=0.013 -> {mw_ansatz[-1]:.2f}")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(p_grid, mw_ansatz, "-", color="crimson", lw=2, label="Technique I: mean weight of failures (ansatz)")
ax.plot(pl, mw_split, "s", color="navy", ms=5, label="Technique III: walker mean weight per rung")
ax.axhline(w0, ls="--", color="darkorange", label=fr"onset $w_0=\\lceil D/2\\rceil={w0}$")
ax.set_xscale("log"); ax.set_xlabel("physical error rate p"); ax.set_ylabel("mean fault weight of failures")
ax.set_title("Kunlun [[18,4,4]] — typical failure weight collapses to the onset as p -> 0")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()''')

# ===========================================================================
md(r"""## §4 — Cross-check: all three vs. brute force

Because the code is tiny, we can compute **direct Monte-Carlo** ground truth at moderate `p` and overlay
it with the three techniques. All of them agree where they overlap: direct MC, the raw importance-sampling
reweighting, the **fitted `f5` ansatz (with `w₀` free — Technique I standalone, robust multistart fit)**,
and replica-exchange splitting. Below the MC reach, Technique I (ansatz) and Technique III (splitting)
extrapolate together along the $p^{w_0}=p^2$ onset slope. The grid runs down to $p=10^{-4}$ — about two
decades below where direct MC is affordable — which is exactly the rare-event regime these techniques
exist for. (The purple line is the Technique-II perfect-decoder onset floor, ~1.5× below — the *decoder*
gap, distinct from the technique agreement.)""")

code('''def direct_mc(p, shots):
    c = BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS)
    d = RelayBPDecoder(); d.setup(c)
    det, obs = c.compile_detector_sampler().sample(shots, separate_observables=True)
    f = np.any(d.decode_batch(det) != obs, axis=1)
    return f.mean(), (max(f.mean(), 1e-9) * (1 - f.mean()) / shots) ** 0.5

# Shot counts kept modest so this cell finishes in a couple of minutes; raise them for tighter bars.
mc_pts = {0.012: 80_000, 0.008: 120_000, 0.005: 200_000, 0.003: 300_000}
mc = {p: direct_mc(p, s) for p, s in mc_pts.items()}

ansatz_LER = LER[PRIMARY]
# raw-IS reweighting (reweight the MEASURED f(w) directly, no fit) over the grid:
raw = importance_sample(circuit, RelayBPDecoder(), p_ref=P_REF, p_values=list(p_grid),
                        weights=list(range(1, 11)), shots_per_weight=6000, seed=1)
raw_LER = raw.P_logical

print("   p        direct-MC        raw-IS      ansatz(f5)    splitting")
for p in (0.012, 0.008, 0.005, 0.003, 0.001, 0.0005, 0.0002, 0.0001):
    mcv = f"{mc[p][0]:.3e}" if p in mc else "   --    "
    print(f"  {p:.1e}   {mcv:>10}   {np.interp(p,p_grid,raw_LER):.3e}   "
          f"{np.interp(p,p_grid,ansatz_LER):.3e}   {np.interp(p,sp,sP):.3e}")

def slope(px, py, lo=1e-4, hi=5e-4):
    return (np.log(np.interp(hi,px,py)) - np.log(np.interp(lo,px,py))) / (np.log(hi) - np.log(lo))
print(f"\\nlow-p slope (expect ~w0={w0}):  ansatz={slope(p_grid,ansatz_LER):.2f}   splitting={slope(sp,sP):.2f}")''')

md(r"""**Figure 10 replica for the Kunlun [[18,4,4]].** Same layout as Fig. 10 of the paper — the
**failure spectrum** (left) and the **logical error rate** (right), for Relay-BP. Following the paper we
fit the f5 ansatz **two ways**: *free*, and with the **onset fixed to the perfect-decoder bound**
`(w₀, f₀*)` from Technique II. As the paper does for its odd-`D` codes (e.g. BB(12)-bplsd, fit for
`w ≥ 4`), the onset-fixed fit **skips the few low weights near onset** (here `w ≥ w₀+2`): there the
decoder gap makes the measured `f(w)` incompatible with an f5 anchored at the low `f₀*` (the fit would
run away), so we anchor the onset at the bound and let the **tail** — where decoders converge toward
saturation — set the shape. The gap between the two f5 curves, on both panels, **is** the decoder gap:
the real decoder (free) sits ~1.5× above what an optimal decoder achieving the onset bound would give.
On the LER panel, Technique I (free ansatz), Technique III (splitting), and direct Monte-Carlo agree
where they overlap.""")

code('''import dataclasses
# Onset-fixed-to-bound f5 fit: pin (w0, f0*) from Technique II, fit the SHAPE only to the tail
# (w >= w0+2), skipping the low weights near onset where the decoder gap makes an f5 anchored at f0*
# ill-posed (it would run gamma1 away). This is the paper's "onset fixed to the bound" curve.
fit_wmin = w0 + 2
_wk = np.array(spec.weights) >= fit_wmin
spec_tail = dataclasses.replace(spec,
    weights=[int(x) for x in np.array(spec.weights)[_wk]],
    failures=[int(x) for x in np.array(spec.failures)[_wk]],
    trials=[int(x) for x in np.array(spec.trials)[_wk]])
fit_bound = fit_failure_spectrum(spec_tail, K=K, model="f5", w0=float(w0), f0=f0_star)
ler_bound = np.asarray(logical_error_rate_from_ansatz(fit_bound, list(p_grid)))

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
ww = np.linspace(w0, max(spec.weights), 200); meas = fw > 0

# LEFT -- failure spectrum: measured f(w), free f5 fit, onset-fixed-to-bound f5 fit, and the f0* star.
ax[0].plot(np.array(spec.weights)[meas], fw[meas], "^", color="navy", ms=7, label="measured f(w) (Relay)")
ax[0].plot(ww, failure_spectrum_ansatz(ww, a=fits[PRIMARY].a, model="f5", **fits[PRIMARY].params),
           "-", color="crimson", lw=2, label="f5 ansatz (free)")
ax[0].plot(ww, failure_spectrum_ansatz(ww, a=fit_bound.a, model="f5", **fit_bound.params),
           "--", color="purple", lw=2, label=f"f5 ansatz (onset fixed to bound, w>={fit_wmin})")
ax[0].plot([w0], [f0_star], "*", color="purple", ms=17, zorder=6, label=fr"onset bound $f_0^*={f0_star:.3f}$")
ax[0].axvline(w0, ls="--", color="darkorange", lw=1, label=f"onset $w_0$={w0}")
ax[0].set_yscale("log"); ax[0].set_xlabel("fault weight w"); ax[0].set_ylabel("f(w)")
ax[0].set_title("failure spectrum"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

# RIGHT -- LER: free f5 ansatz (Technique I), onset-fixed-to-bound f5 ansatz, splitting, direct MC.
ax[1].plot(p_grid, ansatz_LER, "-", color="crimson", lw=2, label="Technique I: f5 ansatz (free)")
ax[1].plot(p_grid, ler_bound, "--", color="purple", lw=2, label="f5 ansatz (onset fixed to bound)")
ax[1].plot(sp, sP, "s-", color="navy", ms=4, label="Technique III: splitting")
mp = sorted(mc); ax[1].errorbar(mp, [mc[p][0] for p in mp], yerr=[mc[p][1] for p in mp],
            fmt="o", color="black", capsize=3, zorder=5, label="direct Monte-Carlo (truth)")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("physical error rate p"); ax[1].set_ylabel("logical error rate")
ax[1].set_title("logical error rate"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

fig.suptitle("Figure 10 replica -- Kunlun [[18,4,4]] with Relay-BP", y=1.02, fontsize=12)
plt.tight_layout(); plt.show()
print(f"onset-fixed-to-bound f5 (w>={fit_wmin}): cost={fit_bound.cost:.2f}, gamma1={fit_bound.params['gamma1']:.1f}; "
      f"LER(bound)/LER(free) at p=1e-3 = {np.interp(1e-3,p_grid,ler_bound)/np.interp(1e-3,p_grid,ansatz_LER):.2f}")''')

# ---------------------------------------------------------------------------
md(r"""## Takeaways

* **Technique II** gave the circuit fault distance `D=3` (note: *below* the code distance 4 — hook
  errors), the exact onset weight `w₀=⌈D/2⌉=2`, the **exact** `|L(D)|` by serial coset enumeration, and
  — via the odd-`D` **Appendix A.6** recipe (`optimal_onset_fraction`, using `L(D)` and `L(D+1)`) — the
  **perfect-decoder onset fraction `f₀*≈0.020`**, the failure-rate floor no decoder can beat.
* **The two techniques together measure a real decoder gap.** Technique II is the *perfect-decoder
  floor*; Technique I is the *real* Relay-BP spectrum. Relay-BP matches the optimal onset **weight**
  (no slope gap — `f(w)=0` below `w₀`), but its onset **fraction** sits ~1.5× *above* `f₀*` — a genuine
  decoder gap, striking precisely **because the code is so small** (a stronger/longer decoder only
  partly closes it). For the gross [[144,12,12]] code the gap is far larger. (This is also why `f₀`
  must be *fitted* in the ansatz rather than pinned to `f₀*`: the real decoder doesn't sit on the floor.)
* **All three techniques agree with direct Monte-Carlo** where they overlap — including the ansatz with
  `w₀` left **free** (Technique I standalone): a robust *multistart* fit lands on the same minimum as
  pinning `w₀` from Technique II (same cost/params), tracks `f(w)`, and matches MC. Below the MC reach,
  Technique I and Technique III extrapolate together along the $p^{w_0}$ onset slope — the payoff of a
  code small enough to verify by brute force. (Caveat: the f5 fit is non-convex; without the multistart a
  single start can fall into a bad basin that only matches the onset — which is why the fit targets
  `log f(w)` and tries several seeds.)

*Generated by `make_tutorial_notebook.py`.*""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "three_techniques_18_4_4.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
