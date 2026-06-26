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
                        _mw_init, _mw_coset_enum_task)
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
md(r"""## §1 — Technique II: distance, onset weight, and the exact `L(D)`  (`min_weight.py`)

We do this **first** because it pins the onset that Technique I needs. Two `src` calls:

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

# ===========================================================================
md(r"""## §2 — Technique I: the failure-spectrum ansatz (`importance_sampling.py`)

Idea (paper §3): the logical error rate is a reweighting of the **failure spectrum** `f(w)` — the
fraction of weight-`w` fault configurations that the decoder gets wrong:

$$\mathrm{LER}(p) \;=\; \sum_w f(w)\,\binom{N}{w}\,q(p)^w\,(1-q(p))^{N-w},\qquad q(p)=q_\text{base}\,p/p_\text{ref}.$$

**Open the hood — the expanded representation.** `_parse_dem` gives the per-mechanism arrays;
`_expand` replicates each mechanism into identical columns so all `N` expanded columns share one base
rate `q_base`. Sampling a weight-`w` config = choosing `w` of the `N` columns.""")

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

md("""That inline loop **is** `importance_sampling._sample_failures_at_weight`. The library
`importance_sample` does it for a whole band of weights at once and returns a `FailureSpectrum`.""")

code('''is_res = importance_sample(circuit, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                           weights=list(range(1, 11)), shots_per_weight=6000, seed=1)
spec = is_res.spectrum
fw = np.array([F/T for F, T in zip(spec.failures, spec.trials)])
print("w   :", list(spec.weights))
print("f(w):", [round(x, 4) for x in fw])
print(f"\\nf(1)={fw[0]:.3f} (below onset), f(2)={fw[1]:.3f} = the onset fraction; f(w) -> saturation as w grows.")''')

md(r"""**Fitting the ansatz — and why Technique II matters.** `fit_failure_spectrum` fits a smooth
3–5 parameter form to `f(w)`. If we let it fit the onset weight `w₀` freely it drifts and the
reweighted `LER` comes out ~1.5× low. **Pinning `w₀=2` from Technique II** (exact, even for odd D)
anchors the fit — and with the richer `f5` form the `LER` then matches direct Monte-Carlo (next
section). `f₀` is still *fitted* (odd D).""")

code('''p_grid = np.geomspace(0.0015, 0.013, 30)
fits, LER = {}, {}
for label, model, w0pin in [("f3, free w0", "f3", None),
                            ("f5, free w0", "f5", None),
                            ("f5, w0 pinned=2 (Technique II)", "f5", float(w0))]:
    fit = fit_failure_spectrum(spec, K=K, model=model, w0=w0pin, f0=None)
    fits[label] = fit
    LER[label] = np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid)))
    print(f"  {label:34s}: fitted params { {k: round(v,4) for k,v in fit.params.items()} }")

PRIMARY = "f5, w0 pinned=2 (Technique II)"      # the one we trust; used in the cross-check

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
meas = fw > 0
ax[0].plot(np.array(spec.weights)[meas], fw[meas], "o", color="steelblue", label="measured f(w)")
ww = np.linspace(w0, max(spec.weights), 200)
ax[0].plot(ww, failure_spectrum_ansatz(ww, a=fits[PRIMARY].a, model="f5", **fits[PRIMARY].params),
           "-", color="crimson", label="f5 ansatz (w0 pinned)")
ax[0].axvline(w0, ls="--", color="darkorange", label=f"onset w0={w0}")
ax[0].set_xlabel("fault weight w"); ax[0].set_ylabel("f(w)"); ax[0].set_title("failure spectrum"); ax[0].legend()
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
    circuit, RelayBPDecoder(), p_ref=P_REF, p_high=0.015, p_low=0.0015, n_levels=10,
    n_walkers=8, local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000,
    distance=D, seed=2, single_sector=False, mw_supports=list(L_D), verbose=False)
sp = np.asarray(temper.p_ladder)[::-1]; sP = np.asarray(temper.P_logical)[::-1]   # ascending in p
print(f"swap-accept (adjacent rungs): {min(diag['swap_accept']):.2f} .. {max(diag['swap_accept']):.2f}  (healthy ~0.2-0.9)")
print(f"mean weight  hi-q -> lo-q   : {diag['mean_weight'][0]:.1f} -> {diag['mean_weight'][-1]:.1f}  (collapses toward the onset w0={w0})")
print(f"ladder: p {sp[0]:.4f} .. {sp[-1]:.4f}   P {sP[0]:.3e} .. {sP[-1]:.3e}")''')

# ===========================================================================
md(r"""## §4 — Cross-check: all three vs. brute force

Because the code is tiny, we can compute **direct Monte-Carlo** ground truth at moderate `p` and
overlay it with the three techniques. The four should agree where they overlap; below the MC reach,
Technique I (ansatz) and Technique III (splitting) extrapolate together along the $p^{w_0}=p^2$ onset
slope.""")

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
for p in (0.012, 0.008, 0.005, 0.003, 0.002):
    mcv = f"{mc[p][0]:.3e}" if p in mc else "   --    "
    print(f"  {p:.4f}   {mcv:>10}   {np.interp(p,p_grid,raw_LER):.3e}   "
          f"{np.interp(p,p_grid,ansatz_LER):.3e}   {np.interp(p,sp,sP):.3e}")

def slope(px, py, lo=0.0015, hi=0.004):
    return (np.log(np.interp(hi,px,py)) - np.log(np.interp(lo,px,py))) / (np.log(hi) - np.log(lo))
print(f"\\nlow-p slope (expect ~w0={w0}):  ansatz={slope(p_grid,ansatz_LER):.2f}   splitting={slope(sp,sP):.2f}")''')

code('''fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.plot(p_grid, raw_LER, "-", color="seagreen", lw=1.5, label="Technique I: raw IS (reweight measured f(w))")
ax.plot(p_grid, ansatz_LER, "-", color="crimson", lw=2, label="Technique I: f5 ansatz (w0 pinned by Tech II)")
ax.plot(sp, sP, "s-", color="navy", ms=4, label="Technique III: replica-exchange splitting")
mp = sorted(mc); ax.errorbar(mp, [mc[p][0] for p in mp], yerr=[mc[p][1] for p in mp],
            fmt="o", color="black", capsize=3, zorder=5, label="direct Monte-Carlo (ground truth)")
# onset asymptote ~ p^w0 anchored at the lowest splitting point
ax.plot(p_grid, sP[0] * (p_grid / sp[0]) ** w0, ":", color="gray", label=fr"onset slope $\propto p^{{w_0}}$ ($w_0={w0}$)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate")
ax.set_title("Kunlun [[18,4,4]] — three techniques vs. direct Monte-Carlo"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------------------
md(r"""## Takeaways

* **Technique II** gave the circuit fault distance `D=3` (note: *below* the code distance 4 — hook
  errors), the exact onset weight `w₀=⌈D/2⌉=2`, and the **exact** `|L(D)|` by serial coset
  enumeration — all checkable because the code is tiny.
* **Technique II anchors Technique I.** Pinning `w₀` from the distance calculation is what makes the
  ansatz `LER(p)` agree with brute force; a free-`w₀` fit underestimates. (`f₀` stays fitted here
  because `D` is odd — Proposition 1's exact onset fraction needs even `D`; the BB6 code is such a
  case.)
* **All three techniques agree with direct Monte-Carlo** where they overlap, and Technique I /
  Technique III extrapolate together along the $p^{w_0}$ onset slope below the MC reach — the payoff
  of a code small enough to verify by brute force.

*Generated by `make_tutorial_notebook.py`.*""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "three_techniques_18_4_4.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
