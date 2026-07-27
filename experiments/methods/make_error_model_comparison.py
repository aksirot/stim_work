"""Generate error_model_comparison_18_4_4.ipynb — a REPORT over cached runner results.

The notebook no longer runs any simulation. All sampling lives in
``run_error_model_comparison.py``, which caches one JSON per task under
``runs/error_model_comparison_18_4_4/`` (rerunning only what is missing or whose config
changed, and recording per-task wall time). The notebook generated here LOADS those files
and renders the tables and plots: the only computation it does is cheap analysis —
binomial reweighting of the stored spectra, crossings, and error propagation — so it
re-executes in seconds and needs neither stim nor a decoder.

The analysis code itself lives in ``src/emc_report.py`` (one ``Report`` method per code
cell, holding the cross-cell state); the cells generated here are one-line calls into it,
so the notebook reads as prose + results. Edit emc_report.py to change an estimator or a
figure; edit this builder only to change the notebook's structure or markdown.

Contents (unchanged story): §0 schedule, §1 Technique II per channel, §2 Technique I
spectra, §3 splitting, §4 direct-MC overlay, §5 leave-one-out ablations, §6 the
Willow-style budget, §7 the true Λ against [[72,4,8]], §7.5 marginal Λ, §8 the
asymmetric operating point. New in the report: a per-section runner-time table, and the
Λ-share boxes (§7.5, §8) carry propagated standard errors plus a zero-failure-bin bound —
a NEGATIVE share is now printed with the evidence for whether it is real or an estimator
artifact of the under-resolved 72-code onset bins.
"""
import json
from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

# ===========================================================================
md(r"""# Error-model experiments on the Kunlun **[[18,4,4]]** — report

How does the full circuit noise decompose into its parts? Using the three fail-fast techniques (from
`three_techniques_18_4_4.ipynb`) **together with direct Monte-Carlo**, we compare the error models on
the *same* [[18,4,4]] code:

| model | noise kept |
|---|---|
| **full symmetric** | everything — `DEPOLARIZE1` + `DEPOLARIZE2` + `X_ERROR` |
| **CZ only** | `DEPOLARIZE2` (two-qubit `CX` gate depolarizing) |
| **meas only** | `X_ERROR` immediately **before** a measurement `M` |
| **prep only** | `X_ERROR` immediately **after** a reset `R` (state preparation) |
| **gate idle** | `DEPOLARIZE1` on data during the one **CX layer** each data qubit sits out |
| **meas idle** | `DEPOLARIZE1` on data during the **ancilla measure+reset stage** |

Channels are isolated by **filtering noise instructions** on one `symmetric(p)` circuit — every variant
uses the *identical* per-location rate `p` — via `bb_code_sim.filter_noise_channel` (predicates live
next to the circuit builder whose layout they encode). `X_ERROR` splits by position into *prep* (after
`R`) and *meas* (before `M`); `DEPOLARIZE1` is *idle* noise except right after an `H` (the 1q-gate
error), split into `gate_idle` / `meas_idle` by schedule position.

**Idle occupancy.** In the depth-7 schedule each data qubit is idle in **2 of the 8 rounds** per
syndrome cycle — round 7 (all data idle while ancillas are measured/reset) plus exactly one of rounds
0/6. So a data qubit picks up idle `DEPOLARIZE1(p)` **twice per cycle** (`1−(1−p)² ≈ 2p`, not `3p`).
The two slots are **separate channels**: on hardware the measure+reset dead time is much longer than a
gate, so device-faithful duration weighting (t_meas ≫ t_gate) belongs on the `meas_idle` rate.

**This notebook is a report.** Every simulation result is loaded from
`runs/error_model_comparison_18_4_4/`, produced (and cached task-by-task) by
`experiments/methods/run_error_model_comparison.py`. To refresh the data, run that script — it
recomputes only tasks whose cache is missing or whose configuration changed — then re-execute this
notebook (seconds; it needs neither stim nor a decoder).""")

# ---------------------------------------------------------------------------
md("""## Setup — load the cached runner results""")

code(r'''# All loading, estimators, tables and figures live in src/emc_report.py (the
# analysis layer, one method per cell of this report); R holds the cross-cell state.
from emc_report import Report

R = Report()          # loads runs/error_model_comparison_18_4_4 (config manifest + task cache)
R.dem_counts()''')

code(r'''R.runner_times()   # cached per-task wall time, grouped by section''')

# ===========================================================================
md(r"""## §0 — The syndrome-extraction schedule, up close

One cycle of the extraction circuit, layer by layer, for two data qubits (one per block) and one
ancilla of each type — derived from the built circuit itself (by the runner), so it is exactly the
layout the noise-channel predicates key on. Inline `·channel` tags show how each noise instruction is
classified (`cz` / `meas` / `prep` / `gate_idle` / `meas_idle`): each data qubit is busy in six of
the seven CX layers, idles through the one it sits out (`·gate_idle`), and idles again while the
ancillas are measured and reset (`·meas_idle`). The second cell renders the NOISY one-cycle schedule
as a stim `timeline-svg` diagram, sliced to a closed 7-qubit star — one data qubit plus its six check
ancillas, gates kept only when both endpoints are inside — so every rail shown is fully involved and
the labelled noise boxes are readable.""")

code(r'''R.schedule_table()''')

code(r'''R.schedule_svg()   # noise boxes tagged/colored by channel; see the method docstring''')

# ===========================================================================
md(r"""## §1 — Technique II: distance, onset, perfect-decoder floor (per model)

For each model: circuit fault distance `D`, onset weight `w₀=⌈D/2⌉`, the exact `L(D)`, and the
perfect-decoder onset fraction `f₀*`. The four *isolated* channels (CZ / meas / prep / idle) each turn
out to have **even** distance 4 (the code distance) — so `f₀*` is exact via Proposition 1 — while the **full**
model has **odd** distance 3: only *combining* channels makes the weight-3 hook that drops it below the
code distance (Appendix A.6 route for `f₀*`). `L(D)` is enumerated with the ldpc-free half-MITM for even
`D` (robust) and the coset search for odd `D`.""")

code(r'''R.tech2_table()''')

# ===========================================================================
md(r"""## §2 — Technique I: failure-spectrum ansatz (per model)

The runner importance-samples the failure spectrum `f(w)` (adaptive 'hit N failures per weight'
allocation; the weight window is sized per model from the binomial mass at the top of the `p` grid,
so the reweighted curves carry no truncation sag anywhere on the grid) and fits the f5 ansatz (onset
`w₀` left free). The report loads both: the measured spectrum drives every point value below (via
binomial reweighting), the fit supplies the smooth `LER(p)` curves.

**Decoder convention (device calibration).** Every estimator in this report shares ONE decoder
(Relay-BP, `num_sets=20`) whose priors are frozen from the task's noise model built at the
**evaluation point** `p* = 5×10⁻⁴` (`DECODER_P` in the runner) — not at the sampling reference
`p_ref = 0.01`. This is the device convention: a real decoder is calibrated to device rates.
It matters because wrong-coset *pair* explanations scale ∼p² against a true single fault's ∼p —
at p_ref-priors, meas-pair explanations of the ×5 mix's weight-1 CZ hooks were up to 7.5× likelier
than the truth (Bayes-rational misdecodes); calibrated at `p*` the same decoder catches every
single fault (breakeven `p ≈ 1.3×10⁻³`). `f(w)` remains rate-independent because the decoder is
*fixed* — but curves above breakeven (the direct-MC anchors, the threshold crossings) describe a
decoder deliberately miscalibrated for that regime, so expect slightly elevated MC points and
slightly lower pseudo-thresholds than a per-point-calibrated decoder would give.""")

code(r'''R.load_spectra()''')

# ===========================================================================
md(r"""## §3 — Technique III: replica-exchange splitting (per model)

Splitting reaches deep into the rare regime, seeded by each model's exact `L(D)` from §1.""")

code(r'''R.load_splitting()''')

# ===========================================================================
md(r"""## §4 — Direct Monte-Carlo + overlay

Direct-MC ground truth at moderate `p`, overlaid with all three techniques. Per model the ansatz
line, the splitting squares, and the MC circles should coincide where they overlap. (MC only
*validates* the curves — the budget never reads it — so the runner's `MC_SCALE` trades error-bar
width for minutes.)""")

code(r'''R.mc_overlay()''')

# ===========================================================================
md(r"""## §5 — Ablation (leave-one-out): the marginal channel contributions

Google's Willow error budget (arXiv:2408.13687) is built from **marginal** contributions — remove (or
scale) one error component in simulation and measure how much the logical error rate drops. The
channel-*isolated* circuits of §1–§4 cannot provide this: §1 showed every isolated channel has even
distance **4** while the full model has odd distance **3**, so the dominant low-`p` failures are
**mixed-channel** faults that no isolated circuit contains. The runner puts the five **all-but-one**
models (`keep = not channel`) through the same pipeline.

The ablated distances are a structural diagnostic: if dropping channel *i* restores `D=4`, then
channel *i* participates in **every** weight-3 hook; if `D` stays 3, the hooks survive without it.""")

code(r'''R.tech2_abl_table()''')

code(r'''R.load_ablations()''')

# ===========================================================================
md(r"""## §6 — The error budget (Willow-style)

Willow's budget reads `1/Λ ≈ Σᵢ pᵢ/p_th,i` — each channel priced against its own threshold. Two
honesty notes for a **single** code:

* There is no Λ (it is a ratio between code *sizes*); the single-code stand-in for `p_th,i` is the
  channel **pseudo-threshold** — the break-even `p` where `LER_i(p) = p`. A *true* Λ needs the
  same-polynomial `(l,m)=(6,6)` sibling **[[72,4,8]]** (`A=1+x+y²`, `B=1+y+x²`, exact d=8) — §7.
* Two decompositions of `LER_full` are shown, and they differ by construction:
  **isolated** `LER_i/LER_full` (misses mixed faults; the deficit is the **mixing bucket**) and
  **marginal** `1 − LER_{no i}/LER_full` (Google's convention; Σ typically **exceeds 1** because a
  mixed fault is killed by removing *any* of its participant channels, so it is counted in each).
  SPAM = meas + prep (their linear budget terms add).

Point values at `p*` come from the MEASURED spectra, binomially reweighted — NOT from the f5 fits
(independently-fitted extrapolations drift apart model-to-model down here and can even invert the
full-vs-ablated ordering into negative "marginals"). The fits still supply the curve-wide
pseudo-thresholds, where they are anchored. The marginal column carries the propagated binomial
`±σ`. Cross-check column: Technique III splitting (§3 reaches p=1e-4; direct MC is impractical at
LERs of 1e-4..1e-6).""")

code(r'''R.budget_table()''')

code(r'''R.fig_budget()''')

# ===========================================================================
md(r"""## §7 — The true Λ: five channels on the [[72,4,8]] sibling

Everything so far used *pseudo*-thresholds because a real threshold is a crossing between code
sizes. The Kunlun polynomials give a distance-scaled partner: **[[72,4,8]]** at `(l,m)=(6,6)` —
same `A = 1+x+y²`, `B = 1+y+x²`, `k=4`, code distance **8 exact** (both sectors: `w≤7` eliminated by
complete split-MITM, weight-8 logicals exhibited). `d: 4 → 8` is two `+2` steps, the K=4 analog of
Google's `d=3,5,7` ladder.

**Conventions.** Rounds scale with distance (`d/2`: 2 rounds for d=4, 4 for d=8), and Λ compares
**per-round** logical error rates `ε(p) = 1 − (1−LER)^{1/rounds}`: `Λ_i(p) = ε₁₈,ᵢ(p)/ε₇₂,ᵢ(p)`.
The channel crossings `ε₁₈,ᵢ = ε₇₂,ᵢ` are the *true* per-channel thresholds `p_th,i` that the
Willow budget divides by. No exact `f₀*` at this size (the `L(D)`/`L(D+1)` enumerations are the
bb144-regime problem) — the Λ budget doesn't need it.

**Cost.** The 144-qubit sweeps are the expensive part of the runner (order an hour+), which is
exactly why they are cached per task: a re-run of `run_error_model_comparison.py` touches them only
if their config changed. `--boost72` buys the 72-code spectra 2× target failures and 5× shots —
see the §7.5 box for when that is worth it.""")

code(r'''R.tech2_72_table()''')

code(r'''R.load_spectra_72()''')

md(r"""### The measured failure spectra — every one of them, both codes

Every point value, budget fraction, and Λ in §6–§8 is a binomial reweighting of a sampled
spectrum, and the spectra are where estimator pathologies hide — so *all* of them are plotted
here (6 models + 5 leave-one-out mixes + 6 asymmetric mixes, per code). Reading guide: the
dashed vertical marks the full model's **predicted onset** `w₀ = ⌈D/2⌉` (computed from the tech2
circuit distance `D`) — the lowest weight at which even a **perfect** minimum-weight decoder can be
forced to misdecode. Dots *at or right of* it are **decoder-in-the-loop failures** that are
irreducible coset ambiguity (any decoder can be forced to misjudge them); dots *left of* `w₀` are
sub-onset decoder artifacts, and this report has eliminated both
known kinds — BP symmetry traps (fixed by `num_sets=20`) and prior-mismatch misdecodes of
trace-free hooks (fixed by calibrating the decoder at `p*` instead of `p_ref` — see the §2
decoder-convention note). A dot reappearing below `w₀` after a decoder or budget change is a
regression in one of those two categories. A **★ star** marks Technique II's *exact* perfect-decoder
onset fraction `f₀*` at `w₀` (`Prop.1` for even `D`, `App.A.6` for odd) — the measured onset dot sits
*above* it by the decoder's excess miscorrection at the onset. The ×5 asymmetric rays carry their
**own** `f₀*` (`compute_f0_asym_18.py`): `D`, `w₀` and the failing *sets* are ray-independent, but the
onset *fraction* is not — mechanisms enter the expanded-uniform measure with rate-proportional
multiplicities, so the ×5 ray re-weights the same failing sets (≈1.6–2.2× the symmetric values here).
The **[[72,4,8]]** onset fraction is
**unpinned** (no exact `L(D)` at 144 qubits), so those panels instead carry a rigorous **lower bound**
`f₀* ≥ f0_lower` (open **▲**, from a partial-`L(D)` search — `compute_f0_lower_72.py`, and
`compute_f0_lower_asym_72.py` for the ×5 mixes, whose bound is re-weighted by the ×5 multiplicities;
for the *ablated* ×5 mixes even `D` is a BP-OSD upper bound computed there) alongside the
onset-**weight** line; the measured onset dot is the matching upper bound, so f₀* lies between the ▲
and the first dot (the bound is loose — it tightens only with far more enumerated logicals). The thin dashed curves are the **f5 ansatz** (Eq. 10
of the paper; stored fits for the models, fitted here on the cached bins for the mixes): it is
identically zero below its fitted `w₀`, so a dot the dashed curve structurally cannot reach is
outside the ansatz family — the decoder-floor signature at a glance. **× marks** are sampled
bins with zero observed failures (drawn at `1/(2T)`): empty low-`w` bins are what the §7.5/§8
truncation interval prices at `3/T`, so watch them whenever budgets change. The alternating
high-`w` gaps are the stride-2 tail sampling (see `emc_report.fill_spectrum`).""")

code(r'''R.fig_spectrum_grid()   # every spectrum §6-§8 reweights, eyeballed in one figure''')

code(r'''R.lambda_table()''')

code(r'''R.fig_74()   # bold = reweighted measured, faint = f5 fit (drifts at low p)''')

# ===========================================================================
md(r"""## §7.5 — Marginal Λ: the ablations on the larger code

§7.3's Λᵢ used channel-*isolated* circuits — the clean per-channel suppression capability, but not
a decomposition of Λ_full (the mixed-channel faults belong to no isolated circuit). Google's
budget convention is *marginal*: ablate one component from the FULL circuit and measure the
change. The runner runs the five leave-one-out circuits on **[[72,4,8]]** (the 18-code ablations
are §5's); we form `Λ_no-i(p*) = ε₁₈,no-i/ε₇₂,no-i` and read channel i's contribution to the full
suppression deficit as `1/Λ_full − 1/Λ_no-i`. The gap between Σ(contributions) and `1/Λ_full` is
the Λ-space mixing — the same story §6 tells in LER-space, now for error suppression.

**Reading the signs.** Each contribution is a *difference of two noisy ratios*, so this box now
carries the full uncertainty budget: `±σ` propagates the binomial errors of all four spectra, and
the `[lo, hi]` interval additionally spans the **zero-bin truncation** — every sampled-but-empty
weight bin priced at its rule-of-three upper bound `f(w) < 3/T`. That pricing deliberately
includes the bins *below* tech2's D: D is the perfect-decoder floor, but the measured spectra
show the actual decoder **miscorrects below it** (see the spectrum figure in §7 — `f(1) > 0` on
the full 18-code mix under a p_ref-calibrated decoder), so an empty low-`w` 72-code bin is
unresolved statistics, not a structural zero — and at low `p*` the reweighting is most sensitive
to exactly those bins. A negative share is only *real* (a channel whose faults the big code
handles better than the small one) if it stays negative within both; otherwise it is an estimator
artifact of the under-resolved [[72,4,8]] low-weight bins. `--boost72` tightens the onset
statistics; the truncation interval, however, shrinks only with more trials on the empty low-`w`
bins themselves (3/T), not with more onset failures.""")

code(r'''R.load_ablations_72()''')

code(r'''R.lambda_box_sym()''')

# ===========================================================================
md(r"""## §8 — A second operating point: meas & meas-idle ×5

Everything above is a **sensitivity analysis at one base point in rate-space** — the symmetric ray
where every channel runs at the same p. The budget fractions are components of a gradient of
`1/Λ` (and of LER), and gradients depend on where you evaluate them. Real devices don't sit on
the symmetric ray: measurement error and the measure/reset dead-time idle typically run several
times hotter than gates. Here we re-evaluate at a device-like point — **meas and meas_idle at
5×p, everything else at p** (`scale_noise_channels`) — and redo the marginal budget and Λ.

Two economies: (i) the **isolated** curves need no resampling — `f(w)` is rate-independent, so
channel i at rate `rᵢ·p` is just `reweight_spectrum(spec_i, [rᵢ·p])`; only the full and ablated
*mixes* (whose channel composition changes) are new sweeps. (ii) Ablating channel i from the
asymmetric mix composes the existing tools: `filter_noise_channel(scale_noise_channels(...))`.
Convention: `p` remains the base rate of the un-boosted channels, so `p*` comparisons across
§6/§8 are at equal gate noise. The Λ box carries the same `±σ` / zero-bin verdicts as §7.5.""")

code(r'''R.load_asym()''')

code(r'''R.budget_box_asym()''')

code(r'''R.lambda_box_asym()''')

# ---------------------------------------------------------------------------
md(r"""### §8.4 — the §7.4 panels on the ×5 ray

The point-decomposition table above is noise-limited at `p* = 5e-4`; the *curves* are not — away
from the deep-suppression regime the same cached spectra resolve the ×5 story cleanly. This is the
§7.4 figure remade at the device-like point, with **no new sampling**: the full mix reweights its
own asymmetric spectra, and each isolated channel reweights its §2/§7.2 spectrum at its **own**
rate `rᵢ·p` (the x-axis stays the base rate `p` of the un-boosted channels, the §6/§8 convention).
Boosted channels are drawn only while `rᵢ·p ≤` the grid top their weight windows were sized for —
beyond that the reweighted curve sags from window truncation, not physics. Curves come in §7.4's
dual-estimator style: **bold** = reweighted measured spectra (lower-bound caveat at very low `p`),
**faint** = the f5-fit estimator on the same ray — isolated channels reuse their stored §2/§7 fits
shifted to `rᵢ·p` (same fit, scaled abscissa), while the full ×5 mix is fitted here on its cached
asym bins (the runner stores no fit for the asym sweeps). Where bold and faint split at low `p` is
the ansatz-vs-decoder-floor break: the fit is identically zero below its fitted `w₀`, the
reweighted sum keeps the measured sub-onset bins. The crossings table recomputes the symmetric-ray
values with the reweighted estimator for an apples-to-apples comparison.""")

code(r'''R.fig_84()   # the §7.4 panels on the ×5 ray, same bold/faint convention''')

md(r"""### §8.5 — the marginal curves: which channel limits what, where (both rays)

§8.4's panels show channel-**isolated** curves (intrinsic capability; their crossings are the
per-channel thresholds). These are the leave-one-out **marginal** curves — the §7.5/§8 budget and
Λ boxes turned into functions of `p`, entirely from cached ablation spectra (no new sampling) —
drawn for **both operating points**: the first figure is the **symmetric ray** (the §5/§7.5
ablations), the second the **×5 ray** (the §8 asymmetric ablations). Left panels: the LER marginal
fraction `1 − LER_no-i/LER_full` per code (solid [[18,4,4]], dashed [[72,4,8]]). Right panels:
each channel's share of the suppression deficit, `(1/Λ_full − 1/Λ_no-i)/(1/Λ_full)`. Read the two
figures together — they are the gradient of the budget at the two base points, and their
difference is the §8 rotation story at a glance: on the symmetric ray CZ leads everywhere; on the
×5 ray the small code's error rate turns meas-type-limited (meas & meas-idle take over the left
panel's solid curves) while the big code — and with it the Λ deficit — stays CZ-led (right
panel). Curves get noisy where the 72-code spectra are onset-limited (low `p`, dashed curves and
the right panels); the §7.5/§8 boxes at `p*` carry the honest `±σ` for exactly these quantities —
a share is only *real* if it verdicts solid there. The dotted vertical line marks `p*`.""")

code(r'''R.fig_85()   # marginal (leave-one-out) curves, symmetric ray then ×5 ray''')

# ---------------------------------------------------------------------------
md(r"""## Takeaways

* **The full-circuit distance-3 reduction needs *mixed* error types.** `CZ only`, `meas only`,
  `prep only`, and `idle only` are each distance **4** (the code distance); only *combining* channels
  produces the weight-3 hook that drops the full model to `D=3`.
* **Two-qubit (CZ) gate errors dominate** the logical error rate; the single-location channels
  (**measurement, preparation, idle-data**) are each far smaller — fewer DEM mechanisms and much lower
  LER — and the full LER sits well above the sum of the isolated channels, i.e. the mixed-type faults
  carry most of it.
* **Preparation vs. measurement:** reset and measurement bit-flips are the same `X_ERROR` at the same
  rate, but split by circuit position they can contribute differently — compare the `prep only` and
  `meas only` curves.
* **The techniques agree with direct MC for every model** and extrapolate together into the rare regime:
  the fail-fast toolkit works channel-by-channel, not just on the full model.
* Both onset routes appear in one comparison: the isolated channels are **even distance 4** (exact `f₀*`
  via Proposition 1), the full model is **odd distance 3** (Appendix A.6).
* **Isolated ≠ marginal (§5–§6).** The channel-only circuits miss the mixed-type faults entirely, so
  their fractions under-count and leave a **mixing bucket**; the leave-one-out marginals (Google's
  convention) count each mixed fault once per participant, so they **over-count** (Σ > 1). The gap
  between the two decompositions *is* the mixed-fault structure — read it together with the ablated
  distances, which name the channels every weight-3 hook needs.
* **Pseudo-thresholds (§6) vs true thresholds (§7).** §6's break-even points need only one code; §7
  replaces them with the real thing — the `ε₁₈,ᵢ(p) = ε₇₂,ᵢ(p)` crossings against the same-polynomial
  sibling **[[72,4,8]]** — and reads off `Λ_i(p)` directly. With `d: 4→8` (two `+2` steps) the
  per-step suppression is `λ = √Λ_full`.
* **The Willow identity `1/Λ ≈ Σᵢ p/p_th,i` is the §7 punchline**: how far the sum falls short of
  `1/Λ_full` measures exactly how much the mixed-channel faults (the §1 `D=3` hook, invisible to
  every isolated channel) break the additive budget.
* **Λ shares are differences of noisy ratios — trust only flagged-solid signs (§7.5, §8).** Each
  marginal contribution now carries a propagated `±σ` and a zero-bin truncation interval; a negative
  share that is `~0 within 2σ` or `not robust to zero-bin truncation` is an artifact of the
  under-resolved 72-code onset bins, not physics. `--boost72` in the runner tightens exactly those
  spectra (and the cache means nothing else reruns).
* **The whole budget is a gradient at a base point (§8).** Re-evaluating at a device-like ray
  (meas + meas-idle ×5) shows how the decomposition rotates with the noise mix. The isolated
  basis reweights analytically to any rate vector (`f(w)` is rate-independent); only the full and
  ablated mixes need resampling — which is what makes multi-point sensitivity maps affordable.

*Report generated by `make_error_model_comparison.py`; data by `run_error_model_comparison.py`
(cached per task under `runs/error_model_comparison_18_4_4/`).*""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "error_model_comparison_18_4_4.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
