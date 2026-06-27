# Replica-exchange splitting vs importance sampling on bb144 — a direct-MC verdict

**TL;DR.** For the gross code `[[144,12,12]]` (single-sector), the importance-sampling **ansatz** is
the trustworthy logical-error-rate estimate (within ~1.5× of direct Monte-Carlo). The
replica-exchange **splitting** estimate **under-estimates** the LER — 2.5× at p=5e‑3, 13× at p=4e‑3,
worsening as p drops — even after the "bridge fix" that repaired its mixing. Trust the IS ansatz;
treat splitting as a sanity cross-check only.

## What the bridge fix did (and didn't) do

`experiments/bravyi/split_bb144_better.py` + the `gap_weights`/weight-matched-placement changes in
`src/splitting.py` fixed the original run's **mixing pathology**:

- swap-acceptance across all 44 rungs: **0.14 → 0.73–0.91** (the original had one choked 0.14 rung).
- mean-weight ladder: the original's cliff (42→15.6) became a **smooth** descent (83.6 → 10.5).
- the estimate rose ~10⁶ vs the original (the bottleneck was over-suppressing).

These are real improvements to the splitting *machinery* (and would help on codes where splitting
works). But they did **not** make the estimate accurate.

## Direct-MC ground truth (single-sector, same decoder)

| p | direct-MC (truth) | IS-ansatz | new splitting | IS-raw |
|---|---|---|---|---|
| 5e‑3 | **0.0643 ± 0.002** | 0.0424 (0.66×) | 0.0253 (0.39×) | 0.0082 (0.13×) |
| 4e‑3 | **0.0071 ± 0.0003** | 0.0060 (**0.84×**) | 0.00053 (**0.075×**) | 0.00098 (0.14×) |

The IS-ansatz is closest to truth (and is *better* deeper sub-threshold, its design regime). IS-raw's
~7–8× low is just the stride-4 weight under-count. Splitting is low and getting worse.

## Why splitting under-estimates (the mechanism)

The reweight ratio `P(q')/P(q) = E_{π_q}[(q'/q)^|x| ((1-q')/(1-q))^(N-|x|)]` depends on the **within-rung
weight distribution** matching the true failing distribution `π_q(w) ∝ f(w) C(N,w) q^w (1-q)^(N-w)`.
The swap-acceptance and smooth mean-weight are *cross-rung* diagnostics; they are necessary but **not
sufficient**. Comparing the chains' mean weight to the true failing mean weight (from the trusted IS
f(w)) shows the chains are systematically **miscalibrated**:

| p | chain mean-w | true failing mean-w | error |
|---|---|---|---|
| 6e‑3 | 83.6 | 78.7 | +5 |
| 2.9e‑3 | 58.7 | 44.1 | +15 |
| 1.4e‑3 | 43.8 | 26.2 | +18 |
| 4e‑4 | 10.5 | 17.9 | −7 |

The chains sit **+10 to +18 too high** through the mid-p region (then overshoot low at the bottom). A
too-high weight makes `(q'/q)^|x|` too small, so every per-rung ratio under-shoots, and the error
**compounds over 44 rungs** into the observed 2.5–13× underestimate. Root cause: single-flip local
moves cannot equilibrate the weight *within* a rung fast enough at N≈1.8×10⁵ in the available sweeps —
a slow within-rung relaxation that the cross-rung swap diagnostics don't reveal.

This quantitatively confirms the paper's caution (arXiv:2511.15177 §5) that replica-exchange splitting
is an unreliable absolute-LER estimator for codes with many inequivalent logicals / large expanded N.

## Reproduce

- Splitting (bridge-fix): `python experiments/bravyi/split_bb144_better.py` (writes
  `runs/bravyi/bb12/bb144_split_better/splitting.json`; `--pilot` for a quick check).
- IS reference: `runs/bravyi/bb12/bb144_adaptive_1e6/` (the adaptive 1e6-cap sweep).
- Ground-truth MC + the bias probe (per-rung weight vs truth): see the analysis in this directory's
  git history / the conversation that produced this note.

## If revisiting

The diagnosis points at within-rung weight equilibration, not cross-rung mixing. Things that *might*
help (all more compute, none guaranteed to close the gap): far more `local_steps`/sweeps; a
weight-biased / cluster proposal that moves O(many) columns per step; many more rungs so each rung's
weight relaxation is easier. Given the IS ansatz is direct-MC-validated, the pragmatic answer is to
use IS for bb144 and keep splitting as a cross-check.
