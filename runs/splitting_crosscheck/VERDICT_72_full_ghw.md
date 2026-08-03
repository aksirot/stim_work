# Splitting at 72-code scale — FINAL VERDICT (amended 2026-08-03): the PAPER'S method
# passes arbitration; our tempering shortcut is what failed

**Paper-faithful arbitration (72_full_paper.json, Alg 2/3 + Eq.18 + σ+Δ, ghw_deep,
8e-3→2e-3, L=4 M=2 T=2000/level): PASS at z=1.3.** Point estimate 2.46e-6 vs MC
7.33e-5 (~30x low) but the across-instance spread (±6.3e-6, the paper's prescribed
error bar) honestly covers the truth — under-sampling fails LOUDLY. Mechanism
confirmed: the fine ladder + typical-config warm starts held mean weight ~24-25 at
every level (no collapse to minimal-weight cores; tempering slid to ~5). The 30x
residual bias is T=2000/level vs the paper's T~1e6. Caveats: the σ+Δ controller
reported convergence anyway (its criterion does not certify accuracy — the
multi-instance spread is NOT optional), and tight bars at this scale cost ~1e7
decodes for a 0.6-decade window (days on rodan; prohibitive at idle decode rates).
CONCLUSIONS: (1) arXiv:2511.15177 Technique III is sound as published; (2) the
replica-exchange/tempering variants (the paper's future-work idea, our v1+v2) are
structurally biased at scale and remain condemned below; (3) IS+top-up still carries
all campaign low-rate numbers on cost; idle-floor pricing by faithful splitting is
possible in principle at a stated (high) price.

# (superseded framing) Splitting ladders at 72-code scale — retired for probabilities (2026-08-02)

**v2 UPDATE (72_full_v2_ghwdeep.json): the ergodicity-hardened rebuild ALSO FAILS,
and more instructively.** Library-diverse seeds (54 verified ghw_deep failures),
3 independent ladders, ghw_deep decoder: ladder(2e-3) = 5.3e-11 vs its own MC anchor
7.3e-5 (z=33, SIX decades low — worse than v1's 1.6) while the three ladders agreed
to 0.34 log10. Conclusions: (1) cross-ladder spread is NOT a certificate — identically
structured ladders share the bias; (2) the root cause is MISSING ENTROPY, not missing
basins: low rungs collapse to the lightest configs (mean-w -> 5) because individual
light configs dominate π pointwise, but the rung terms need the weight DISTRIBUTION,
whose mass is the astronomically many heavier failing configs — a handful of walkers
cannot represent that multiplicity and the estimator has no volume correction;
(3) the coset-jump move is inert as built (0/7650 accepted — logical jumps exit the
failing set; weight pre-accept kills the rest). A viable rebuild would need a
density-of-states estimator (Wang–Landau-style flat histogram over weight) — a
research project, not a rerun. SPLITTING-FOR-PROBABILITIES IS RETIRED for this
campaign; IS + top-up (twice-validated) carries all low-rate numbers; the
descent/harvest machinery keeps its existence duties (onset bounds, distance certs,
libraries), where it keeps winning.

# 72_full_ghw splitting ladder — v1 VERDICT: biased low at scale, do not quote (2026-08-01)

The full-budget ladder (16 rungs, 8 walkers, 80 sweeps, 15.5 h) PASSES every internal
honesty gate — swap acceptance 0.49–0.79 on all adjacent pairs, mean weight monotone
28→8, "full ladder quotable" — and is nevertheless WRONG by orders of magnitude:

| p | splitting | reweighted IS | direct MC |
|---|---|---|---|
| 5e-3 | 1.98e-2 | 1.34e-2 (z=2.6) | — |
| 2e-3 | 1.69e-6 | 6.34e-5 | **14/200k = 7.0e-5** (final) |
| 5e-4 | 1.89e-14 | 1.42e-7 | — |

The direct-MC arbitration at p=2e-3 (`arbiter_mc.log`, ghw decoder, seed 808) confirms
reweighted IS to within z≈0.4 (14 observed vs 12.7 predicted). Splitting's prediction
of 0.34 expected events against 14 observed is excluded at ~1e-16. The same code path passed a 3-way agreement (splitting/IS/MC,
worst z=2.6) on the [[18,4,4]] code — this failure is scale-dependent, not a bug in
units or conventions.

**Diagnosis** (consistent with the earlier 8x tiny-budget bias, compounded): 8 walkers
× 80 sweeps cannot equilibrate the failing-set VOLUME of a ~46k-mechanism circuit.
Replicas collapse into a few low-weight cores (w_min_seen=5; bottom-rung mean weight
8.2) and the rung-to-rung ratio chain then estimates the probability of that basin,
not of the full failing set — biased low, worse at every rung down the ladder. Swap
acceptance and mean-weight monotonicity certify LOCAL exchange health, not coverage:
necessary, not sufficient.

**Consequences**
- Do not quote this ladder (or any same-budget splitting on 72-code-scale circuits).
  The A/B notebook's §4 splitting overlay must carry this verdict.
- The idle-chain ladders (camp: one adjacent pair frozen at swap 0.00) cannot price
  the idle sub-onset floor at these budgets. Kill after collecting camp diagnostics.
- Fix directions, in order of promise: (1) batched-proposal decode (lockstep across
  walkers×rungs, ~5-10x more sweeps per wall-hour), (2) many more walkers (volume
  coverage), (3) a volume-aware estimator across rungs, (4) always pair splitting
  with an IS/MC overlap point in the measurable regime as an acceptance test —
  internal gates alone are not a certificate.
- UNAFFECTED: onset-hunt mode + distance bounds (existence certificates, not
  probabilities): 18-code D≤3 exact, 72-code D≤8 exact; w_min_seen/min_config_mechs
  diagnostics; harvest+strip machinery (feeds the decoder-loop library).
