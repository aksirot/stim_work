# Decoder-selection loop — final recommendation

## UPDATE 2026-08-01: superseded by `pre640_sets1200` — VERIFIED 0 sub-onset failures

After the rodan device-top-up specimens landed (124 channel-harvested failure modes,
perfect transfer: ghw fails 124/124 locally as recorded), nb_pre_x2 failed 52 of them
and the search reopened per the user's criterion: a verified decoder with ZERO
sub-onset failures. The w3 certification sweep (`w3_certify.py`) found the two-knob
family pre640×sets600 down to a SINGLE resistant specimen ([51,53,334], prep-harvest);
the escalation probe showed nearly any further knob fixes it; of the four merges only
**pre640_sets1200** certified clean on all 90 w=3 entries (double-pass, 0 flaky).

**Verification (`verify_pre640_sets1200.py`): PASS with a strict win** — paired 10k
shots vs nb_pre_x2 at w=10/14: b01=3, b10=0 (fixes 3 the rival misses, regresses on
none). Full 524-entry library: 470 fixes, risk 1.10e-2 (best ever benched), w3 0/90.
Throughput 24.2 dec/s at typical weights — FASTER than nb_pre_x2 (12.3/s) and ghw
(14/s): the 1200 sets engage only on hard syndromes (stop_nconv early exit), so the
ensemble is nearly free in expectation.

**FIELD CONFIRMATION 2026-08-03 (runs/probe_w3_meas_ghwdeep.json)**: the library
certification generalizes off-library. A standalone 3e6-shot w=3 bin on the MEAS-ONLY
model (ghw's worst sub-onset channel) under device-calibrated ghw_deep returned
**0 / 3,000,000** — f(3) < 1.0e-6 (rule of three), versus ghw's measured 10/3.01M =
3.3e-6. Under ghw's rate the probability of seeing zero is e^-10 ~ 5e-5, so this is a
>3.3x improvement at 95% and a decisive rejection of rate-parity. 45 min at 72 threads
(~1100 dec/s: clean syndromes never engage the 1200-set ensemble).

**CONSTRUCTIVE CONFIRMATION 2026-08-03 (runs/decoder_loop/tech2_subset_seeds.json)**:
Technique-II-seeded candidates — every subset of 3 verified weight-8 circuit logicals,
enumerated EXHAUSTIVELY, no sampling. At w=3 (complement heavier, so a failure is an
unambiguous decoder defect): baseline 12/168, **ghw 1/168, ghw_deep 0/168**. This is
the first evidence separating ghw from ghw_deep on structure-derived rather than
harvested candidates, and it agrees with the loop's ranking. At w=4 (exact D/2 tie,
where no decoder can be perfect): baseline 139/210 = 66%, ghw 110/210 = 52%,
ghw_deep 108/210 = 51% — ghw_deep tie-breaks at essentially chance while baseline is
markedly worse than chance, i.e. systematically drawn to the wrong coset.

**RECOMMENDED: `pre640_sets1200`** = ghw lineage with pre_iter=640, num_sets=1200
(gamma0=0.0625, set_max_iter=120, gamma interval (−0.5, 1.0), stop_nconv=5).
Same caution as all wide-interval configs: 72-code only; 18-code keeps baseline
(system-level Λ convention). Campaign promotion into DECODER_VARIANTS remains a
human decision. The v2 splitting production should run with this decoder.

---

# Original closing report (2026-07-31)

Six ~1 h iterations on the [[72,4,8]] full-symmetric model (branch
`ghw-decoder-campaign`; driver `experiments/methods/decoder_loop.py`; state and
per-iteration JSON in this directory). The loop is CLOSED: iterations 4–6 plateaued
(every paired verify statistically even, promotions on small library margins), and the
deadline-free closing bench (`decoder_loop_final_bench.py` → `final_bench.json`)
settled the standings on the full library.

## Recommendation

**`nb_pre_x2` — the ghw config with `pre_iter` doubled to 640** (gamma0=0.0625,
num_sets=200, set_max_iter=120, gamma_dist_interval=(−0.5, 1.0), stop_nconv=5):

| config | fixes/356 | risk (LER proxy) | w=3 fails | rate |
|---|---|---|---|---|
| **nb_pre_x2** | 314 | **1.12e-2** | **0/64** | 12.3/s |
| nb_g0_x2 | 322 | 3.33e-2 | 0/64 | 8.6/s |
| heavy | 317 | 4.74e-2 | 5/64 | 31.8/s |
| baseline | 292 | 5.15e-2 | 7/64 | 49.4/s |
| ghw | 286 | 6.26e-2 | 0/64 | 14.0/s |
| heavy+nb_g0_x2 (portfolio) | 330 | 3.37e-2 | 0/64 | 5.4/s |

Risk = Σ over failed entries of P(W=w at p*=5e-4)/n(w) — the library-restricted LER
proxy. `nb_pre_x2` is 3–6× below everything else, keeps ghw's perfect sub-onset record
(0/64 at w=3), and its 10k-shot paired verify vs ghw-lineage incumbents showed no
above-onset regression. Cost: ~4× baseline decode time (~2× ghw). The fixes-count
leaders (nb_g0_x2, portfolios) lose on risk: their extra fixes sit in the w≥12
ambiguity band worth little LER; their misses include costlier low-w entries.

Promotion into campaign `DECODER_VARIANTS` stays a human decision (this loop
recommends). Suggested variant name: `ghw_deep`.

## Promotion chain (why each transition happened)

1. **ghw** (start) → survived iters 1–2 (iter-1 nb_pre_x2 "promotion" was a 0-shot
   verify bug, reverted; gate hardened: MIN_SHOTS=2000 + per-challenger box slices).
2. **heavy** (iter 3, flat-fixes era) — library margin 18 under the count metric;
   later shown to fail 5/64 w=3 sub-onset entries the count metric under-weighted.
3. **nb_pre_x2** (iter 5) — after the w=3 harvest fold-in + risk weighting; full-budget
   paired verify, b01=b10=1, strictly better library, clears every w=4 entry.
4. **nb_g0_x2** (iter 6) — promoted on a SUBSAMPLED bench whose tail draw missed the
   entries dominating its risk; the full-library closing bench reverses the ordering
   (1.12e-2 vs 3.33e-2). Paired test between them was even, so the reversal is a
   library-evidence call, not a contradiction of measured LER.

## Library (permanent regression suite — `library.json`)

356 entries, w=3–16, all mechanism-support sets on the full-symmetric DEM:
- 292 generated by the nc1 fast-generator harvest + lockstep strip + plateau-escape
  descent (descent reached the TRUE onset w₀=4 with no knowledge of it — the
  code-agnostic census design validated on a checkable code).
- 64 mapped from the per-model w=3 sub-onset harvest (footprint-matched into the
  full-symmetric DEM; `add_w3_harvest_to_library.py`).
- Incumbent top-up failures from three different reigning decoders.
- Census: ≥10 entries at w ≤ ŵ+1 held from iteration 2 on (66 at w ≤ 4 at close).

Certified onset upper bounds (bounds-only semantics): every config that fails a w=3
entry has onset_ub=3; nb-family configs fix all w=3/w=4 entries → onset_ub=5 against
this library. The rodan full-symmetric top-up independently shows ghw-family decoders
have ≥1 real w=3 failure in 3M shots, so onset_ub=3 is the true statement for the
family; the library simply hasn't captured that specimen yet.

## Open items

- **Specimen feed**: tonight's rodan device-dir top-up records failing configs
  (w=2–10, ≤200/weight) per sub-model; under EMC_CALIB=device they transfer exactly —
  fold into this library (`add_w3_harvest_to_library.py` pattern, no re-verify needed).
- **nb_pre_x2 on the 18-code**: it inherits ghw's wide gamma interval, which is
  measured-broken there — the system-level (per-code) Λ convention stands. If a shared
  decoder is wanted, bench `heavy` (narrow interval) on the 18-code instead.
- **Portfolio with nb_pre_x2 as a member** was never benched (combos were built around
  the then-incumbent); oracle ceilings suggest ≤ modest gains at ≥2× cost.
- **Λ regeneration**: if nb_pre_x2 is promoted, the 72-code campaign spectra deserve a
  regeneration under it (new EMC_RESULTS; the 9.9× ghw improvement at p* is the floor,
  nb_pre_x2's measured LER should be same-or-better).
