# Cluster campaign runbook — K=12 memory ladder + LPU error budgets

Manual-submission runbook for the campaign plan of 2026-07-13 (plan:
`~/.claude/plans/delegated-conjuring-kite.md`; predecessor context in the repo memory).
Each wave is a copy-paste section: run top to bottom, check every box before `sbatch`.

> **Launching Wave 6i on rodan? Use [`RODAN_QUICKSTART.md`](RODAN_QUICKSTART.md)** — the
> copy-paste sequence (fetch/checkout, verify, box check, smoke, launch, resume, pull) with
> a troubleshooting table. This runbook keeps the rationale and the per-wave gates.

**Global rules**
- Always submit through `bash experiments/slurm/submit.sh --only <substring> [--dry-run]` —
  never bare `submit.sh` (it would resubmit earlier waves against live outdirs).
- Dry-run first, every time. Verify the printed entries are EXACTLY the wave's jobs.
- Pin the cluster checkout per wave: record `git rev-parse HEAD` below at submit time and do
  not pull mid-wave — a requeued task must resume against the same code. Fast-forward only
  between waves, after the wave's gate passes.
- A timed-out/killed task is resumable: re-run the SAME sbatch line (per-weight checkpoint
  loses at most one weight). If it aborts with "weights_plan/seed mismatch": the config
  changed under a live run — do NOT edit configs of running jobs.
- **rodan has NO host python and NO scheduler.** Everything runs inside the podman image:
  `podman run --rm ... localhost/stim-work-qec:latest python ...`. Any runbook line below that
  invokes bare `python`/`pytest`/`srun`/`sbatch` on the login node is from the SLURM era and will
  not work there. Host-side tooling is limited to git, rsync, podman and shell (`box_load.sh` is
  deliberately pure bash for this reason). Analysis that needs python — `qc_wave.py`,
  `lambda_analysis`, the report notebooks — runs on the LOCAL box after an rsync, not on rodan.
- Pull results home from the local box (any time; checkpoints are atomic):
  `rsync -av cluster:stim_work/runs/framework/ runs/cluster/framework/`
  `rsync -av cluster:stim_work/runs/slurm/     runs/cluster/slurm/`

---

## Wave 1 — memory full-noise ladder            submit day: 2026-07-14
Jobs: bb6_memory (Fig-10 control, 16c/8h), bb6_memory_m100 (ladder leg, 16c/12h),
bb144_memory (32c/72h), bb288_memory (48c/96h).
Pinned SHA: `____________________` (fill at submit: `git rev-parse HEAD`)

0. Preconditions (from local Day-0, all done 2026-07-13):
   - [x] all five original configs smoke locally; pytest green
   - [x] bb144 adaptive_failures=25 (+72h), bb288 failures=15 + explicit contiguous-head weights
   - [x] bb6_memory_m100 exists (decoder-unified num_sets=100; no f0 pin)
   - [x] submit.sh has --only + per-entry mem; manifest indices frozen (0-5)
1. One-time env build (login node), from the repo root at the pinned SHA:
   ```
   git clone <your-fork-url> stim_work && cd stim_work        # or git pull if cloned
   git checkout bb144-split-better && git rev-parse HEAD       # record above
   pip install -r requirements.txt && pip install -e .         # relay_bp needs the Rust toolchain
   python test.py                                              # preflight: packages + editable
                                                               # install + decode round-trip +
                                                               # SLURM reachable — must be 15/15
   python -m pytest -q                                         # must be green before anything
   ```
2. 15-minute compute-node smoke of the LARGEST config (catches env/arch issues at minute 15,
   not hour 90 — do not skip):
   ```
   srun --cpus-per-task=4 --mem=16G --time=00:20:00 \
     python -m experiment_runner --config experiments/configs/bb288_memory.yaml --smoke --cpus 4
   ```
   PASS = `runs/framework/bb288/memory_smoke/result.npz` exists.
3. Dry-run — verify EXACTLY four entries print (indices 0-3, cpus/time/mem as above):
   ```
   bash experiments/slurm/submit.sh --only memory --dry-run
   ```
4. Submit: `bash experiments/slurm/submit.sh --only memory`
5. Monitor: `squeue -u $USER`; logs `tail -f runs/slurm/qec-bb288_memory_*.out`.
   Expected completions: bb6 pair same day; bb144 ~07-17; bb288 ~07-18.
6. Pull results home (rsync lines above) — safe mid-run; bb6 lands within ~a day.
7. Close: on the local box, `python experiments/methods/qc_wave.py --wave 1` (available after
   LA lands, ~Day 4). PASS = G1: Fig-10 reproduction in tolerance; Λ(bb6_m100→bb144)(p_ref)
   finite ±σ and **Λ>1** (Λ<1 anywhere = decoder-degradation tell → STOP, no W2);
   bb144 onset-region zero-failure-bin fraction <20%.

## Wave 1b — LPU x1/z1 full-noise               submit day: 2026-07-14/15
Jobs: gross_lpu_x1, gross_lpu_z1 (16c/24h each). May share the submit day with Wave 1.
**Quarantine: no LPU-vs-memory comparison until the TdG-vs-Bravyi presentation test (G3a).**

1. Dry-run — exactly two entries (indices 4-5): `bash experiments/slurm/submit.sh --only lpu --dry-run`
2. Submit: `bash experiments/slurm/submit.sh --only lpu`
3. Monitor/pull as above. Results land in `runs/framework/bb144/lpu_x1|lpu_z1`.

## Wave 2 — bb6+bb144 memory channels           submit day: ~2026-07-16/17
20 jobs: `configs/channels/bb6_memory_m100__*` + `bb144_memory__*` (iso+abl × 5 channels).
Pinned SHA: `____________________`

0. Preconditions:
   - [ ] G1 passed (`qc_wave --wave 1`; bb144 may be partial — its checkpoint reweights)
   - [ ] G2 passed locally (channel tests + one iso/abl smoke per code; done Day-0/1)
   - [ ] Wave-2 manifest block uncommented (indices 6-25; do NOT reorder anything above it)
1. Fast-forward the cluster checkout to the recorded W2 SHA; `pytest -q`.
2. Dry-run: `bash experiments/slurm/submit.sh --only channels/bb6 --only channels/bb144 --dry-run`
   — exactly 20 entries, walltimes 4-72h per the manifest.
3. Submit: same line without `--dry-run`.
4. Close: `qc_wave --wave 2` — decomposition identity (Σ isolated + residual vs full within σ),
   verdict table at p_ref and 2×p_ref, no systematic Λ<1.

## Wave 3 — bb288 channels + LPU x1/z1 channels submit day: ~2026-07-20/21
30 jobs: `configs/channels/bb288_memory__*` + `gross_lpu_x1__*` + `gross_lpu_z1__*`.
Preconditions: G3 (LPU partition test green — already in tests/test_channel_filter.py;
G3a presentation test resolved either way; bb288 full-noise complete through the onset head;
W2 QC clean). Uncomment the Wave-3 block; submit
`bash experiments/slurm/submit.sh --only channels/bb288 --only channels/gross_lpu`.

## Wave 4 — targeted repair                     submit day: ~2026-07-24+ (data-driven)
Job list comes ONLY from `qc_wave` verdict tables: onset-boost configs (new config, explicit
onset weights stride 1, seed 43 — pooled downstream, never in-place) for "not robust" shares;
splitting anchors for bb288 full + iso_cz and one LPU circuit; Tech-II gap-fill for
bb288 iso_cz/abl_cz; p_hi-extension reruns for crossings that fell outside sampled mass.

## Wave 5 — new LPU operations                  after G5 (L3 battery) passes locally
In-module joint Pauli → shift automorphism → out-of-module joint Pauli. Per op: full-noise
first (sizing probe sets walltime), channels after that op's own partition test. Nothing
ships without: zero-noise determinism + TableauSimulator logical correctness + mini-scale
three-way (MC/IS/Tech-II) budget agreement.

**W5 local pilot LANDED 2026-07-21** (user-driven; Tech I+II only, adaptive_shots_max=3000):
`lpu_idle` (fail-fast §2.4 memory: 12 noisy + 1 fault-free cycle, K=12), `joint_pauli` = Ȳ1
(Tour de Gross App A.1: FULL LPU, one uniform convention — X-vertex/Z-cycle checks on edges,
|0⟩ edge init, Z readout, two-qubit Bell check with CY→x⁴R, CZ deformation on the X-check
side; the X̄1/Z̄1 half-LPU branches are Hadamard-duals and CANNOT compose into Ȳ1), and
`automorphism` (δ=y default: 6-step swap + syndrome cycle per instruction, ×C=10,
shifted-label detectors, observables through the GF(2)-recomputed δ^C action).
Configs: `experiments/configs/gross_lpu_{idle,y1}.yaml`, `gross_automorphism.yaml` —
paper-faithful `lpu_idle_noise: true` (NOT comparable to Wave-1b x1/z1, which predate the
flag), shared analysis grid p ∈ [1e-4, 5e-3], cheap tuned Relay (num_sets=20/stop_nconv=5;
runner default 600 = paper settings). Idle noise dominates the deep circuits' fault mass
(μ(p_ref): idle 175 / automorphism 569 / Y1 979) — hence the capped grid + strided windows.
Known gaps: Y1 memory observables are the 11-element Z-logical commuting-subgroup basis
(the paper's K=24 X-rows need a second basis); module-style layering, not the paper's
12-timestep coloring schedule (fault counts differ from the paper's N). Gate battery at
landing: p=0 determinism ×5 builders, Tableau Ȳ1-eigenstate + automorphism-action checks,
DEM builds (idle M=1872 / Y1 M=5456 / automorphism M=7776, all K=12), RelayBP setup,
single-fault decoder-floor probe, smoke runs. Tests: `tests/test_lpu_circuits.py`.
**2026-07-22 status:** idle + automorphism floors CLEAN (0/70416, 0/208512) — production
launched (Tech II on idle: D=10, w0=5, matching the paper's BB(12)-circuit ≤10). Tech II
dropped from the Y1/automorphism configs (~2 h per BP-OSD decode at 210k columns; the
paper's Table 2 likewise has no Y1 row — w0 from first observed failure). **Y1 BENCHED:**
floor probe found 128 weight-1 fails (weighted 4.4e-4) that survive 600-set Relay; root
cause = 46 same-syndrome different-action single-fault groups, ALL differing in observable
0 — the outcome bit as framed (MPP ⊕ last-round vertex product) is not closed by the
detector set (the paper's per-round m̄ chain + return-boundary anchoring is the missing
structure). Fix in progress. The weight-1 degeneracy scan (full DEM, zero
same-syndrome-different-action groups) is now a MANDATORY Wave-5 gate for every new
operation builder — it catches informationally-unresolvable floors that no decoder probe
can attribute.
**COMPANION GATE, added 2026-07-27 (Wave 6): the low-weight failure spectrum f(w), via
`experiments/tour_de_gross/failure_spectrum_probe.py`.** The degeneracy scan catches
*undetectable* faults; f(w) catches *miscorrected* ones, and it is the only cheap check that
is p-INDEPENDENT. **A Monte-Carlo LER at p_ref cannot distinguish a broken observable from a
circuit operating far above threshold** — Wave 6 lost a session to exactly this, recording the
inter-module X̄₁⊗X̄₁ circuit as having an "obs0 floor" at LER≈0.40 (p=5e-3) when the DEM says
it sees ~106 expected faults/shot there and the *known-good* Y1 baseline floors just as hard
(0.35 idle-off / 0.51 idle-on) at the same point. The same circuit decodes to LER 0.020±0.010
at p=1e-3. Rules:
- Always compare a new builder against a VALIDATED baseline, **never against zero**. Y1 itself
  is nonzero at w=3,4,6 (1,1,3 per 400) — decoder miscorrection at the cheap 20-set relay, not
  undetectable errors. "Matches the baseline" is the standard.
- Quote the resolution. At T samples/weight, f(w) resolves to ~1/T, so a clean 0/T *bounds* the
  floor at ~1/T; it does not disprove one. Ruling out the ~5e-4-class floor the configs
  reference needs T≳10000 ⇒ a cluster job (the probe is chunked, checkpointed per weight, and
  tops up on re-run with a larger --T).
- Before trusting a distance number, note `shortest_graphlike_error` skips hyperedge/gauge
  errors and `compute_distance` returns a spurious 1 on these deformed non-CSS circuits. When
  MC, graphlike distance, and f(w) disagree, **f(w) is the one to trust.**
**Symmetry (2026-07-22):** the IDLE circuit is the bare gross code and keeps full Z12xZ6
toric symmetry (72 perms; build_circuit_translation_perms works with l=12,m=6) — so
gross_lpu_idle.yaml uses mw_use_symmetry:true. REQUIRED, not just faster: idle has even
D=10, |L(D)| feeds the exact Prop.1 onset f0, and without orbit expansion |L(D)| (hence
f0) is 72x-loose. The Y1/automorphism circuits genuinely lack it (LPU checks / shift), keep
false. GOTCHA: experiment_runner does NOT persist the found L(D) supports (only counts in
distance.json), so a symmetry-off run's loose |L(D)|/f0 is UNRECOVERABLE — must re-run
Tech II, can't post-expand. SYMMETRY-PRUNE (IMPLEMENTED 2026-07-22, commit 5cebce12): find_min_weight_logicals now takes
`systematic_masks`; run_technique_ii computes one functional per translation orbit via
`symmetry_orbit_representatives` (mw_symmetry_prune, default True) instead of all 2^K-1. Proven
exact (translation preserves weight; functionals transform by the contragredient T_sp^{-T};
test_symmetry_prune_matches_full_sweep asserts pruned==full L(D) on [[18,4,4]]). Gross idle:
4095 -> 155 reps = 26.4x fewer decodes (orbits avg ~26, NOT the naive 72 — sizes 3..36). One-
time GF(2) preimage solve up front (~4 min big-int on idle; O(N^2) — optimize for two-gross).
Biggest win where a decode is expensive: two-gross (same K=12, ~15s/decode) systematic phase
shrinks ~26x. Random-trial phase unchanged.

## Wave 6 — double-gross LPU                    open-ended
Blocked on derive_lpu_layout at (12,12). First job = sizing probe, then mirror W1b→W3.
**NAME COLLISION — read this.** This section is the LPU on the two-gross CODE ((l,m)=(12,12),
20v/32e/11-cycle U_l, paper tex lines 218-240). It is NOT the branch `wave6-intermodule`, which
is a different experiment: TWO separate [[144,12,12]] modules joined by the code-code adapter.
That one is READY TO SUBMIT — see "Wave 6i" below. This double-gross section is still blocked.


---

## Wave 5b — LPU boost pass (deeper shots)      submit day: ____________
Jobs: gross_lpu_idle_boost, gross_automorphism_boost, gross_lpu_y1_boost (48c/96h, 32/48/64G).
Pinned SHA: `____________________` (fill at submit: `git rev-parse HEAD`)

Purpose: the Wave-5 runs capped at `adaptive_shots_max: 3000`, so every zero-failure bin is
bounded only at the rule-of-three limit 3/3000 ≈ 1e-3. This pass re-samples at a deeper cap to
push that floor toward 1e-6, and coarsens idle's stride from 1 to 3 so the budget buys depth
rather than breadth. Boosts are NEW configs at seed 43 in their OWN `*_boost` outdirs, pooled
with the parents downstream — the completed runs are never touched.

**Caps are per campaign and are NOT interchangeable.** `run_is_sweep` checkpoints per WEIGHT with
no intra-bin save, so a bin that cannot finish inside one walltime never lands: every requeue
restarts it and the job burns the allocation on one weight, silently and forever. Each cap is
sized so the worst bin is ~1/4 of the 96 h wall at 48 cores.

**rodan has NO SCHEDULER** (no `sbatch`/`srun`) and is a SHARED 96-core box — discovered
2026-07-27. `experiments/slurm/submit_lpu_boost.sh` is therefore unusable there; launch with
`bash container/run_lpu_boost.sh` instead (detached podman, `CPUS=8` per job). Note
`container/run_local.sh` will NOT work: it mounts only `runs/`, so it would run against the
image's baked configs and never see the boost files.

Budget: 8 threads/job x 3 jobs = 24 of 96 cores (25%). Caps were cut 10x from the SLURM-era
values to keep the wall near 5 d at that footprint.

| campaign     | onset w0 | s/shot @8c | cap     | stride | worst bin | floor 3/T | run @8c |
|--------------|----------|------------|---------|--------|-----------|-----------|---------|
| idle         | 39       | 0.153      | 300,000 | 1 -> 3 | ~14.9 h   | 1.0e-5    | ~4.6 d  |
| automorphism | 69       | 2.24       | 20,000  | 4 -> 8 | ~12.4 h   | 1.5e-4    | ~3.5 d  |
| Y1           | 25       | 0.531      | 100,000 | 6      | ~17.2 h   | 3.0e-5    | ~4.9 d  |

The per-bin-must-fit-in-one-walltime rule below drove the ORIGINAL (SLURM) caps. With no
scheduler there is no walltime kill, so it no longer binds — but per-weight checkpointing still
matters: a killed container resumes from `spectrum.json` losing at most one bin.

Boost strides are COARSER than the parents' and must stay a MULTIPLE of them, so every deep bin
lands exactly on a parent bin and pools additively. This costs no grid resolution — the parents
already sampled the finer grid, so the pooled spectrum keeps it; the boost only chooses which of
those bins get the deep budget.

Do NOT trim the window to the low-f bins: cost by failure-rate band is 0.1-1.9% above f=1e-2 and
70-96% in the sub-onset capped bins (f<1e-4), because the adaptive rule spends 20/f shots and a
high-f bin costs ~40. The cheap high-f bins also bootstrap the descending sweep's predictor —
`predict_failure_fraction` uses only 0<f<1 points, so a window starting at f~1e-2 gives the first
bin ~20 shots, it measures 0, and the "only zero points" branch then returns 1e-12 and sends the
NEXT bin to shots_max even where 4000 shots would do. Trimming makes the run slower, not faster.

The automorphism's cap is 15x smaller than idle's for a physical reason worth remembering: the
bins that hit the cap are always the SUB-ONSET ones (above the onset, adaptive stops early on the
20-failure target and the cap never binds), and decode cost climbs steeply with fault weight
because RelayBP needs more legs on a dense syndrome. The automorphism has the steepest spectrum
of the three (gamma1 = 13.9) so it survives to w=69 before failing at all, and every shot up
there costs ~2.24 s vs 0.53 s for Y1 at ITS onset of w=25. Its robustness is exactly what makes
its floor expensive to measure. Rates measured 2026-07-27: idle from a local probe (0.153 s/shot
at w=40, 8 threads), the other two from their own prod logs.

0. Preconditions
   - [ ] Wave-5 parents complete (`runs/framework/bb144/{lpu_idle,automorphism,joint_pauli}`)
   - [ ] `python -m pytest -q` green at the pinned SHA
   - [ ] configs load: `python -c "from experiment_runner import load_config; [load_config(p) for p in __import__('pathlib').Path('experiments/configs').glob('gross_*_boost.yaml')]"`
1. 15-minute compute-node smoke of the LARGEST (Y1, 64G is the least-tested figure):
   ```
   srun --cpus-per-task=4 --mem=16G --time=00:20:00 \
     python -m experiment_runner --config experiments/configs/gross_lpu_y1_boost.yaml --smoke --cpus 4
   ```
   PASS = `runs/framework/bb144/joint_pauli_boost_smoke/result.npz` exists.
2. Dry-run — verify EXACTLY three entries, cpus/time/mem as in the table:
   ```
   bash experiments/slurm/submit_lpu_boost.sh --dry-run
   ```
3. Submit: `bash experiments/slurm/submit_lpu_boost.sh`
4. Monitor: `squeue -u $USER`; logs `tail -f runs/slurm/gross_*_boost_*.out`.
   **idle and the automorphism are expected to TIME OUT once** — that is planned, not a failure.
   Re-run the same line to resume (per-weight checkpoints lose at most one bin). Never edit a
   config while its job is live: the guard at `experiment_runner.py:642` aborts the resume on any
   weights_plan/seed change.
5. Pull results home: `rsync -av cluster:stim_work/runs/framework/ runs/cluster/framework/`
6. Close: pool each boost with its parent and confirm the floor actually moved.
   ```python
   from lambda_analysis import load_run, pool_spectra, fill_spectrum, rw_stats, zero_bin_fraction
   for op in ["lpu_idle", "automorphism", "joint_pauli"]:
       a, b = load_run(f"runs/framework/bb144/{op}"), load_run(f"runs/cluster/framework/bb144/{op}_boost")
       s = fill_spectrum(pool_spectra(a.spectrum, b.spectrum))
       print(op, rw_stats(s, 1e-3), zero_bin_fraction(s))
   ```
   PASS = `pool_spectra` does not raise (identical n_expanded/q_base/p_ref = same circuit, the
   validity condition for pooling), zero-bin fraction drops, and the p=1e-3 headroom shrinks by
   roughly the cap ratio. Then re-run `notebooks/tour_de_gross/wave5_lpu_ops_report.ipynb`
   against the pooled spectra.

## Wave 6i — gross-to-gross inter-module X̄₁⊗X̄₁   submit day: ____________
Jobs: gross_intermodule_r1, gross_intermodule_r10. Branch `wave6-intermodule` (merged to main).
Pinned SHA: `____________________` (fill at submit: `git rev-parse HEAD`)

Distinct from "Wave 6 — double-gross LPU" above: this is TWO [[144,12,12]] modules (A frame 0,
B frame 378) Bell-coupled by the Tour de Gross code-code adapter, benchmarking X̄₁(A)⊗X̄₁(B).
r1 = paper-faithful "all connections equally faulty"; r10 = the paper's flagged "couplers ~10×
worse". Technique II dropped (`[IS, I]`) — `compute_distance` is unreliable on these deformed
non-CSS circuits.

**⚠️ NO SCHEDULER ON RODAN.** Per Wave 5b above, rodan has no `sbatch`/`srun` and is a SHARED
96-core box. The manifest entry (48c/96h/64G) and `submit.sh --only intermodule` therefore apply
ONLY on a real SLURM cluster — on rodan, launch via detached podman as Wave 5b does. **No
`container/run_intermodule.sh` exists yet**: it is a two-line addition to the
`container/run_lpu_boost.sh` pattern (same env-var thread capping, same `experiments/` + `runs/`
mounts with `:Z`, same `--dry-run`/`--only` flags), swapping the three `launch` lines for
    launch im_r1  experiments/configs/gross_intermodule_r1.yaml
    launch im_r10 experiments/configs/gross_intermodule_r10.yaml
Write that launcher BEFORE submit day. Do NOT use `container/run_local.sh` — it mounts only
`runs/` and would run the image's baked configs, never seeing these two.

**SHARED-BOX BUDGET — HALF THE BOX, campaign-only (set 2026-07-27).** `CPUS=24` × 2 jobs =
**48 of 96 cores**. This is now the ONLY campaign running: the Wave-5b LPU boost pass is not being
launched, so its 24 cores are not in play. Half is a deliberate self-imposed cap on a shared
machine, not a machine limit — do not raise it to fill the box. Drop to `CPUS=16` (32/96) if
others are on. Confirm who else is on before launching.

**MEASURED COST (2026-07-27 probe, local 24-core box, C=10/d_init=12 idle ON).** Decode rate rises
with fault weight then SATURATES — 0.44 s/shot at w=5, 0.98 at w=20, 1.51 at w=50, then flat at
~1.84 from w=150 through w=1518. First failures appear between w=50 and w=150, so the onset sits
around w~100. Peak RSS **9.0 GB for one job** — the manifest's 64G is over-provisioned, not tight;
two jobs is ~18 GB. Projected sweep, r1 block, stride 6 (279 bins):

| band | share of cost |
|---|---|
| sub-onset, capped at 3000 shots (w < ~100, ~17 bins) | 52% |
| transition, adaptive 20/f shots (w ~100-400) | 37% |
| saturated, ~20-40 shots/bin (w > 400, ~212 bins) | 11% |

**~35 h per leg at 24 threads ⇒ both legs in parallel ≈ 1.5 days wall** at the half-box setting.
NB the measured rate ALREADY includes Relay-BP's rayon parallelism — it is not a per-core figure
and must not be divided by core count again.

0. Preconditions (all DONE 2026-07-27 unless noted):
   - [x] E1 (p=0 determinism), E2 (obs0 = MPP ref), E4 (DEM+decoder) green
   - [x] the recorded "obs0 floor" blocker RETIRED — it was above-threshold operation, not a
         broken observable. f(w) per 400, w=1..6: Y1 baseline 0,0,1,1,0,3 / inter pre-fix
         0,0,0,0,2,0 / inter+closure 0,0,1,0,1,2. Statistically indistinguishable ⇒ no floor.
   - [x] `weights_range` FROZEN from the idle-ON sizing probe: r1 [1,1674], r10 [1,1742].
         The old [1,900] placeholder stopped short of the p_hi mass at w≈1518-1583.
   - [x] tests/test_lpu_circuits.py 18 passed
   - [x] `container/run_intermodule.sh` written (mounts src/ — the shipped image predates the
         builder; it also preflights the import before launching)
   - [ ] step 2 smoke passed — **memory is an UNTESTED ESTIMATE, do not skip it**
1. **Check the box has room** — there is no scheduler, so nothing is tracking allocations and the
   only truth is what is running right now:
   ```
   bash container/box_load.sh              # report + a recommended CPUS
   bash container/box_load.sh --want 48    # does the half-box footprint fit?
   ```
   Reads the 1-min load average against `nproc`. **`podman ps` is NOT sufficient** — podman is
   rootless, so it shows only YOUR containers; a colleague can be using 60 cores through their own
   podman and your `podman ps` is empty. `box_load.sh` uses loadavg and `ps`, which see all users.
   If it reports less headroom than the policy cap, launch with the smaller `CPUS` it suggests.
2. **Get the cluster checkout onto `main`** — do NOT just `git pull`. The Wave-1 instructions
   above put the cluster on `bb144-split-better`, and that branch **no longer exists on either
   remote**; a bare `git pull` there fails or silently does nothing. Everything (including this
   campaign) is now merged to `main`:
   ```
   cd stim_work
   git branch --show-current          # likely a deleted branch - expect a surprise
   git fetch origin --prune
   git checkout main || git checkout -b main origin/main
   git pull --ff-only origin main     # ff-only: refuse a surprise merge on the cluster
   git rev-parse HEAD                 # record as the pinned SHA above
   ```
   Then run the suite **inside the container** — rodan has NO host python (see Global rules).
   Mount `tests/` as well as `src/`/`experiments/`, or you test the image's baked copies instead
   of the checkout you just pulled:
   ```
   podman run --rm -t \
     -v "$PWD/src:/opt/stim_work/src:Z" \
     -v "$PWD/experiments:/opt/stim_work/experiments:Z" \
     -v "$PWD/tests:/opt/stim_work/tests:Z" \
     -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work \
     localhost/stim-work-qec:latest python -u -m pytest -q
   ```
3. Smoke it FIRST. This circuit is bb288-class (418354 mechanisms, 10903 detectors at production
   geometry) and the DEM build ALONE took ~400-800s locally, so this is what validates the memory
   footprint and the per-shot rate before you commit days of shared-box time:
   ```
   podman run --rm -e OMP_NUM_THREADS=4 -e RAYON_NUM_THREADS=4 \
     -v "$PWD/src:/opt/stim_work/src:Z" \
     -v "$PWD/experiments:/opt/stim_work/experiments:Z" \
     -v "$PWD/runs:/opt/stim_work/runs:Z" \
     -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work \
     localhost/stim-work-qec:latest \
     python -m experiment_runner --config experiments/configs/gross_intermodule_r1.yaml \
       --smoke --cpus 4
   ```
   The `src/` mount is REQUIRED here for the same reason the launcher has it: without it the
   smoke runs the image's BAKED code, which predates the builder, and dies on an import error
   that reads like a circuit bug.
   PASS = `runs/framework/bb144/intermodule_r1_smoke/result.npz` exists. Watch RSS while it runs
   (`podman stats`): there is no scheduler to enforce a memory cap, so an OOM here takes down
   whatever else shares the box, not just this job.
4. Dry-run the launcher: `bash container/run_intermodule.sh --dry-run` — exactly two podman
   commands, threads as budgeted.
5. Launch: `bash container/run_intermodule.sh`. Watch with `podman ps` / `podman logs -f im_r1`.
   Resume after a kill: `podman rm <name>` then re-run with `--only <name>` — `run_is_sweep`
   checkpoints per weight, losing at most one bin. Never edit a config while its job is live:
   the guard at `experiment_runner.py:642` aborts the resume on any weights_plan/seed change.
**No separate large-T f(w) job is needed.** The production sweep subsumes it: the frozen block
starts at w=1 with stride 6, and sub-onset bins run to the 3000-shot cap, so f(1) lands with
resolution ~3.3e-4 — finer than the ~5e-4-class floor the configs reference, and measured at the
PRODUCTION geometry rather than the C=3 validation one. `failure_spectrum_probe.py` remains the
tool for a standalone check of a NEW builder; it is redundant once this sweep runs.

6. Pull results home: `rsync -av rodan:stim_work/runs/framework/ runs/cluster/framework/`
7. Close: f(w) for inter_module within Poisson error of the y1 baseline at every weight, and
   the IS spectra land inside the frozen weight blocks (no mass piled at w_hi).

KNOWN GAP (not a launch blocker): `lpu_include_memory_obs: false` — the K=23 merged-graph
memory-observable recipe has not landed, so these runs carry the operator observables only.
