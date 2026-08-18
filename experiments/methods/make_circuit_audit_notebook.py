"""Generate circuit_audit_intermodule.ipynb — our inter-module circuit vs the paper's.

The audit notebook for the 2026-08-18 Fig-7 discrepancy investigation: renders the
circuit's structure in the PAPER'S units (timestep = one gate/measure/prep layer),
the per-round noise-mass profile (TICK-segmented), the conventions table (K, coupler
rate, framing), and the referee numbers (our IS LER vs Fig 7). Regenerate + re-execute
after the adapter-interleave change to show the before/after in one place.
"""
import json

from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# Inter-module measurement: our circuit vs Tour de Gross Fig 7

**The discrepancy (2026-08-18).** Our joint-parity LER at p=1e-3 is **0.186** (r1,
fair couplers); the paper's Fig 7 inter-module curve reads **~2e-3** under a *harsher*
any-logical-failure count. This notebook audits the construction differences.

**Conventions pinned so far:**

| item | paper (Fig 7) | ours (gross_intermodule_r1) |
|---|---|---|
| observables | full logical content (curves saturate at 1) | joint X̄₁⊗X̄₁ parity only (K=1, f(w) tail → 0.4997) |
| coupler rate | p (baseline) | r1: p ✓ (r10 leg = 10p, deliberately) |
| operation window | C=10 merged cycles, ~120 timesteps | C=10 merged rounds, each **3 serialized sub-cycles** |
| framing | minimal | 12 bare rounds before + return + 12 after (noise-cheap: ~4e-5 at 1e-3) |
| decoder | production relay-BP | campaign relay (num_sets=20); A/B vs num_sets=600 in flight |

**The audit's headline** (reproduced below): merged rounds carry **4.9×** a bare
round's noise mass, 91% of it idle-type — the serialized [A-cycle, B-cycle, adapter]
emission charges ~25-33 idle layers per merged round where an interleaved deformed
cycle would charge ~12. The merge window holds 64% of the circuit's total noise.
The fix: interleave the adapter into the module cycle (one ~12-timestep merged
cycle, the paper's construction).""")

code(r'''import sys
from collections import defaultdict
import numpy as np
from experiment_runner import load_config, build_circuit
from repo_paths import REPO_ROOT

cfg = load_config(str(REPO_ROOT / "experiments/configs/gross_intermodule_r1.yaml"))
circ = build_circuit(cfg)
print(f"qubits {circ.num_qubits}, measurements {circ.num_measurements}, "
      f"detectors {circ.num_detectors}, observables {circ.num_observables}")''')

md(r"""## Per-round noise mass (TICK-segmented)

One row per TICK segment. Structure: framing, 12 bare rounds, edge init, 10 merged
rounds, return, 12 trailing bare rounds, readout. Bare rounds ≈ 33 mass; merged
rounds ≈ 149 (91% DEPOLARIZE1). After the interleave, the merged rows should drop
to ~60-65.""")

code(r'''NOISY = {"DEPOLARIZE1": "D1", "DEPOLARIZE2": "D2", "X_ERROR": "other",
         "Z_ERROR": "other"}
seg, segs = defaultdict(float), []
for inst in circ.flattened():
    if inst.name == "TICK":
        segs.append(dict(seg)); seg = defaultdict(float); continue
    if inst.name in NOISY:
        n = len(inst.targets_copy()) // (2 if inst.name == "DEPOLARIZE2" else 1)
        seg[NOISY[inst.name]] += n * float(inst.gate_args_copy()[0])
segs.append(dict(seg))
print(f"{'seg':>4} {'D1':>8} {'D2':>7} {'other':>6} {'total':>8}")
for i, s in enumerate(segs):
    t = sum(s.values())
    if t > 0.01:
        print(f"{i:4d} {s.get('D1',0):8.2f} {s.get('D2',0):7.2f} "
              f"{s.get('other',0):6.2f} {t:8.2f}")
bare = [sum(s.values()) for s in segs[1:13]]
merged = [sum(s.values()) for s in segs[14:24]]
print(f"\nbare ~{np.mean(bare):.1f}/round   merged ~{np.mean(merged):.1f}/round   "
      f"ratio {np.mean(merged)/np.mean(bare):.2f}")
print(f"merge-window share of total noise: "
      f"{sum(merged)/sum(sum(s.values()) for s in segs):.0%}")''')

md(r"""## Duration in the paper's timestep units

Timestep = one gate layer / measurement / prep. A gross cycle ≈ 9 (7 two-qubit
layers + M + R). The paper's operation ≈ 120 timesteps; our noisy circuit ≈ 495,
of which the merge window is ~270 (should be ~120 after the interleave).""")

code(r'''rows = [("12 init bare rounds", 12, 12 * 9),
        ("merge window (10 rounds x 3 serialized sub-cycles)", 30, 30 * 9),
        ("return + boundary", 1, 9),
        ("12 trailing bare rounds", 12, 12 * 9)]
tot_c = sum(r[1] for r in rows); tot_t = sum(r[2] for r in rows)
for name, cyc, ts in rows:
    print(f"{name:52s} {cyc:3d} cycles  ~{ts:3d} timesteps")
print(f"{'TOTAL noisy':52s} {tot_c:3d} cycles  ~{tot_t:3d} timesteps   "
      f"(paper: ~120 for the operation)")''')

md(r"""## The referee numbers

IS-reweighted joint-parity LER from the measured spectrum (stride-filled), vs the
paper's Fig 7 read-offs. The decoder A/B (cheap num_sets=20 vs paper-grade 600, same
sampled configs) apportions the gap between decoder and construction — its results
land in `runs/framework/bb144/inter_module_ab.json`.""")

code(r'''import json, pathlib
from importance_sampling import FailureSpectrum, reweight_spectrum
d = REPO_ROOT / "runs/cluster/framework/bb144/bb144/inter_module_r1"
j = json.loads((d / "spectrum.json").read_text(encoding="utf-8"))
tw, fw = j["trials_by_weight"], j["failures_by_weight"]
ws = sorted(int(w) for w in tw)
tr = [int(tw[str(w)]) for w in ws]; fa = [int(fw[str(w)]) for w in ws]
wf, tf, ff = [], [], []
for i, (w, t, f) in enumerate(zip(ws, tr, fa)):
    wf.append(w); tf.append(t); ff.append(f)
    if i + 1 < len(ws):
        for wm in range(w + 1, ws[i + 1]):
            wf.append(wm); tf.append(t + tr[i + 1]); ff.append(f + fa[i + 1])
spec = FailureSpectrum(weights=wf, trials=tf, failures=ff,
                       n_expanded=int(j["n_expanded"]), q_base=float(j["q_base"]),
                       p_ref=float(j["p_ref"]))
for p, v in zip([1e-3, 5e-4, 1e-4], reweight_spectrum(spec, [1e-3, 5e-4, 1e-4]).P_logical):
    print(f"ours  LER({p:.0e}) = {v:.3e}   (joint parity, K=1, a=1/2)")
print("paper LER(1e-3) ~ 2e-3 (Fig 7, any-logical-failure)")
ab = REPO_ROOT / "runs/framework/bb144/inter_module_ab.json"
if ab.exists():
    print("\ndecoder A/B (same configs, cheap vs paper-grade):")
    for w, r in sorted(json.loads(ab.read_text(encoding="utf-8")).items(), key=lambda kv: int(kv[0])):
        print(f"  w={w}: cheap {r['cheap_fails']}/{r['shots']}  paper "
              f"{r['paper_fails']}/{r['shots']}  (b01={r['b01']} b10={r['b10']})")''')

md(r"""## Reading it

* The **framing sandwich is innocent** — 24 bare rounds contribute ~4e-5 at p=1e-3.
* The **merge window is the defect**: 64% of the noise, charged at ~2.4× the density
  an interleaved cycle would have, on exactly the rounds whose syndrome protects the
  joint parity.
* The **decoder buys the A/B's measured factor** (a few ×, not the ~100×).
* Prediction for the interleaved rebuild: merged-round mass ~60-65, window ~120
  timesteps, effective threshold up ~2-2.5×, and the p=1e-3 point falling from 0.186
  toward the paper's regime. Residual gap after that points at the adapter's
  stabilizer content (the U1 convention / E2 arbitration questions), not its schedule.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "tour_de_gross" / "circuit_audit_intermodule.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
