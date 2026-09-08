"""Generate intermodule_coupler_split.ipynb — the memory/output split study.

Reads the four-cell run outputs (2026-08-31 study): MC cells with the K=23
output/memory failure split, and IS total spectra, for symmetric (r1) vs
10x-worse-coupler (r10) inter-module X̄₁(A)⊗X̄₁(B) measurement, deep600 decoder,
interleaved idle. Renders whatever is present (jobs finish at different times) —
run it after any rsync of runs/framework/bb144/. Self-contained code cells; needs
numpy/scipy/matplotlib (+ stim only via importance_sampling import — present in the
container). Regenerate structure here; edit and re-execute the notebook for data.
"""
import json
from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# Inter-module joint measurement: memory vs output failures, coupler sensitivity

**The circuit.** The gross-to-gross inter-module joint measurement of X̄₁(A)⊗X̄₁(B)
(experiment `inter_module` — NOT the in-module `joint_pauli` Y1), interleaved idle
model, **deep600** decoder (the validated best gross-code relay).

**The split (K=23).** obs 0 is the measurement **output** (joint parity); obs 1..22
are **memory** — the preserved Z-logicals commuting with the measured operator,
reconstructed from the final transversal readout with edge corrections. So every
failure is attributable to output, memory, or both — the paper's
F / F_out / F_mem / F_both.

**Two conventions to keep in mind when reading this:**
* **22 of 23 memory logicals.** The joint Z̄₁(A)⊗Z̄₁(B) is omitted (its correction
  routes through the bridge — the deferred merged-graph recipe). Tiny undercount of
  F_mem: only failures flipping *exclusively* that one logical are missed.
* **F_mem is Z-basis-visible memory error** (logical X/Y on stored qubits), from the
  |0⟩-init / Z-readout framing — phase-type memory errors are invisible here. Same
  convention as the in-module K=12 Table-4 match.

**The two error models.** r1 = symmetric couplers (p_coupler = p). r10 = Bell/coupler
fidelity 10× worse (p_coupler = 10·p). Everything else identical, so r1-vs-r10
isolates Bell-pair-fidelity sensitivity — and the split says whether bad couplers
threaten the **output** parity, the **stored memory**, or both.

**IS + MC.** MC cells give the split directly at p=1e-3 (3000 shots). IS cells give
the total failure spectrum f(w) (full both-sector DEM), reweightable to a curve and
cross-checked against the MC totals.""")

code(r'''import json, pathlib
import numpy as np
import matplotlib.pyplot as plt
from repo_paths import REPO_ROOT

BB = REPO_ROOT / "runs" / "framework" / "bb144"

def load_mc():
    """Merge the per-cell MC json files (mc_r1.json, mc_r10.json)."""
    out = {}
    for f in ("mc_r1.json", "mc_r10.json", "lpu_direct_mc.json"):
        p = BB / f
        if p.exists():
            out.update(json.loads(p.read_text(encoding="utf-8")))
    return out

MC = load_mc()
print(f"MC cells present: {sorted(MC)}" if MC else "no MC results yet")''')

md(r"""## The memory/output split (direct MC, p = 1e-3)

Wilson-ish binomial errors on each rate. `F = F_out + F_mem − F_both`.""")

code(r'''def se(k, n): return (np.sqrt(k)/n) if (k and n) else (0.0 if n else float("nan"))
rows = [("symmetric (r1)", "im_r1_deep600@1e-03"),
        ("10x coupler (r10)", "im_r10_deep600@1e-03")]
print(f"{'model':18s} {'shots':>8} {'LER_total':>18} {'LER_out':>18} {'LER_mem':>18} {'both':>6}")
mc_summary = {}
for label, key in rows:
    r = MC.get(key)
    if not r:
        print(f"{label:18s}  (not finished)"); continue
    n = r["shots"]
    tot, out, mem = r["fails"]/n, r["f_out"]/n, r["f_mem"]/n
    mc_summary[label] = dict(n=n, tot=tot, out=out, mem=mem,
                             F=r["fails"], Fout=r["f_out"], Fmem=r["f_mem"], Fboth=r["f_both"])
    print(f"{label:18s} {n:>8} "
          f"{tot:.3e}±{se(r['fails'],n):.1e}  "
          f"{out:.3e}±{se(r['f_out'],n):.1e}  "
          f"{mem:.3e}±{se(r['f_mem'],n):.1e}  {r['f_both']:>6}")''')

md(r"""## Coupler sensitivity: does 10× worse Bell fidelity hit output or memory?

The ratio r10/r1 on each channel. If the couplers threaten the **output** parity
(the measurement's job), the output ratio is large and the memory ratio ~1. If they
corrupt **stored memory**, the reverse. This is the study's central question.""")

code(r'''a, b = mc_summary.get("symmetric (r1)"), mc_summary.get("10x coupler (r10)")
if a and b:
    def ratio(x, y): return (y/x) if x else float("inf")
    print(f"{'channel':10s} {'r1':>12} {'r10':>12} {'r10/r1':>8}")
    for ch, k in (("total","tot"), ("output","out"), ("memory","mem")):
        print(f"{ch:10s} {a[k]:12.3e} {b[k]:12.3e} {ratio(a[k],b[k]):8.2f}")
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(3); w = 0.38
    ax.bar(x-w/2, [a["tot"],a["out"],a["mem"]], w, label="symmetric (r1)")
    ax.bar(x+w/2, [b["tot"],b["out"],b["mem"]], w, label="10x coupler (r10)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["total","output","memory"])
    ax.set_ylabel("LER at p=1e-3"); ax.set_title("coupler-fidelity sensitivity by channel")
    ax.legend(); ax.grid(alpha=0.3, which="both", axis="y"); plt.tight_layout(); plt.show()
else:
    print("need both MC cells finished for the sensitivity comparison")''')

md(r"""## IS total failure spectra f(w), and reweighted LER vs the MC anchors

The IS cells give the full spectrum (any-observable failure, full both-sector DEM).
Reweighting gives LER(p); the filled MC-total points at 1e-3 are the cross-check
(they should sit on the IS curves within errors — the whole reason both were run).
NB the IS *total* is not yet split into output/memory (per-observable IS is a
follow-on); the split lives in the MC section above.""")

code(r'''import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from importance_sampling import FailureSpectrum, reweight_spectrum

def load_spec(sub):
    p = BB / sub / "spectrum.json"
    if not p.exists(): return None
    j = json.loads(p.read_text(encoding="utf-8"))
    tw, fw = j["trials_by_weight"], j["failures_by_weight"]
    ws = sorted(int(w) for w in tw)
    tr = [int(tw[str(w)]) for w in ws]; fa = [int(fw[str(w)]) for w in ws]
    # stride-fill (pool gaps) so reweight covers the mass
    wf, tf, ff = [], [], []
    for i,(w,t,f) in enumerate(zip(ws,tr,fa)):
        wf.append(w); tf.append(t); ff.append(f)
        if i+1 < len(ws):
            for wm in range(w+1, ws[i+1]):
                wf.append(wm); tf.append(t+tr[i+1]); ff.append(f+fa[i+1])
    spec = FailureSpectrum(weights=wf, trials=tf, failures=ff,
                           n_expanded=int(j["n_expanded"]), q_base=float(j["q_base"]),
                           p_ref=float(j["p_ref"]))
    return ws, tr, fa, spec

CELLS = [("symmetric (r1)", "inter_module_r1_il_deep600", "C0"),
         ("10x coupler (r10)", "inter_module_r10_il_deep600", "C1")]
pg = np.geomspace(1e-4, 5e-3, 60)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
for label, sub, col in CELLS:
    s = load_spec(sub)
    if s is None:
        print(f"{label}: IS spectrum not present yet"); continue
    ws, tr, fa, spec = s
    m = [(w, f/t) for w,t,f in zip(ws,tr,fa) if f>0]
    if m:
        axL.plot([w for w,_ in m], [f for _,f in m], "o-", ms=4, color=col, label=label)
    rw = reweight_spectrum(spec, pg)
    axR.plot(pg, rw.P_logical, "-", color=col, label=f"{label} (IS total)")
# MC-total anchors
for label, _, col in CELLS:
    r = MC.get("im_r1_deep600@1e-03" if "r1" in label else "im_r10_deep600@1e-03")
    if r:
        axR.plot(r["p"], r["ler"], "s", color=col, ms=9, mec="k", zorder=5)
axL.set_xlabel("fault weight w"); axL.set_ylabel("f(w)"); axL.set_yscale("log")
axL.set_title("IS failure spectrum (total)"); axL.grid(alpha=0.3, which="both"); axL.legend(fontsize=8)
axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("physical error rate p")
axR.set_ylabel("LER (per shot)"); axR.set_title("reweighted IS total + MC anchor (squares)")
axR.grid(alpha=0.3, which="both"); axR.legend(fontsize=8); plt.tight_layout(); plt.show()''')

md(r"""## Reading it

* **The sensitivity table/bars are the headline** — the `r10/r1` ratio per channel
  says whether 10× worse Bell couplers threaten the measurement **output**, the
  **stored memory**, or both.
* **Squares on the right panel are the MC totals**; if they land off the IS curves,
  the IS-vs-MC cross-check has failed (suspect the DEM sector or the reweight) —
  they should agree within errors.
* **Bounds vs zeros**: an IS bin with 0 measured failures is a rule-of-three bound,
  not a zero; the deep low-weight tail is unmeasured, not absent.
* **Caveats** (from the top cell): 22/23 memory logicals; F_mem is Z-basis-visible
  only. Don't over-read the absolute memory numbers against the paper's K-harness
  framing — the r1-vs-r10 *ratio* is the robust quantity.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "tour_de_gross" / "intermodule_coupler_split.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
