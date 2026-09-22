"""Generate intermodule_coupler_split.ipynb — the memory/output split + coupler study.

Reads whatever inter-module run outputs are present under runs/framework/bb144/ and
renders gracefully (jobs finish at different times):
  * IS total spectra  inter_module_r{1,10}_il_{deep600,fast}/  (complete = result.npz)
  * MC cells          mc_r1.json, mc_r10.json, lpu_direct_mc.json, mc_dinit.json
Sections: MC memory/output split; coupler sensitivity r10/r1; IS spectra + reweighted
curves with MC anchors; coverage table (what is measured vs extrapolated); f5 ansatz
extrapolation to the 1e-4 regime with the paper's PINNED onset w0=ceil(d_circ/2)=5
(Tour de Gross Table 4: inter-module gross d_circ<=10), a bootstrap band, the free-w0
fit as a reference and the decoder-floor bracket; the d_init sweep (merge vs idle).

Self-contained code cells; needs numpy/scipy/matplotlib + the repo's src/ (stim is
imported transitively — run in the qec env or the container). Regenerate structure
here; edit and re-execute the notebook for data.
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
model, C=10 merged rounds padded by d_init=12 bare rounds each side (34 noisy rounds).

**Decoders — read the labels.** Two relay configurations appear: the **campaign relay
(num_sets=20)** — the fast local decoder behind the IS spectra and the d_init sweep —
and **deep600** (num_sets=600, the validated paper-grade config, ~12× fewer failures
than the campaign relay on this code) behind the fish MC cells. Absolute LERs differ
between them; *ratios within one decoder* (r10/r1, d6/d12) are the robust quantities.

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

**IS + MC.** MC cells give the split directly at a few p (≤3000 shots). IS cells give
the total failure spectrum f(w) (full both-sector DEM), reweightable to a curve and
cross-checked against the MC totals.""")

code(r'''import json, pathlib, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import gammaln
from repo_paths import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
from importance_sampling import (FailureSpectrum, reweight_spectrum,
                                 fit_failure_spectrum, logical_error_rate_from_ansatz)

BB = REPO_ROOT / "runs" / "framework" / "bb144"

def load_mc():
    """Merge the per-cell MC json files."""
    out = {}
    for f in ("mc_r1.json", "mc_r10.json", "lpu_direct_mc.json", "mc_dinit.json"):
        p = BB / f
        if p.exists():
            out.update(json.loads(p.read_text(encoding="utf-8")))
    return out

MC = load_mc()
print(f"MC cells present: {sorted(MC)}" if MC else "no MC results yet")''')

md(r"""## The memory/output split (direct MC)

Binomial errors on each rate. `F = F_out + F_mem − F_both`. One row per (model,
decoder) at p = 1e-3; the deep600 rows are the fish cells, the campaign rows the local
d_init=12 cross-check cells.""")

code(r'''def se(k, n): return (np.sqrt(k)/n) if (k and n) else (0.0 if n else float("nan"))
rows = [("symmetric (r1)",   "deep600",  "im_r1_deep600@1e-03"),
        ("10x coupler (r10)", "deep600",  "im_r10_deep600@1e-03"),
        ("symmetric (r1)",   "campaign", "im_r1_fast@1e-03"),
        ("10x coupler (r10)", "campaign", "im_r10_fast@1e-03")]
print(f"{'model':18s} {'decoder':9s} {'shots':>6} {'LER_total':>18} {'LER_out':>18} {'LER_mem':>18} {'both':>5}")
mc_summary = {}          # (label, decoder) -> dict
for label, dec, key in rows:
    r = MC.get(key)
    if not r:
        print(f"{label:18s} {dec:9s}  (not finished)"); continue
    n = r["shots"]
    tot, out, mem = r["fails"]/n, r["f_out"]/n, r["f_mem"]/n
    mc_summary[(label, dec)] = dict(n=n, tot=tot, out=out, mem=mem,
                                    F=r["fails"], Fout=r["f_out"], Fmem=r["f_mem"], Fboth=r["f_both"])
    print(f"{label:18s} {dec:9s} {n:>6} "
          f"{tot:.3e}±{se(r['fails'],n):.1e}  "
          f"{out:.3e}±{se(r['f_out'],n):.1e}  "
          f"{mem:.3e}±{se(r['f_mem'],n):.1e}  {r['f_both']:>5}")''')

md(r"""## Coupler sensitivity: does 10× worse Bell fidelity hit output or memory?

The ratio r10/r1 on each channel. If the couplers threaten the **output** parity
(the measurement's job), the output ratio is large and the memory ratio ~1. If they
corrupt **stored memory**, the reverse. This is the study's central question. Uses
the deep600 pair when both are present, else the campaign pair.""")

code(r'''pair = None
for dec in ("deep600", "campaign"):
    a, b = mc_summary.get(("symmetric (r1)", dec)), mc_summary.get(("10x coupler (r10)", dec))
    if a and b: pair = (dec, a, b); break
if pair:
    dec, a, b = pair
    def ratio(x, y): return (y/x) if x else float("inf")
    print(f"decoder: {dec}")
    print(f"{'channel':10s} {'r1':>12} {'r10':>12} {'r10/r1':>8}")
    for ch, k in (("total","tot"), ("output","out"), ("memory","mem")):
        print(f"{ch:10s} {a[k]:12.3e} {b[k]:12.3e} {ratio(a[k],b[k]):8.2f}")
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(3); w = 0.38
    ax.bar(x-w/2, [a["tot"],a["out"],a["mem"]], w, label="symmetric (r1)")
    ax.bar(x+w/2, [b["tot"],b["out"],b["mem"]], w, label="10x coupler (r10)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["total","output","memory"])
    ax.set_ylabel("LER at p=1e-3"); ax.set_title(f"coupler-fidelity sensitivity by channel ({dec})")
    ax.legend(); ax.grid(alpha=0.3, which="both", axis="y"); plt.tight_layout(); plt.show()
else:
    print("need an r1 + r10 MC pair (same decoder) at 1e-3 for the per-channel sensitivity")''')

md(r"""## IS total failure spectra f(w), and reweighted LER vs the MC anchors

The IS cells give the full spectrum (any-observable failure, full both-sector DEM).
Reweighting the *measured* f(w) gives LER(p) with no ansatz. Anchors: filled squares =
deep600 MC totals (only comparable to a deep600 spectrum), open diamonds = campaign
MC totals (the d_init=12 cross-check cells; these should sit on the campaign IS
curves within errors — the IS-vs-MC check). Run discovery prefers a *complete*
deep600 spectrum (result.npz present) and falls back to the fast campaign one.""")

code(r'''def load_spec(sub):
    """Sampled bins + two FailureSpectrum views: `spec` stride-fills the gaps between
    sampled weights (so reweight covers the binomial mass), `raw` is the sampled bins
    only (what the ansatz is fitted to — pooled fill-ins would double-count points)."""
    p = BB / sub / "spectrum.json"
    if not p.exists(): return None
    j = json.loads(p.read_text(encoding="utf-8"))
    cfgp = BB / sub / "config.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8")) if cfgp.exists() else {}
    tw, fw = j["trials_by_weight"], j["failures_by_weight"]
    ws = sorted(int(w) for w in tw)
    tr = [int(tw[str(w)]) for w in ws]; fa = [int(fw[str(w)]) for w in ws]
    meta = dict(n_expanded=int(j["n_expanded"]), q_base=float(j["q_base"]), p_ref=float(j["p_ref"]))
    wf, tf, ff = [], [], []
    for i,(w,t,f) in enumerate(zip(ws,tr,fa)):
        wf.append(w); tf.append(t); ff.append(f)
        if i+1 < len(ws):
            for wm in range(w+1, ws[i+1]):
                wf.append(wm); tf.append(t+tr[i+1]); ff.append(f+fa[i+1])
    return dict(ws=ws, tr=tr, fa=fa, cfg=cfg,
                spec=FailureSpectrum(weights=wf, trials=tf, failures=ff, **meta),
                raw=FailureSpectrum(weights=ws, trials=tr, failures=fa, **meta))

def pick_run(tag):
    """Prefer a COMPLETE deep600 spectrum (result.npz written), else the fast one."""
    for sub in (f"inter_module_{tag}_il_deep600", f"inter_module_{tag}_il_fast"):
        if (BB / sub / "result.npz").exists():
            return sub
    return None

RUNS = {}
for tag, label, col in (("r1", "symmetric (r1)", "C0"), ("r10", "10x coupler (r10)", "C1")):
    sub = pick_run(tag)
    s = load_spec(sub) if sub else None
    if s is None:
        print(f"{label}: no complete IS spectrum yet"); continue
    ns = s["cfg"].get("relay_num_sets", "?")
    s.update(tag=tag, label=label, col=col, sub=sub,
             dec=("deep600" if ns == 600 else f"campaign (num_sets={ns})"))
    RUNS[tag] = s
    print(f"{label}: {sub}  [{s['dec']}, d_init={s['cfg'].get('lpu_d_init','?')}]  "
          f"bins w={s['ws'][0]}..{s['ws'][-1]}, {sum(s['tr'])} trials, {sum(s['fa'])} fails")

def mass_below(s, p):
    """Binomial weight mass at p lying BELOW the lowest sampled weight — the part of
    LER(p) the reweight silently sets to zero. Also returns mu = mean fault count."""
    N, qb, pref = s["spec"].n_expanded, s["spec"].q_base, s["spec"].p_ref
    q = qb * (np.asarray(p, float) / pref)
    return N * q, binom.cdf(s["ws"][0] - 1, N, q)

pg = np.geomspace(1e-4, 5e-3, 60)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
for tag, s in RUNS.items():
    col = s["col"]
    m = [(w, f/t) for w,t,f in zip(s["ws"], s["tr"], s["fa"]) if f > 0]
    axL.plot([w for w,_ in m], [f for _,f in m], "o-", ms=3, lw=1, color=col,
             label=f"{s['label']} [{s['dec']}]")
    rw = reweight_spectrum(s["spec"], pg)
    _, below = mass_below(s, pg)
    ok = below < 0.01
    axR.plot(pg[ok], rw.P_logical[ok], "-", color=col, lw=2, label=f"{s['label']} IS reweight")
    axR.plot(pg[~ok], rw.P_logical[~ok], ":", color=col, lw=1)
# MC anchors — every MC cell for these two models, marker by decoder
for k, r in MC.items():
    if not k.startswith("im_r") or r.get("d_init", 12) != 12: continue
    col = "C1" if k.startswith("im_r10") else "C0"
    mk, mfc = (("s", col) if r.get("decoder") == "deep600" else ("D", "none"))
    axR.plot(r["p"], r["ler"], mk, color=col, mfc=mfc, ms=8, mec="k" if mk == "s" else col, zorder=5)
axL.set_xlabel("fault weight w"); axL.set_ylabel("f(w)"); axL.set_yscale("log")
axL.set_title("IS failure spectrum (total, any of 23 obs)"); axL.grid(alpha=0.3, which="both"); axL.legend(fontsize=8)
axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("physical error rate p")
axR.set_ylabel("LER (per shot)"); axR.set_title("reweighted IS (solid = mass sampled) + MC anchors\n■ deep600 MC   ◇ campaign MC")
axR.grid(alpha=0.3, which="both"); axR.legend(fontsize=8); plt.tight_layout(); plt.show()''')

md(r"""## What is measured and what is extrapolated

The reweight is exact (no ansatz) at any p where the sampled weights cover the
binomial mass. The IS sampler marched down from w=450 and stopped after three
consecutive zero-failure bins at 300 shots (r1 at w=50, r10 at w=59) — *nothing below
the frontier was sampled*. This table gives, per p, the mean fault count μ and the
fraction of the binomial mass that sits under the frontier (which the reweight sets
to zero). Below ~4e-4 the answer is ansatz-only.""")

code(r'''P_TAB = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3, 2e-3]
hdr = f"{'p':>7}"
for s in RUNS.values(): hdr += f"   {s['tag']+' mu':>7} {'below':>6}"
print(hdr + "   verdict")
P_RELIABLE = None
for p in P_TAB:
    line, worst = f"{p:7.1e}", 0.0
    for s in RUNS.values():
        mu, bel = mass_below(s, p); worst = max(worst, float(bel))
        line += f"   {float(mu):7.0f} {float(bel):6.1%}"
    v = "measured" if worst < 0.01 else ("marginal" if worst < 0.1 else "EXTRAPOLATED (ansatz only)")
    if worst < 0.01 and P_RELIABLE is None: P_RELIABLE = p
    print(line + f"   {v}")
print(f"\nlowest p with the mass sampled in every run: {P_RELIABLE}")''')

md(r"""## Ansatz extrapolation toward the 1e-4 regime — paper convention, onset pinned

The paper's f5 ansatz (arXiv:2511.15177 Eq. 10: onset w0, f0=f(w0), power-law ramp
γ₁→γ₂ with crossover wc, saturating at 1−2⁻ᴷ) is fitted to the sampled bins with ≥1
failure and pushed through the binomial sum to p=1e-4.

**The onset is pinned, as in Tour de Gross.** Table 4 of arXiv:2506.03094 gives the
gross-code inter-module measurement a circuit distance **d_circ ≤ 10** (the deformed
code's distance is limited to d−1 = 11 bridge qubits; in-module is also ≤ 10), and
their ansatz "vanishes for w < w0 = ⌈d_circ/2⌉", i.e. **w0 = 5**. A *free* w0 fitted to
these spectra lands at 48–54, which is the 300-shot *detection limit* (f < 1/300),
not a physical onset — so the free fit is shown only as a faint reference, to make
the difference visible. Our own Technique-II probes of this circuit were inconclusive
(time-like boundary artifacts), so ≤10 is the paper's bound; a true d_circ of 8–9
would move the pin to 4–5.

**What pinning does with *these* data.** Every sampled bin is at w ≥ 50, so f0 = f(5)
and the whole ramp from w=5 to 50 are the power law extrapolated downward from the
measured w ≥ 53 shape — assumed, not measured (the paper had bins near its onset).
The bootstrap band (resample each bin's failures ~Binom(T, f̂), refit warm-started,
16–84 %) is therefore only the *statistical* part; the systematic question — is the
w≥53 power law the right shape down to w=5 — is not in the band. The printed
f_pinned(w) at w = 10, 20, 30, 40 is what a future low-weight probe should be compared
against.

**The pessimistic bracket stays.** The campaign relay is known to miscorrect at low
weight (Y1: w=3–6 at ~1/400 each). If f(w) below the frontier is a flat decoder floor
rather than the ramp, LER(p) → f_floor·P(W ≥ w_min) ≈ f_floor at low p; the dotted
horizontal lines are the 95 % bound on such a floor from the pooled zero-failure bins
at the frontier (0/900 → 3.3e-3). **What would settle it:** direct low-weight bins —
resolving a 1e-3 floor needs ~3000 shots per bin at ~5 s/decode, so a 4-weight probe
(w = 10, 20, 30, 40) is ~12 h locally; a full w=5..45 fill is a fish job; deep600
removes the floor but is ~2× slower per shot here.

**Ratios under extrapolation.** Independently fitted shapes amplify differences
exponentially at low p (the `reweight_spectrum` docstring's warning), so read the
extrapolated r10/r1 with its band and prefer the measured ratio (solid, ≥4e-4).""")

code(r'''def ler_ansatz(fit, p):
    """LER(p) = sum_w Binom(w; N, q(p)) f_ansatz(w) — same as the library's
    logical_error_rate_from_ansatz but with the weight sum truncated where the binomial
    mass ends (w <= mu + 12 sigma). The library sums to N_expanded = 2.9e6 rows and
    takes ~10 s per call; this is instant, which the bootstrap needs."""
    p = np.atleast_1d(np.asarray(p, float)); N = fit.n_expanded
    q = np.clip(fit.q_base * (p / fit.p_ref), 1e-300, 1 - 1e-15)
    mu_max = N * q.max(); w_hi = int(min(N, mu_max + 12 * np.sqrt(mu_max) + 50))
    w = np.arange(int(np.ceil(fit.params["w0"])), w_hi + 1, dtype=float)
    logb = gammaln(N + 1) - gammaln(w + 1) - gammaln(N - w + 1)
    logt = logb[:, None] + w[:, None] * np.log(q)[None, :] + (N - w)[:, None] * np.log1p(-q)[None, :]
    return (fit.f(w)[:, None] * np.exp(logt)).sum(axis=0)

K_OBS  = 23
D_CIRC = 10                       # Tour de Gross Table 4: inter-module gross d_circ <= 10
W0_PIN = float(np.ceil(D_CIRC / 2))   # = 5, the paper's ansatz onset
N_BOOT = 100                      # warm-started f5 refits per run (each ~0.05 s + one LER curve)
rng = np.random.default_rng(7)
ANS = {}
for tag, s in RUNS.items():
    raw = s["raw"]
    # PRIMARY: onset pinned at w0 = ceil(d_circ/2) (multistart; nothing to warm-start from)
    fit = fit_failure_spectrum(raw, K_OBS, model="f5", w0=W0_PIN, f0=None)
    P_fit = ler_ansatz(fit, pg)
    # REFERENCE: free w0 (warm-started from the framework's own fit when present)
    fj = BB / s["sub"] / "ansatz_fit.json"
    fw = json.loads(fj.read_text(encoding="utf-8"))["params"] if fj.exists() else None
    fit_free = fit_failure_spectrum(raw, K_OBS, model="f5", init_params=fw)
    P_free = ler_ansatz(fit_free, pg)
    if tag == "r1":   # one-time check of the truncated sum against the library, at p=1e-3
        lib = float(logical_error_rate_from_ansatz(fit, [1e-3])[0]); mine = float(ler_ansatz(fit, [1e-3])[0])
        print(f"truncated-sum check at 1e-3: library {lib:.4e}  here {mine:.4e}  (rel diff {abs(lib-mine)/lib:.1e})")
    tr, fa = np.array(s["tr"]), np.array(s["fa"])
    boots = []
    for _ in range(N_BOOT):
        fb = rng.binomial(tr, np.clip(fa / tr, 0, 1))
        try:
            bfit = fit_failure_spectrum(
                FailureSpectrum(weights=s["ws"], trials=list(tr), failures=list(fb),
                                n_expanded=raw.n_expanded, q_base=raw.q_base, p_ref=raw.p_ref),
                K_OBS, model="f5", w0=W0_PIN, f0=None, init_params=fit.params)
            boots.append(ler_ansatz(bfit, pg))
        except (ValueError, RuntimeError):
            pass
    boots = np.array(boots)
    # decoder-floor bracket: 95% bound on a flat f below the frontier from the pooled
    # zero-failure bins at the frontier (the three stop-rule bins)
    T0 = sum(t for w, t, f in zip(s["ws"], s["tr"], s["fa"]) if f == 0 and w < s["ws"][0] + 5)
    ANS[tag] = dict(fit=fit, P=P_fit, boots=boots, fit_free=fit_free, P_free=P_free,
                    lo=(np.percentile(boots, 16, axis=0) if len(boots) else None),
                    hi=(np.percentile(boots, 84, axis=0) if len(boots) else None),
                    f_floor95=(3.0 / T0 if T0 else float("nan")))
    pp, pf = fit.params, fit_free.params
    print(f"\n{s['label']} [{s['dec']}]")
    print(f"   PINNED w0={pp['w0']:.0f}:  f0=f({pp['w0']:.0f})={pp['f0']:.2e}  gamma1={pp['gamma1']:.2f}  "
          f"gamma2={pp['gamma2']:.2f}  wc={pp['wc']:.0f}   (n={fit.n_points}, cost={fit.cost:.0f}, "
          f"{len(boots)}/{N_BOOT} boots converged)")
    print(f"   free w0 (ref):  w0={pf['w0']:.1f}  f0={pf['f0']:.2e}  gamma1={pf['gamma1']:.2f}  "
          f"gamma2={pf['gamma2']:.2f}  wc={pf['wc']:.0f}   (cost={fit_free.cost:.0f})")
    wprobe = np.array([10, 20, 30, 40, s["ws"][0]], float)
    print("   pinned f(w) below the frontier: " +
          "  ".join(f"f({int(w)})={v:.1e}" for w, v in zip(wprobe, fit.f(wprobe))) +
          f"   | flat-floor 95% bound: f <= {ANS[tag]['f_floor95']:.1e}")''')

code(r'''fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
reliable = np.ones_like(pg, bool)
for tag, s in RUNS.items():
    col, a = s["col"], ANS[tag]
    rw = reweight_spectrum(s["spec"], pg)
    _, below = mass_below(s, pg); ok = below < 0.01; reliable &= ok
    axL.plot(pg[ok], rw.P_logical[ok], "-", color=col, lw=2.2, label=f"{s['label']} measured")
    axL.plot(pg, a["P"], "--", color=col, lw=1.6, label=f"{s['label']} f5, w0 pinned = {W0_PIN:.0f}")
    if a["lo"] is not None:
        axL.fill_between(pg, np.maximum(a["lo"], 1e-16), a["hi"], color=col, alpha=0.15)
    axL.plot(pg, a["P_free"], "-.", color=col, lw=0.9, alpha=0.5, label=f"{s['label']} f5, free w0 (ref)")
    axL.axhline(a["f_floor95"], color=col, ls=":", lw=1.2)
if reliable.any():
    axL.axvspan(pg[0], pg[reliable].min(), color="grey", alpha=0.10)
    axL.text(pg[0]*1.1, 3e-2, "ansatz only\n(mass below\nsampled frontier)", fontsize=8, color="0.3")
axL.set_xscale("log"); axL.set_yscale("log"); axL.set_ylim(1e-14, 1.5)
axL.set_xlabel("physical error rate p"); axL.set_ylabel("LER (per shot)")
axL.set_title(f"measured reweight (solid) vs f5 extrapolation, w0 pinned at {W0_PIN:.0f} (dashed, 16–84% band)\n"
              "dash-dot: free-w0 fit (detection-limit onset); dotted: 95% bound if f is a flat floor below the frontier",
              fontsize=9)
axL.grid(alpha=0.3, which="both"); axL.legend(fontsize=7, loc="lower right")

if {"r1", "r10"} <= set(RUNS):
    r1, r10 = reweight_spectrum(RUNS["r1"]["spec"], pg), reweight_spectrum(RUNS["r10"]["spec"], pg)
    with np.errstate(divide="ignore", invalid="ignore"):
        meas = r10.P_logical / r1.P_logical
        ans = ANS["r10"]["P"] / ANS["r1"]["P"]
        ans_free = ANS["r10"]["P_free"] / ANS["r1"]["P_free"]
    axR.plot(pg[reliable], meas[reliable], "-", color="k", lw=2.2, label="measured (both spectra cover the mass)")
    axR.plot(pg, ans, "--", color="C3", lw=1.6, label=f"f5 ratio, w0 pinned = {W0_PIN:.0f}")
    axR.plot(pg, ans_free, "-.", color="C3", lw=0.9, alpha=0.5, label="f5 ratio, free w0 (ref; inverts = fit artifact)")
    b1, b10 = ANS["r1"]["boots"], ANS["r10"]["boots"]
    n = min(len(b1), len(b10))
    if n > 10:
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = b10[:n] / b1[:n]
        axR.fill_between(pg, np.nanpercentile(rb, 16, axis=0), np.nanpercentile(rb, 84, axis=0),
                         color="C3", alpha=0.15)
    axR.axhline(1.0, color="0.5", lw=1)
    if reliable.any(): axR.axvspan(pg[0], pg[reliable].min(), color="grey", alpha=0.10)
    axR.set_xscale("log"); axR.set_xlabel("physical error rate p"); axR.set_ylabel("LER(r10) / LER(r1)")
    axR.set_ylim(0, 3); axR.set_title("coupler sensitivity: 10× worse Bell fidelity → total LER ratio")
    axR.grid(alpha=0.3, which="both"); axR.legend(fontsize=8)
plt.tight_layout(); plt.show()

# the numbers
print(f"{'p':>7} " + " ".join(f"{'LER '+t+' meas':>14} {'f5 pin':>10} {'f5 free':>10}" for t in RUNS) +
      f" {'ratio meas':>10} {'ratio pin':>9}  status")
for p in P_TAB:
    line = f"{p:7.1e} "
    vals = {}
    for tag, s in RUNS.items():
        mu, bel = mass_below(s, p)
        m = float(reweight_spectrum(s["spec"], [p]).P_logical[0])
        a = float(ler_ansatz(ANS[tag]["fit"], [p])[0]); af = float(ler_ansatz(ANS[tag]["fit_free"], [p])[0])
        vals[tag] = (m, a, float(bel))
        line += f" {(f'{m:.3e}' if bel < 0.01 else '   (not covered)'):>14} {a:10.2e} {af:10.2e}"
    if {"r1", "r10"} <= set(vals):
        m1, a1, b1_ = vals["r1"]; m10, a10, b10_ = vals["r10"]
        cov = max(b1_, b10_) < 0.01
        line += f" {(f'{m10/m1:.2f}' if cov and m1 else '—'):>10} {(a10/a1 if a1 else float('nan')):9.2f}"
        line += "  measured" if cov else "  extrapolated (pinned-onset ansatz)"
    print(line)''')

md(r"""## The d_init sweep: merge cost vs idle cost

The production circuit pads the C=10 merged rounds with d_init=12 bare rounds on each
side, so its LER mixes the **merge** (the thing we want) with **24 rounds of idle
memory**. The literature's four conventions for separating them are: no padding
(Tour de Gross per-operation numbers), d-padding decoded holistically, post-hoc
spacetime attribution, or a **padding sweep** — vary d_init and fit LER at fixed p
linearly in the number of bare rounds: the intercept at zero padding is the merge-only
cost, the slope the per-idle-round cost. This section does the sweep from the direct
MC cells (`mc_dinit.json`: campaign decoder, d_init ∈ {12, 6}, p ∈ {1e-3, 7e-4, 5e-4}),
overlaying the IS-reweighted d_init=12 value as a cross-check.

**Two confounds, stated up front.** Fewer padding rounds removes idle faults (LER
down, ~linear in rounds) but *also* removes decoding context after the merge — the
sliding-window buffer argument — which raises the merge's own failure rate once
d_init falls below the code distance, so a two-point intercept biases the merge cost
**upward**. And the time-like distance shrinks with d_init, so the sweep is meaningful
in the measured window (p ≥ 4e-4, where failures are volume-dominated), not for
extrapolation.

**What the data showed (2026-09-22, campaign decoder).** The linear-in-rounds model
fails outright: LER(d12)/LER(d6) is 3, 7 and 21 at p = 1e-3, 7e-4, 5e-4, where a
volume model allows at most ~2. Every intercept is negative. Halving the padding
removed far more failures than the removed rounds could have produced, so the
per-round failure rate itself grows with circuit length — the campaign relay
(num_sets=20) is decoder-limited on the 34-round circuit at these p (consistent with
the IS diagnostic that its mean failing weight equals μ, i.e. it sits at threshold).
The cell therefore reports the intercept only with a validity verdict, and gives what
the sweep does establish: **LER(d6) is an upper bound on the merge-only cost**, and
the **coupler sensitivity at reduced padding** (r10/r1 at d6 ≈ 2.4, 3.1, ~11 at 1e-3,
7e-4, 5e-4 vs 1.2, 1.5, 1.5 at d12) — the padding was diluting the coupler effect.
The way to a real merge cost is a decoder that is not length-limited (deep600 at d6
on fish) or the paper's no-padding convention, not a third d_init with this decoder.""")

code(r'''def sweep_rows(coupler):
    """[(p, d_init, fails, shots, ler, ler_out, ler_mem)] for the campaign-decoder MC cells."""
    out = []
    for k, r in MC.items():
        if r.get("decoder") != "campaign" or r.get("d_init") is None: continue
        if r.get("coupler_factor") != coupler or not k.startswith("im_r"): continue
        out.append((r["p"], r["d_init"], r["fails"], r["shots"], r["ler"], r["ler_out"], r["ler_mem"]))
    return sorted(out)

SWEEP = {"r1": sweep_rows(1), "r10": sweep_rows(10)}
have = {t: sorted({d for _, d, *_ in rows}) for t, rows in SWEEP.items()}
print("d_init values present per model:", have)

print(f"\n{'model':5s} {'p':>7} {'d_init':>6} {'rounds':>6} {'fails/shots':>12} {'LER':>10} {'LER_out':>10} {'LER_mem':>10}")
for t, rows in SWEEP.items():
    for p, d, F, n, ler, lo, lm in rows:
        print(f"{t:5s} {p:7.1e} {d:6d} {2*d+10:6d} {F:5d}/{n:<6d} {ler:10.3e} {lo:10.3e} {lm:10.3e}")

def point(t, p, d):
    """(LER, se, source) at (model, p, d_init): MC cell if present, else — for d_init=12 —
    the IS reweight where the mass is sampled. None if neither."""
    for pp, dd, F, n, ler, *_ in SWEEP[t]:
        if abs(pp - p) < 1e-12 and dd == d:
            return ler, np.sqrt(max(F, 1)) / n, "MC"
    if d == 12 and t in RUNS and RUNS[t]["cfg"].get("lpu_d_init", 12) == 12:
        _, bel = mass_below(RUNS[t], p)
        if bel < 0.01:
            rw = reweight_spectrum(RUNS[t]["spec"], [p])
            return float(rw.P_logical[0]), float(rw.P_logical_se[0]), "IS"
    return None

INTERCEPTS = {}
ps = sorted({p for rows in SWEEP.values() for p, *_ in rows})
DS = sorted({d for rows in SWEEP.values() for _, d, *_ in rows} | {12})
if ps:
    fig, axes = plt.subplots(1, len(ps), figsize=(4.6*len(ps), 4.2), squeeze=False)
    for ax, p in zip(axes[0], ps):
        for t, col in (("r1", "C0"), ("r10", "C1")):
            pts = [(d, point(t, p, d)) for d in DS]
            pts = [(2*d, v[0], v[1], v[2]) for d, v in pts if v]
            if not pts: continue
            for x_, y_, e_, src in pts:
                ax.errorbar([x_], [y_], yerr=[e_], fmt=("o" if src == "MC" else "x"), color=col,
                            capsize=3, ms=(6 if src == "MC" else 9), mew=2,
                            label=f"{t} {src}" if src == "IS" or x_ == min(q[0] for q in pts) else None)
            x, y, e = (np.array([q[i] for q in pts]) for i in range(3))
            if len(np.unique(x)) >= 2:
                W = 1.0 / np.maximum(e, 1e-12)
                slope, icpt = np.polyfit(x, y, 1, w=W)
                xx = np.linspace(0, 26, 20); ax.plot(xx, icpt + slope*xx, "--", color=col, lw=1)
                INTERCEPTS[(t, p)] = (icpt, slope)
                ax.plot(0, icpt, "s", color=col, mfc="none", ms=8)
        ax.set_title(f"p = {p:.0e}"); ax.set_xlabel("bare (idle) rounds = 2·d_init"); ax.set_ylabel("LER (total)")
        ax.set_yscale("log"); ax.set_xlim(-1, 26); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    plt.suptitle("d_init sweep (log y): dashed = weighted linear fit in rounds; open square = its intercept (merge-only IF linear)")
    plt.tight_layout(); plt.show()

# --- is LER linear in rounds at all?  volume model: LER(d12)/LER(d6) <= 34/22 = 1.55 (all rounds)
#     or 24/12 = 2 (bare rounds only). Much larger => the per-round failure rate itself grows
#     with circuit length (decoder-limited regime), and a linear intercept is meaningless.
print(f"\n{'p':>7} {'model':5s} {'LER d12':>10} {'src':>3} {'LER d6':>10} {'d12/d6':>7}  verdict")
LINEAR_OK = {}
for p in ps:
    for t in ("r1", "r10"):
        a, b = point(t, p, 12), point(t, p, 6)
        if not (a and b): continue
        r = a[0] / b[0] if b[0] > 0 else float("inf")
        ok = r <= 2.2
        LINEAR_OK[(t, p)] = ok
        print(f"{p:7.1e} {t:5s} {a[0]:10.3e} {a[2]:>3} {b[0]:10.3e} {r:7.1f}  "
              f"{'volume-like: linear model usable' if ok else 'per-round failure GROWS with length -> linear intercept invalid'}")

if INTERCEPTS:
    print(f"\n{'p':>7} {'model':5s} {'intercept (merge-only if linear)':>32} {'slope / bare round':>18}  status")
    for p in ps:
        for t in ("r1", "r10"):
            v = INTERCEPTS.get((t, p))
            if not v: continue
            icpt, slope = v
            st = ("OK" if (icpt > 0 and LINEAR_OK.get((t, p), False)) else
                  "NEGATIVE — not a merge cost" if icpt <= 0 else "linear model not supported by the d12/d6 ratio")
            print(f"{p:7.1e} {t:5s} {icpt:32.3e} {slope:18.3e}  {st}")

# --- what the sweep DOES establish: bounds and the coupler sensitivity at reduced padding
print("\nUpper bound on the merge-only cost (adding rounds only adds failures): merge-only <= LER(d_init=6)")
print(f"{'p':>7} {'r1: LER(d6)':>12} {'r10: LER(d6)':>13} {'r10/r1':>7} {'(±)':>6}   {'r10/r1 at d12':>14}   channel split at d6 (out / mem), r10/r1")
for p in ps:
    a, b = point("r1", p, 6), point("r10", p, 6)
    if not (a and b): continue
    ratio = b[0] / a[0] if a[0] > 0 else float("inf")
    rel = np.sqrt((a[1]/a[0])**2 + (b[1]/b[0])**2) if a[0] > 0 and b[0] > 0 else float("nan")
    a12, b12 = point("r1", p, 12), point("r10", p, 12)
    r12 = f"{b12[0]/a12[0]:.2f} ({b12[2]})" if (a12 and b12 and a12[0] > 0) else "—"
    # channel split from the raw MC rows
    def split(t):
        for pp, d, F, n, ler, lo, lm in SWEEP[t]:
            if abs(pp - p) < 1e-12 and d == 6: return lo, lm, n
        return None
    s1, s10 = split("r1"), split("r10")
    ch = ""
    if s1 and s10:
        ro = (s10[0]/s1[0]) if s1[0] > 0 else float("inf"); rm = (s10[1]/s1[1]) if s1[1] > 0 else float("inf")
        ch = f"out {s1[0]:.1e}->{s10[0]:.1e} (×{ro:.1f})   mem {s1[1]:.1e}->{s10[1]:.1e} (×{rm:.1f})"
    print(f"{p:7.1e} {a[0]:12.3e} {b[0]:13.3e} {ratio:7.2f} {ratio*rel:6.2f}   {r12:>14}   {ch}")
if not ps:
    print("\nneed MC cells (mc_dinit.json) for the sweep")''')

md(r"""## Reading it

* **The sensitivity table/bars are the headline** — the `r10/r1` ratio per channel
  says whether 10× worse Bell couplers threaten the measurement **output**, the
  **stored memory**, or both.
* **Measured coupler sensitivity (IS reweight, campaign decoder, d_init=12):**
  r10/r1 ≈ 1.7 at 4e-4, 1.5 at 5e-4–7e-4, 1.2 at 1e-3, →1 as both saturate. It grows
  as p falls because the bulk memory term dies faster than the O(d) coupler seam.
* **Squares/diamonds on the reweight panel are MC totals**; a diamond off its
  campaign curve means the IS-vs-MC cross-check failed (suspect the DEM sector or the
  reweight) — they should agree within errors.
* **The extrapolation uses the paper's pinned onset w0 = ⌈d_circ/2⌉ = 5.** Its band
  is statistical only: the ramp from w=5 to the w=50 frontier is the measured power
  law extrapolated, and the flat-floor bound is the pessimistic alternative. Quote the
  measured window (≥4e-4) as fact and the pinned extrapolation as the paper-convention
  estimate; low-weight bins would close the gap.
* **The d_init sweep did not yield a merge cost with this decoder**: LER is not linear
  in rounds (d12/d6 ratios of 3–21 vs ≤2 for a volume model), so the intercepts are
  negative and meaningless. What it does give: LER(d6) as an upper bound on the merge
  cost, and a coupler sensitivity of 2.4–3× (1e-3, 7e-4) rising to ~11× at 5e-4 once
  half the idle padding is removed — versus 1.2–1.5× with full padding.
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
