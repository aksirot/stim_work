"""Splitting vs direct MC on the [[72,4,8]] full-symmetric model, down to p = 1e-4.

The paper-faithful ladder (Alg. 2/3 + eq18, `splitting_paper_72.py`) PASSED arbitration
at 2e-3 under ghw_deep — but the window stopped where direct MC stopped being affordable
under a slow decoder. This driver swaps in **ghw_nc1** (ghw with stop_nconv=1: kept all
71/71 sub-onset harvest fixes at 3.5-1500x the speed, 2026-07-28 bench) on BOTH sides of
the comparison. Method agreement is a property of the estimator pair, not of the decoder
— any FIXED decoder is a valid referee — and the fast decoder helps twice over: the
ladder runs ~100x faster, and its higher above-onset LER pulls the direct-MC overlap
points down into 1e-4 territory.

Failure convention (both estimators): predicted obs != actual obs; empty-syndrome shots
are not decoded — their prediction is 0, so an undetected logical flip still counts.

Stages, each cached as its own JSON under runs/splitting_crosscheck/fast_ladder/:

  probe    quick direct MC at 8e-3 + 2e-3: measured LER + dec/s, then projected shot
           budgets for the deeper checkpoints (extrapolating the local slope)
  ladder   multi-seeded eq18 ladder 8e-3 -> 1e-4 (multi_seeded_split_estimate)
  mc       direct-MC checkpoints; MC_PS env picks the p list (default "8e-4,3e-4" —
           add "1e-4" once the probe confirms its cost)
  report   ladder interpolated (log-log) onto every finished checkpoint, z-score per
           point, PASS/FAIL

    python experiments/methods/splitting_fast_ladder_72.py probe
    python experiments/methods/splitting_fast_ladder_72.py ladder
    MC_PS=8e-4,3e-4 python experiments/methods/splitting_fast_ladder_72.py mc
    MC_PS=1e-4 MC_TARGET=15 python experiments/methods/splitting_fast_ladder_72.py mc
    python experiments/methods/splitting_fast_ladder_72.py report

Env knobs: FL_DECODER (DECODER_VARIANTS key, default ghw_nc1), MC_PS, MC_TARGET
(failures per checkpoint, default 25), MC_SHOTS_MAX (default 5e8), FL_SEED (default 7),
SMOKE=1 (tiny budgets everywhere: wiring check only).
"""
import json
import os
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS", "EMC_CALIB"):
    os.environ.pop(_k, None)
import run_error_model_comparison as rmc
from importance_sampling import (FailureSpectrum, importance_sample_adaptive,
                                 reweight_spectrum)
from splitting import multi_seeded_split_estimate
from repo_paths import REPO_ROOT

OUT = REPO_ROOT / "runs" / "splitting_crosscheck" / "fast_ladder"
SMOKE = bool(int(os.environ.get("SMOKE", "0")))
SEED = int(os.environ.get("FL_SEED", "7"))
DEC_NAME = os.environ.get("FL_DECODER", "ghw_nc1")
DEC_CFG = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS[DEC_NAME])
# 4e-3, not the old window's 8e-3: the 2026-08-05 smoke probe measured nc1 at
# LER=0.38 and 4 dec/s at 8e-3 — the saturation shoulder, dense syndromes, and the
# eq18 rungs there cost ~100x their scaling-regime price. 4e-3 is still MC-trivial.
P_HIGH, P_LOW = 4e-3, 1e-4


def make_decoder(circ):
    """Device-convention decoder: priors frozen from full symmetric at DECODER_P."""
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
    dec = rmc.DEC(calib, DEC_CFG)
    dec.setup(circ)
    return dec


def direct_mc(p, target_fails, shots_max, batch=None, seed=SEED):
    """Direct MC at physical rate p. Returns dict with fails/shots/ler/se/dec rate."""
    if batch is None:
        batch = 1024 if p >= 2e-3 else 8192   # dense syndromes decode ~100x slower
    log_every = batch * (4 if p >= 2e-3 else 64)
    circ = rmc.make_circuit72("full symmetric", p)
    dec = make_decoder(circ)
    sampler = circ.compile_detector_sampler(seed=seed)
    fails = shots = decoded = 0
    t0 = time.time()
    dec_t = 0.0
    while fails < target_fails and shots < shots_max:
        dets, obs = sampler.sample(batch, separate_observables=True)
        nz = dets.any(axis=1)
        # empty syndrome: prediction is 0, an actual flip is an (undetected) failure
        fails += int(obs[~nz].any(axis=1).sum())
        if nz.any():
            td = time.time()
            preds = dec.decode_batch(dets[nz])
            dec_t += time.time() - td
            decoded += int(nz.sum())
            fails += int((preds.astype(bool) != obs[nz].astype(bool)).any(axis=1).sum())
        shots += batch
        if shots % log_every == 0:
            print(f"    p={p:.1e}  {fails} fails / {shots:,} shots "
                  f"({decoded:,} decoded, {decoded / max(dec_t, 1e-9):,.0f} dec/s)",
                  flush=True)
    ler = fails / shots if shots else float("nan")
    return dict(p=p, fails=fails, shots=shots, ler=ler,
                se_rel=(1.0 / np.sqrt(fails) if fails else None),
                decoded=decoded, dec_per_s=decoded / max(dec_t, 1e-9),
                nonzero_frac=decoded / shots if shots else None,
                elapsed_s=time.time() - t0)


def stage_probe():
    """Fast MC at the two easy rates; project budgets for the deep checkpoints."""
    budgets = ([(P_HIGH, 20, 4_000), (2e-3, 10, 20_000)] if SMOKE else
               [(P_HIGH, 200, 100_000), (2e-3, 100, 1_500_000)])
    pts = []
    for p, tgt, cap in budgets:
        print(f"[probe] MC at p={p:.1e} (target {tgt} fails, cap {cap:,})", flush=True)
        r = direct_mc(p, tgt, cap)
        print(f"[probe]   -> LER = {r['ler']:.3e} ({r['fails']}/{r['shots']:,}), "
              f"{r['dec_per_s']:,.0f} dec/s", flush=True)
        pts.append(r)
    # local slope -> projected LER and cost at the deep checkpoints
    (pa, la), (pb, lb) = [(q["p"], q["ler"]) for q in pts]
    slope = float(np.log(la / lb) / np.log(pa / pb))
    target = int(os.environ.get("MC_TARGET", "25"))
    proj = {}
    for p in (8e-4, 3e-4, 1e-4):
        ler = lb * (p / pb) ** slope
        shots = target / ler
        hours = shots * pts[1]["nonzero_frac"] / pts[1]["dec_per_s"] / 3600
        proj[f"{p:.0e}"] = dict(ler_proj=ler, shots_for_target=shots, cpu_hours=hours)
        print(f"[probe] projected p={p:.1e}: LER~{ler:.2e}, {target} fails ~ "
              f"{shots:.2e} shots ~ {hours:.1f} cpu-h", flush=True)
    out = dict(decoder=DEC_NAME, decoder_cfg={k: list(v) if isinstance(v, tuple) else v
                                              for k, v in DEC_CFG.items()},
               calibrated_at=rmc.DECODER_P, points=pts, slope=slope, projections=proj,
               smoke=SMOKE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'probe.json'}", flush=True)


def stage_ladder():
    """Paper ladder, extended: eq18 from 4e-3 all the way to 1e-4.

    Budgets are env-overridable for the fish paper-scale test (2026-08-18): the
    laptop run at T_init=4k FAILED arbitration below ~2e-3 with a frozen weight
    distribution; the paper's own budget is T_init=1e6. FL_TAG names this shard's
    output (ladder_<tag>.json), so N nohup'd processes with distinct FL_SEEDs
    shard the L*M instances across a node; the shard-combiner pools them.
    """
    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    dec = make_decoder(circ)
    kw = (dict(L=2, M=2, T_init=300, T_cap=600, anchor_shots=1_000) if SMOKE else
          dict(L=int(os.environ.get("FL_L", "6")),
               M=int(os.environ.get("FL_M", "3")),
               T_init=int(float(os.environ.get("FL_TINIT", "4000"))),
               T_cap=int(float(os.environ.get("FL_TCAP", "16000"))),
               anchor_shots=int(float(os.environ.get("FL_ANCHOR", "20000")))))
    t0 = time.time()
    res, diag = multi_seeded_split_estimate(
        circ, dec, p_ref=rmc.P_REF, p_high=P_HIGH, p_low=P_LOW,
        eps=0.3, ladder="eq18", distance=8, seed=SEED, **kw)
    out = dict(decoder=DEC_NAME, algorithm="multi_seeded_split_estimate eq18",
               p_ref=rmc.P_REF, calibrated_at=rmc.DECODER_P,
               p_high=P_HIGH, p_low=P_LOW, budgets=kw, smoke=SMOKE,
               sp=np.asarray(res.p_ladder).tolist(),
               sP=np.asarray(res.P_logical).tolist(),
               sP_se=np.asarray(res.P_logical_se).tolist(),
               elapsed_s=time.time() - t0,
               diag={k: v for k, v in diag.items()
                     if isinstance(v, (int, float, str, list, dict))})
    OUT.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("FL_TAG", "")
    fname = f"ladder_{tag}.json" if tag else "ladder.json"
    (OUT / fname).write_text(json.dumps(out, indent=1, default=str),
                             encoding="utf-8")
    print(f"[ladder] S({P_LOW:.0e}) = {out['sP'][-1]:.3e}  "
          f"({len(out['sp'])} rungs, {out['elapsed_s'] / 3600:.2f} h)", flush=True)
    print(f"wrote {OUT / fname}", flush=True)


def stage_mc():
    """Direct-MC checkpoints at the MC_PS list."""
    ps = [float(s) for s in os.environ.get("MC_PS", "8e-4,3e-4").split(",")]
    target = int(os.environ.get("MC_TARGET", "25"))
    shots_max = int(float(os.environ.get("MC_SHOTS_MAX", "5e8")))
    if SMOKE:
        target, shots_max = 3, 100_000
    OUT.mkdir(parents=True, exist_ok=True)
    for p in ps:
        print(f"[mc] checkpoint p={p:.1e} (target {target} fails, cap {shots_max:,})",
              flush=True)
        r = direct_mc(p, target, shots_max)
        r.update(decoder=DEC_NAME, calibrated_at=rmc.DECODER_P, smoke=SMOKE)
        path = OUT / f"mc_{p:.0e}.json"
        path.write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"[mc]   -> LER = {r['ler']:.3e} ({r['fails']}/{r['shots']:,})  "
              f"wrote {path.name}", flush=True)


def stage_is():
    """Weight-stratified IS spectrum with the SAME decoder; reweighted to the ladder grid.

    The third estimator (2026-08-05, replacing the cancelled deep-MC checkpoints):
    IS-reweighting and splitting are different estimator FAMILIES sharing only the
    decoder, so their agreement at depth is genuine cross-validation, with the two
    cheap MC points arbitrating the top. Binomial reweighting of the measured f(w) is
    rate-independent — one spectrum serves every p. Zero bins are a LOW bias (the
    EMC lesson), so the report prints the lightest measured-failing weight alongside.
    """
    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    dec = make_decoder(circ)
    ws = [8, 16] if SMOKE else list(range(3, 57, 2))
    p_grid = sorted(set(np.geomspace(P_LOW, P_HIGH, 13).tolist() + [2e-3, 8e-4]))
    kw = (dict(target_failures=5, shots_max=2_000) if SMOKE else
          dict(target_failures=30, shots_max=300_000))
    t0 = time.time()
    res = importance_sample_adaptive(circ, dec, p_ref=rmc.P_REF, p_values=p_grid,
                                     weights=ws, seed=SEED, **kw)
    sp = res.spectrum
    meas = [w for w, f in zip(sp.weights, sp.failures) if f > 0]
    out = dict(decoder=DEC_NAME, calibrated_at=rmc.DECODER_P, p_ref=rmc.P_REF,
               smoke=SMOKE, budgets=kw, elapsed_s=time.time() - t0,
               p_values=list(map(float, res.p_values)),
               P_logical=list(map(float, res.P_logical)),
               P_logical_se=list(map(float, res.P_logical_se)),
               spectrum=dict(weights=list(map(int, sp.weights)),
                             trials=list(map(int, sp.trials)),
                             failures=list(map(int, sp.failures)),
                             n_expanded=int(sp.n_expanded), q_base=float(sp.q_base),
                             p_ref=float(sp.p_ref)),
               w_meas_min=(min(meas) if meas else None))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "is.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[is] lightest measured-failing weight: {out['w_meas_min']}  "
          f"LER({P_LOW:.0e}) = {out['P_logical'][0]:.3e}  "
          f"({out['elapsed_s'] / 3600:.2f} h)", flush=True)
    print(f"wrote {OUT / 'is.json'}", flush=True)


def stage_report():
    """z-score of the ladder against every finished MC checkpoint."""
    lad = json.loads((OUT / "ladder.json").read_text(encoding="utf-8"))
    sp = np.asarray(lad["sp"], float)
    sP = np.asarray(lad["sP"], float)
    sse = np.asarray(lad["sP_se"], float)
    order = np.argsort(sp)
    lp, lP = np.log(sp[order]), np.log(sP[order])
    rel = (sse / np.clip(sP, 1e-300, None))[order]
    print(f"ladder: {len(sp)} rungs, {lad['p_high']:.0e} -> {lad['p_low']:.0e}, "
          f"decoder {lad['decoder']}")
    rows, worst = [], 0.0
    for f in sorted(OUT.glob("mc_*.json")):
        mc = json.loads(f.read_text(encoding="utf-8"))
        if not mc["fails"]:
            print(f"  {f.name}: 0 fails in {mc['shots']:,} shots — bound only, skipped")
            continue
        x = np.log(mc["p"])
        S = float(np.exp(np.interp(x, lp, lP)))
        Srel = float(np.interp(x, lp, rel))
        z = abs(np.log(S) - np.log(mc["ler"])) / np.hypot(Srel, mc["se_rel"])
        verdict = "PASS" if z < 3 else "FAIL"
        worst = max(worst, z)
        rows.append(dict(p=mc["p"], mc_ler=mc["ler"], mc_fails=mc["fails"],
                         mc_shots=mc["shots"], ladder_S=S, z=z, verdict=verdict))
        print(f"  p={mc['p']:.1e}  MC {mc['ler']:.3e} ({mc['fails']}/{mc['shots']:,})"
              f"  ladder {S:.3e}  z={z:.2f}  {verdict}")
    # --- ladder vs IS reweighting, pointwise down the whole grid ---
    is_rows = []
    is_path = OUT / "is.json"
    if is_path.exists():
        isr = json.loads(is_path.read_text(encoding="utf-8"))
        # STRIDE FIX (2026-08-05): the IS sweep samples stride-2 weights, and
        # reweight_spectrum sums only sampled weights — the stored P_logical
        # undercounts ~2x. Pool-fill the gaps (emc_report.fill_spectrum rule:
        # each missing weight carries its bracketing bins' pooled counts) and
        # recompute; the cached spectrum makes this free.
        sd = isr["spectrum"]
        W_, T_, F_ = list(sd["weights"]), list(sd["trials"]), list(sd["failures"])
        wf, tf, ff = [], [], []
        for i, (w_i, t_i, f_i) in enumerate(zip(W_, T_, F_)):
            wf.append(w_i); tf.append(t_i); ff.append(f_i)
            if i + 1 < len(W_):
                for wm in range(w_i + 1, W_[i + 1]):
                    wf.append(wm); tf.append(t_i + T_[i + 1]); ff.append(f_i + F_[i + 1])
        filled = FailureSpectrum(weights=wf, trials=tf, failures=ff,
                                 n_expanded=sd["n_expanded"], q_base=sd["q_base"],
                                 p_ref=sd["p_ref"])
        rw = reweight_spectrum(filled, isr["p_values"])
        print(f"\nladder vs IS reweighting (same decoder; stride-filled; lightest "
              f"measured-failing weight w={isr['w_meas_min']}):")
        print(f"  {'p':>9}  {'ladder S':>11}  {'IS reweight':>11}  {'S/IS':>7}  {'z':>5}")
        for p, P, Pse in zip(isr["p_values"], rw.P_logical, rw.P_logical_se):
            if not (sp.min() <= p <= sp.max()) or P <= 0:
                continue
            x = np.log(p)
            S = float(np.exp(np.interp(x, lp, lP)))
            Srel = float(np.interp(x, lp, rel))
            z = abs(np.log(S) - np.log(P)) / np.hypot(Srel, Pse / P)
            is_rows.append(dict(p=p, ladder_S=S, is_ler=P, is_se=Pse,
                                ratio=S / P, z=z))
            print(f"  {p:9.2e}  {S:11.3e}  {P:11.3e}  {S / P:7.2f}  {z:5.2f}")
        if is_rows:
            worst_is = max(r["z"] for r in is_rows)
            print(f"  worst z (ladder vs IS) = {worst_is:.2f}")
        # IS vs MC — with the ladder refuted below ~1.5e-3, THIS is the arbitration
        # that certifies the low-p estimator (the checkpoint p's are on the IS grid)
        pv = list(isr["p_values"])
        for f in sorted(OUT.glob("mc_*.json")):
            mc = json.loads(f.read_text(encoding="utf-8"))
            if not mc["fails"] or mc["p"] not in pv:
                continue
            i = pv.index(mc["p"])
            P, Pse = float(rw.P_logical[i]), float(rw.P_logical_se[i])
            z = abs(np.log(P) - np.log(mc["ler"])) / np.hypot(Pse / P, mc["se_rel"])
            print(f"  IS vs MC  p={mc['p']:.1e}:  IS {P:.3e}  MC {mc['ler']:.3e}  "
                  f"ratio {P / mc['ler']:.2f}  z={z:.2f}  "
                  f"{'PASS' if z < 3 else 'FAIL'}")
    out = dict(decoder=lad["decoder"], checkpoints=rows, is_comparison=is_rows,
               worst_z_mc=worst,
               worst_z_is=(max(r["z"] for r in is_rows) if is_rows else None),
               verdict=("PASS" if rows and worst < 3 else
                        "FAIL" if rows else "NO CHECKPOINTS"))
    (OUT / "report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[report] MC arbitration: {out['verdict']} (worst z = {worst:.2f})  "
          f"wrote report.json", flush=True)


STAGES = dict(probe=stage_probe, ladder=stage_ladder, mc=stage_mc, is_=stage_is,
              report=stage_report)
STAGES["is"] = stage_is

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: {sys.argv[0]} {{{'|'.join(STAGES)}}}")
    STAGES[sys.argv[1]]()
