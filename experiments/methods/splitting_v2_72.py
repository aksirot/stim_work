"""v2 splitting on [[72,4,8]] full symmetric with nb_pre_x2 — the validation run.

The rebuilt ladder (splitting_v2.replica_exchange_v2: library-diverse seeds,
coset-jump moves, independent ladders) must EARN trust by matching a direct-MC
overlap point in the measurable regime before its unmeasurable-regime numbers mean
anything. Decoder = the decoder-loop's VERIFIED recommendation ghw_deep / pre640_sets1200
(0/90 sub-onset certified, paired b01=3/b10=0 vs nb_pre_x2). Seeds = the
decoder-loop failure library entries that fail under it, plus the standard sources.
Acceptance = z(|ladder − MC|) at p=2e-3 and ladder spread. NOTE: the coset-jump
move currently contributes nothing (logical jumps exit the failing set — needs
low-weight DEM stabilizers, v2 to-do); the operative upgrades are seed diversity,
independent ladders, and the MC overlap gate.

Output: runs/splitting_crosscheck/72_full_v2_ghwdeep.json + .log
"""
import json
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import os
for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS"):
    os.environ.pop(_k, None)
import run_error_model_comparison as rmc
from splitting_v2 import replica_exchange_v2
from repo_paths import REPO_ROOT

OUT = REPO_ROOT / "runs" / "splitting_crosscheck"
GHW_DEEP = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_deep"])

MC_P = 2e-3
MC_SHOTS = 150_000


def main():
    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
    dec = rmc.DEC(calib, GHW_DEEP)

    # --- seeds: decoder-loop library entries that fail under nb_pre_x2 ---
    lib = json.loads((REPO_ROOT / "runs" / "decoder_loop" / "library.json")
                     .read_text(encoding="utf-8"))
    supports = [e["mechs"] for e in lib["entries"]]
    print(f"library: {len(supports)} entries (v2 seeder verifies which fail "
          f"under nb_pre_x2)", flush=True)

    # known logicals from the onset hunt (x^y verified)
    onset = json.loads((OUT / "72_full_ghw_onset.json").read_text(encoding="utf-8"))
    logicals = [onset["best_logical_mechs"]]

    # --- direct-MC overlap point (same decoder) — the acceptance target ---
    t0 = time.perf_counter()
    mc_circ = rmc.make_circuit72("full symmetric", MC_P)
    mc_dec = rmc.DEC(calib, GHW_DEEP)
    mc_dec.setup(mc_circ)
    sampler = mc_circ.compile_detector_sampler(seed=909)
    fails = shots = 0
    while shots < MC_SHOTS:
        dets, obs = sampler.sample(15_000, separate_observables=True)
        pred = mc_dec.decode_batch(dets)
        fails += int(np.any(pred != obs, axis=1).sum())
        shots += 15_000
        print(f"  [mc] {shots} shots, {fails} fails", flush=True)
    mc_ler = fails / shots
    mc_se = np.sqrt(max(fails, 1)) / shots
    print(f"[mc] overlap point: LER({MC_P}) = {fails}/{shots} = {mc_ler:.3e} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # --- the v2 ladder ---
    res, diag = replica_exchange_v2(
        circ, dec, p_ref=rmc.P_REF, p_high=0.008, p_low=1e-4, n_levels=16,
        n_ladders=3, n_walkers=8, local_steps=3, n_sweeps=50, burn_in=15,
        anchor_shots=4000, distance=8, seed=42,
        seed_supports=supports, extra_logicals=logicals,
        gap_weights=[12, 20, 32], jump_every=1,
        pool_cache=str(OUT / "72_full_v2_ghwdeep_pool.json"),
        cache_key=dict(decoder="ghw_deep", model="full symmetric",
                       lib_n=len(supports)))

    # --- acceptance: ladder vs MC at the overlap p ---
    sp, sP, sse = np.asarray(res.p_ladder), np.asarray(res.P_logical), np.asarray(res.P_logical_se)
    S = float(np.exp(np.interp(np.log(MC_P), np.log(sp[::-1]), np.log(sP[::-1]))))
    Sse_rel = float(np.interp(np.log(MC_P), np.log(sp[::-1]), (sse / np.maximum(sP, 1e-300))[::-1]))
    z = abs(np.log(S) - np.log(mc_ler)) / np.hypot(Sse_rel, mc_se / mc_ler)
    spread = diag["ladder_spread_log10"][-1]
    verdict = "PASS" if (z < 3 and spread < 0.5) else "FAIL"
    print(f"[verdict] ladder({MC_P}) = {S:.3e} vs MC {mc_ler:.3e}  z = {z:.1f}; "
          f"ladder spread at p_low = {spread:.2f} log10 -> {verdict}", flush=True)

    out = dict(target="72_full_v2_ghwdeep", decoder="ghw_deep", decoder_cfg={
                   k: (list(v) if isinstance(v, tuple) else v) for k, v in GHW_DEEP.items()},
               p_ref=rmc.P_REF, calibrated_at=rmc.DECODER_P,
               sp=sp.tolist(), sP=sP.tolist(), sP_se=sse.tolist(),
               mc_overlap=dict(p=MC_P, fails=fails, shots=shots, ler=mc_ler),
               overlap_z=float(z), verdict=verdict, **diag)
    path = OUT / "72_full_v2_ghwdeep.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
