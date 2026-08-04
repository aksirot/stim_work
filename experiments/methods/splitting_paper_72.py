"""Paper-faithful splitting (arXiv:2511.15177 Alg. 2/3 + §5.3) on the [[72,4,8]] code.

The arbitration the tempering variants failed twice — now against the paper's ACTUAL
method: Eq.18 fine ladder (steps 2^(-1/sqrt(w)) — chains drift down in small overlaps),
multi-seeded warm starts from TYPICAL failing configs at p0, and the adaptive
sigma+Delta chain-growth controller. Range restricted to the MC-checkable window
(8e-3 -> 2e-3); the acceptance target is the existing ghw_deep direct-MC point
LER(2e-3) = 11/150k = 7.33e-5 (runs/splitting_crosscheck/72_full_v2_ghwdeep.json).

Budgets are laptop-scaled (paper uses L=12, M=3, T_init=1e6): the controller may stop
on T_cap rather than eps — the diagnostics record per-level convergence status either
way, which is itself part of the answer.

Output: runs/splitting_crosscheck/72_full_paper.json + .log
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import os
for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS", "EMC_CALIB"):
    os.environ.pop(_k, None)
import run_error_model_comparison as rmc
from splitting import multi_seeded_split_estimate
from repo_paths import REPO_ROOT

OUT = REPO_ROOT / "runs" / "splitting_crosscheck"
GHW_DEEP = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_deep"])

MC_P, MC_LER, MC_FAILS, MC_SHOTS = 2e-3, 7.333e-05, 11, 150_000   # v2 run's anchor


def main():
    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
    dec = rmc.DEC(calib, GHW_DEEP)

    res, diag = multi_seeded_split_estimate(
        circ, dec, p_ref=rmc.P_REF, p_high=8e-3, p_low=MC_P,
        L=4, M=2, T_init=2_000, T_cap=8_000, eps=0.35,
        ladder="eq18", distance=8, anchor_shots=3_000, seed=99)

    sp, sP, sse = np.asarray(res.p_ladder), np.asarray(res.P_logical), np.asarray(res.P_logical_se)
    S = float(sP[-1])                              # ladder endpoint IS p_low = MC_P
    Sse_rel = float(sse[-1] / max(S, 1e-300))
    mc_rel = 1.0 / np.sqrt(MC_FAILS)
    z = abs(np.log(S) - np.log(MC_LER)) / np.hypot(Sse_rel, mc_rel)
    verdict = "PASS" if z < 3 else "FAIL"
    print(f"[verdict] paper-ladder({MC_P}) = {S:.3e} vs MC {MC_LER:.3e}  "
          f"z = {z:.1f} -> {verdict}", flush=True)

    out = dict(target="72_full_paper", decoder="ghw_deep",
               algorithm="multi_seeded_split_estimate (Alg2/3 + eq18 + sigma-delta)",
               p_ref=rmc.P_REF, calibrated_at=rmc.DECODER_P,
               sp=sp.tolist(), sP=sP.tolist(), sP_se=sse.tolist(),
               mc_overlap=dict(p=MC_P, fails=MC_FAILS, shots=MC_SHOTS, ler=MC_LER),
               overlap_z=float(z), verdict=verdict,
               diag={k: v for k, v in diag.items()
                     if isinstance(v, (int, float, str, list, dict))})
    path = OUT / "72_full_paper.json"
    OUT.mkdir(parents=True, exist_ok=True)   # fresh clone (e.g. a cluster) has no runs/
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
