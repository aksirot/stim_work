"""Standalone w=3 probe: meas-only faults vs the device-calibrated ghw_deep decoder.

No campaign dependency (the top-up script's spectrum requirement is merge
bookkeeping, not science): sampling circuit = meas-only at P_REF, decoder =
ghw_deep calibrated on FULL SYMMETRIC at DECODER_P (the device convention).
Identical sampling semantics to onset_topup (uniform expanded-column draws
without replacement at fixed weight); failing configs recorded as specimens.

Env: PROBE_SHOTS_MAX (3e6), PROBE_TARGET (20), PROBE_CHUNK (50000).
Output: runs/probe_w3_meas_ghwdeep.json (+ stdout log)
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
from importance_sampling import _parse_dem, _expand
from repo_paths import REPO_ROOT

W = 3
SHOTS_MAX = int(float(os.environ.get("PROBE_SHOTS_MAX", "3000000")))
TARGET = int(os.environ.get("PROBE_TARGET", "20"))
CHUNK = int(os.environ.get("PROBE_CHUNK", "50000"))
GHW_DEEP = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_deep"])


def main():
    circ = rmc.make_circuit72("meas only", rmc.P_REF)
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)   # DEVICE decoder
    dec = rmc.DEC(calib, GHW_DEEP)
    dec.setup(circ)                       # no-op post-calibration: priors stay device
    probs, det_mat, obs_mat = _parse_dem(circ)
    col_to_mech, q_base, _ = _expand(probs, None)
    N_exp = col_to_mech.shape[0]
    M, K = det_mat.shape[1], obs_mat.shape[1]
    rng = np.random.default_rng([4242, W])
    print(f"probe: meas-only w={W} vs ghw_deep(device calib)  N_exp={N_exp}  "
          f"cap={SHOTS_MAX} target={TARGET}", flush=True)

    shots = fails = 0
    specimens = []
    t0 = time.perf_counter()
    while fails < TARGET and shots < SHOTS_MAX:
        B = min(CHUNK, SHOTS_MAX - shots)
        syn = np.zeros((B, M), dtype=bool)
        tru = np.zeros((B, K), dtype=bool)
        rows = []
        for t in range(B):
            mech_idxs = col_to_mech[rng.choice(N_exp, size=W, replace=False)]
            rows.append(mech_idxs)
            syn[t] = np.bitwise_xor.reduce(det_mat[mech_idxs], axis=0)
            tru[t] = np.bitwise_xor.reduce(obs_mat[mech_idxs], axis=0)
        bad = np.any(dec.decode_batch(syn) != tru, axis=1)
        for i in np.nonzero(bad)[0]:
            specimens.append(sorted(int(m) for m in rows[i]))
        fails += int(bad.sum())
        shots += B
        el = time.perf_counter() - t0
        print(f"  {shots}/{SHOTS_MAX} shots, {fails} fails  "
              f"({shots/el:.0f} dec/s, {el:.0f}s)", flush=True)

    f_hat = fails / shots
    bound = "" if fails else f"  (f < {3/shots:.1e} rule-of-three)"
    print(f"RESULT: w={W} meas-only under ghw_deep: {fails}/{shots} = "
          f"{f_hat:.2e}{bound}  specimens={len(specimens)}", flush=True)
    out = REPO_ROOT / "runs" / "probe_w3_meas_ghwdeep.json"
    out.write_text(json.dumps(dict(
        model="meas only", w=W, decoder="ghw_deep", calib="device_full_symmetric",
        calibrated_at=rmc.DECODER_P, p_ref=rmc.P_REF, n_expanded=int(N_exp),
        q_base=float(q_base), shots=shots, fails=fails,
        specimens=specimens, elapsed_s=round(time.perf_counter() - t0, 1)),
        indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
