"""Is ghw_deep the right decoder for the x5 ASYM family? (gates the deep report's §8)

ghw_deep's hyperparameters were selected entirely on SYMMETRIC-family evidence: the
decoder-loop library is full-symmetric plus symmetric channel slices, and not one asym
failing configuration entered the search. The asym ray is a DIFFERENT device (meas and
meas-idle priors at 2.5e-3 rather than 5e-4), and the one previous time a decoder was
carried across model families -- ghw onto the [[18,4,4]] code -- it broke badly.

The warning sign: asym__full_72 took 30,577 s under ghw_deep versus 6,105 s under ghw,
a 5x slowdown, which means its 1200-set ensemble is engaging on nearly every syndrome.
That is equally consistent with "working hard and winning" and with "flailing on a
mismatch"; only a paired test separates them.

This decodes the SAME sampled fault configurations with both decoders at several
weights (McNemar), under the asym device calibration both would use in production, and
reports throughput alongside -- so the verdict carries its cost.

    python experiments/methods/asym_decoder_check.py
    python experiments/methods/asym_decoder_check.py --weights 12,20,30 --shots 800
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binomtest

for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS", "EMC_CALIB"):
    os.environ.pop(_k, None)
import run_error_model_comparison as rmc
from repo_paths import REPO_ROOT
from importance_sampling import _parse_dem, _expand


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="12,20,30")
    ap.add_argument("--shots", type=int, default=800)
    a = ap.parse_args(argv)
    weights = [int(w) for w in a.weights.split(",")]

    # the asym device: sampling at P_REF, priors from the FULL ASYM circuit at DECODER_P
    circ = rmc.make_full_asym(rmc.BB_72_4_8, rmc.ROUNDS72, rmc.P_REF)
    calib = rmc.make_full_asym(rmc.BB_72_4_8, rmc.ROUNDS72, rmc.DECODER_P)
    cfgs = {"ghw": dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"]),
            "ghw_deep": dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_deep"])}
    decs = {k: rmc.DEC(calib, c) for k, c in cfgs.items()}
    for d in decs.values():
        d.setup(circ)

    probs, det, obs = _parse_dem(circ)
    c2m, q_base, _ = _expand(probs, None)
    N = c2m.shape[0]
    print(f"[ctx] asym full x5, {det.shape[0]} mechs, N_exp={N}; "
          f"decoders {list(cfgs)} under asym device calibration", flush=True)

    rng = np.random.default_rng(4242)
    rows, rates = [], {k: [0.0, 0] for k in cfgs}
    for w in weights:
        idx = rng.integers(0, N, size=(a.shots, w))
        while True:                                   # distinct columns per shot
            s = np.sort(idx, axis=1)
            bad = (s[:, 1:] == s[:, :-1]).any(axis=1)
            if not bad.any():
                break
            idx[bad] = rng.integers(0, N, size=(int(bad.sum()), w))
        syn = np.bitwise_xor.reduce(det[c2m[idx]], axis=1)
        tru = np.bitwise_xor.reduce(obs[c2m[idx]], axis=1)

        fails = {}
        for k, d in decs.items():
            t0 = time.time()
            fails[k] = np.any(d.decode_batch(syn) != tru, axis=1)
            dt = time.time() - t0
            rates[k][0] += dt; rates[k][1] += a.shots
        A, B = fails["ghw"], fails["ghw_deep"]
        b01 = int((A & ~B).sum())      # ghw fails, ghw_deep fixes  -> deep better
        b10 = int((~A & B).sum())      # ghw fixes, ghw_deep fails  -> deep worse
        p = binomtest(b10, b10 + b01, 0.5, alternative="greater").pvalue if (b01 + b10) else 1.0
        rows.append(dict(w=w, shots=a.shots, ghw_fails=int(A.sum()),
                         deep_fails=int(B.sum()), b01=b01, b10=b10,
                         p_deep_worse=round(float(p), 4)))
        print(f"  w={w:3d}  ghw {int(A.sum()):4d}/{a.shots}   ghw_deep {int(B.sum()):4d}/{a.shots}"
              f"   b01={b01:3d} (deep fixes)  b10={b10:3d} (deep breaks)  "
              f"p(deep worse)={p:.3f}", flush=True)

    tot01 = sum(r["b01"] for r in rows); tot10 = sum(r["b10"] for r in rows)
    p_all = binomtest(tot10, tot10 + tot01, 0.5, alternative="greater").pvalue \
        if (tot01 + tot10) else 1.0
    print(f"\npooled: b01={tot01} (ghw_deep fixes what ghw misses), "
          f"b10={tot10} (ghw_deep breaks what ghw gets), p(deep worse)={p_all:.4f}")
    for k, (dt, n) in rates.items():
        print(f"  {k:9s} {n/dt:8.1f} decodes/s")
    slow = (rates['ghw'][0] and rates['ghw_deep'][0]
            and (rates['ghw_deep'][0] / rates['ghw'][0]))
    print(f"  ghw_deep costs {slow:.1f}x ghw on this circuit")

    out = REPO_ROOT / "runs" / "decoder_loop" / "asym_decoder_check.json"
    out.write_text(json.dumps(dict(circuit="asym full x5 (72)", rows=rows,
                                   pooled=dict(b01=tot01, b10=tot10, p_deep_worse=p_all),
                                   rates={k: n / dt for k, (dt, n) in rates.items()},
                                   cost_ratio=slow), indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
