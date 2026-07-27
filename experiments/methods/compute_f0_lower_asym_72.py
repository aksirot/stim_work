#!/usr/bin/env python
"""Rigorous LOWER BOUND on f0* for the [[72,4,8]] ASYMMETRIC (x5-ray) mixes.

Companion to compute_f0_lower_72.py (isolated/full models on the symmetric ray) and
compute_f0_asym_18.py (exact 18-code x5 f0*). The x5 ray re-weights mechanisms by their
rate-proportional multiplicities, so the 72-code asym mixes need their own bound, computed
on the x5 circuits themselves.

Distances: the full x5 mix shares its support with the symmetric full model, so D=8 is reused
from tech2_72__full_symmetric. The ablated x5 mixes have NO cached 72-code distance (the
campaign never ran tech2 on any 72-code ablation), so their D is computed here first — a
BP-OSD upper bound on D (=> the recorded w0 is itself bound-flavored, like tech2_72's D).

Writes tech2_asym__<mix>_72.json in the report's load() format. Resumable: a mix whose JSON
already has f0_lower is skipped; rerun at higher --max-trials only ever tightens the bound.
"""
from __future__ import annotations

import argparse
import json
import time

import run_error_model_comparison as R
from min_weight import compute_distance, optimal_onset_fraction

MIXES = ["full"] + list(R.ABLATED)


def make(mix):
    if mix == "full":
        return R.make_full_asym(R.BB_72_4_8, R.ROUNDS72, R.P_REF)
    return R.make_abl_asym(R.BB_72_4_8, R.ROUNDS72, R.ABLATED[mix], R.P_REF)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-trials", type=int, default=300)
    ap.add_argument("--max-restrictions", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    D_full = json.loads((R.RESULTS / "tech2_72__full_symmetric.json")
                        .read_text(encoding="utf-8"))["result"]["D"]
    for mix in MIXES:
        name = f"tech2_asym__{R.slug(mix)}_72"
        path = R.RESULTS / f"{name}.json"
        if path.exists():
            res = json.loads(path.read_text(encoding="utf-8"))["result"]
            if res.get("f0_lower") not in (None, ""):
                print(f"[{name}] f0_lower already present ({res['f0_lower']:.3e}); skip", flush=True)
                continue
        circ = make(mix)
        t0 = time.time()
        try:
            if mix == "full":
                D, d_note = int(D_full), "reused from tech2_72__full_symmetric (same support)"
            else:
                D = int(compute_distance(circ).distance)
                d_note = "BP-OSD upper bound computed here (no cached 72-code ablation distance)"
            o = optimal_onset_fraction(circ, distance=D, max_trials=args.max_trials,
                                       max_restrictions=args.max_restrictions, seed=args.seed)
            result = dict(D=D, w0=int(o.onset), f0_lower=float(o.onset_fraction),
                          n_LD_lower=int(o.n_min_logicals), n_expanded=int(o.n_expanded),
                          D_note=d_note,
                          f0_lower_note=(f"lower bound f0* >= fails/C(N,w0) from {o.n_min_logicals} "
                                         f"partial weight-{D} logicals (max_trials={args.max_trials}), "
                                         f"x5-ray multiplicities"))
            print(f"[{name}] D={D} w0={o.onset} f0_lower={o.onset_fraction:.3e} "
                  f"n_LD={o.n_min_logicals} t={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            result = dict(f0_lower=None, f0_lower_note=f"unavailable: {type(e).__name__}: {e}")
            print(f"[{name}] FAILED: {type(e).__name__}: {e} (t={time.time()-t0:.0f}s)", flush=True)
        out = dict(config=dict(source="compute_f0_lower_asym_72.py", seed=args.seed,
                               max_trials=args.max_trials, max_restrictions=args.max_restrictions),
                   elapsed_s=time.time() - t0, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   result=result)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
