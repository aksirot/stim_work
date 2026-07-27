#!/usr/bin/env python
"""Rigorous LOWER BOUND on the [[72,4,8]] perfect-decoder onset fraction f0*.

Exact f0* is infeasible at this size (the L(D) enumeration is the bb144-regime problem), so the
campaign's ``tech2_72`` tasks store only the distance ``D``. But ``optimal_onset_fraction`` with a
PARTIAL weight-D logical set returns ``fails / C(N, w0)``, which is a rigorous LOWER bound on
|F(w0)| / f0* (see min_weight.py:1420). This tool enumerates a partial L(D) via BP-OSD sampling
and records the resulting floor into each ``tech2_72__<model>.json`` result block:

    f0_lower     the lower bound on f0*
    w0           onset weight = ceil(D/2)
    n_LD_lower   how many weight-D logicals the partial search found (more -> tighter)
    f0_lower_note provenance

Resumable: a model whose JSON already has ``f0_lower`` is skipped. The bound only ever RISES with
more logicals, so a cluster rerun at higher ``--max-trials`` can only tighten it. Pair with the
measured onset dot f(w0) (a perfect decoder is <= the actual one) for an upper bound: f0* in
[f0_lower, f(w0)].

    python experiments/methods/compute_f0_lower_72.py                 # all 6 models
    python experiments/methods/compute_f0_lower_72.py --max-trials 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_error_model_comparison as R
from min_weight import optimal_onset_fraction

MAX_TRIALS = int(os.environ.get("F0LO_MAX_TRIALS", "300"))
MAX_RESTRICTIONS = int(os.environ.get("F0LO_MAX_RESTRICTIONS", "5000000"))
SEED = int(os.environ.get("F0LO_SEED", "1"))


def compute(max_trials: int, max_restrictions: int, seed: int) -> None:
    for name in R.MODELS:
        task = f"tech2_72__{R.slug(name)}"
        path = R.RESULTS / f"{task}.json"
        j = json.loads(path.read_text(encoding="utf-8"))
        res = j["result"]
        if res.get("f0_lower") not in (None, ""):
            print(f"[{task}] f0_lower already present ({res['f0_lower']:.3e}); skip", flush=True)
            continue
        D = int(res["D"])
        circ = R.make_circuit72(name, R.P_REF)
        t0 = time.time()
        try:
            o = optimal_onset_fraction(circ, distance=D, max_trials=max_trials,
                                       max_restrictions=max_restrictions, seed=seed)
            res["f0_lower"] = float(o.onset_fraction)
            res["w0"] = int(o.onset)
            res["n_LD_lower"] = int(o.n_min_logicals)
            res["n_expanded"] = int(o.n_expanded)
            res["f0_lower_note"] = (f"lower bound f0* >= fails/C(N,w0) from {o.n_min_logicals} "
                                    f"partial weight-{D} logicals (max_trials={max_trials})")
            print(f"[{task}] D={D} w0={o.onset} f0_lower={o.onset_fraction:.3e} "
                  f"n_LD={o.n_min_logicals} t={time.time()-t0:.0f}s", flush=True)
        except Exception as e:                       # enumeration too large / search failure
            res["f0_lower"] = None
            res["f0_lower_note"] = f"lower bound unavailable: {type(e).__name__}: {e}"
            print(f"[{task}] FAILED: {type(e).__name__}: {e} (t={time.time()-t0:.0f}s)", flush=True)
        path.write_text(json.dumps(j, indent=1), encoding="utf-8")
    print("done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-trials", type=int, default=MAX_TRIALS,
                    help="BP-OSD logical-search trials per model (more -> tighter bound)")
    ap.add_argument("--max-restrictions", type=int, default=MAX_RESTRICTIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    print(f"f0* lower bound (72-code): max_trials={args.max_trials}, "
          f"max_restrictions={args.max_restrictions}, seed={args.seed}", flush=True)
    compute(args.max_trials, args.max_restrictions, args.seed)


if __name__ == "__main__":
    main()
