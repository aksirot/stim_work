#!/usr/bin/env python
"""Exact f0* for the 18-code ASYMMETRIC (x5-ray) mixes -> tech2_asym__*_18.json.

D, w0 and the failing SETS are ray-independent (support-level combinatorics), so the symmetric
tech2 tasks' D and L(D) are reused verbatim. The onset FRACTION is not: mechanisms enter the
expanded-uniform measure with multiplicity mult_j = round(p_j/q_base) (min_weight.py:180), so
scaling meas/meas_idle x5 re-weights every failing configuration. This recomputes only that
cheap fail-count re-weighting on the x5 circuits (seconds at 18-code) and caches the result in
the report's load() format. Rerun-safe: existing outputs are overwritten deterministically.
"""
from __future__ import annotations

import json
import time

import run_error_model_comparison as R
from min_weight import optimal_onset_fraction


def sym_tech2(mix):
    t = "tech2__full_symmetric" if mix == "full" else f"tech2_abl__{R.slug(mix)}"
    return json.loads((R.RESULTS / f"{t}.json").read_text(encoding="utf-8"))["result"]


def main():
    for mix in ["full"] + list(R.ABLATED):
        t2 = sym_tech2(mix)
        D = int(t2["D"])
        LD = {frozenset(s) for s in t2["LD"]}
        circ = (R.make_full_asym(R.P, R.ROUNDS, R.P_REF) if mix == "full"
                else R.make_abl_asym(R.P, R.ROUNDS, R.ABLATED[mix], R.P_REF))
        t0 = time.time()
        o = optimal_onset_fraction(circ, distance=D, logicals=LD)
        out = dict(config=dict(source="compute_f0_asym_18.py", reuses=("D", "LD"),
                               note="x5-ray multiplicities re-weight the symmetric failing sets"),
                   elapsed_s=time.time() - t0, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   result=dict(D=D, w0=int(o.onset), f0=float(o.onset_fraction),
                               f0_symmetric=float(t2["f0"]), n_LD=int(o.n_min_logicals),
                               n_expanded=int(o.n_expanded),
                               route=("Prop.1 (even D)" if D % 2 == 0 else "App.A.6 (odd D)")
                                     + ", x5-ray multiplicities"))
        name = f"tech2_asym__{R.slug(mix)}_18"
        (R.RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[{name}] D={D} w0={o.onset} f0_x5={o.onset_fraction:.4f} "
              f"(sym {t2['f0']:.4f}) t={out['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
