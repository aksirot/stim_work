"""Fold the runs/subonset_tune_72 w=3 harvest into the decoder-loop library.

The 71 harvested w=3 sub-onset failures were collected per SUB-MODEL, so their
mechanism indices live in each sub-model's DEM. Under the device-calibration
convention every sub-model mechanism exists in the full-symmetric DEM with the
identical (detector, observable) footprint, so each config is mapped by footprint
matching and verified to reproduce the same syndrome. Entries join the library as
permanent regression tests (generator tag records provenance); this drops the
library floor from w_hat=4 to w=3 -- the sub-onset band the descent machinery
cannot reach on its own (nc1 fixes all w=3 singles, so it never generates them).
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from decoder_loop import Ctx, GHW_CFG, OUT
import run_error_model_comparison as rmc
from importance_sampling import _parse_dem

HEAVY_CFG = dict(rmc.DEC_CFG, gamma0=0.125, pre_iter=320, num_sets=200,
                 set_max_iter=120, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)


def main():
    ctx = Ctx()
    _, det_f, obs_f = _parse_dem(ctx.circ)
    foot = {(det_f[i].tobytes(), obs_f[i].tobytes()): i for i in range(det_f.shape[0])}

    j = json.loads(open('runs/subonset_tune_72/hunt_and_bench_w3.json', encoding='utf-8').read())
    lib_path = OUT / "library.json"
    lib = json.loads(lib_path.read_text(encoding="utf-8"))
    have = {frozenset(e["mechs"]) for e in lib["entries"]}

    added, unmapped = 0, 0
    new_supports = []
    for model, h in j["harvest"].items():
        circ_m = rmc.make_circuit72(model, rmc.P_REF)
        _, det_m, obs_m = _parse_dem(circ_m)
        tag = f"w3_harvest_{model.replace(' ', '_')}"
        for triple in h["mechs"]:
            mapped = [foot.get((det_m[m].tobytes(), obs_m[m].tobytes())) for m in triple]
            if any(m is None for m in mapped):
                unmapped += 1
                continue
            sup = frozenset(mapped)
            if len(sup) != len(triple) or sup in have:
                continue
            # verify identical syndrome under the full-sym matrices
            syn_m = np.bitwise_xor.reduce(det_m[list(triple)], axis=0)
            syn_f = np.bitwise_xor.reduce(det_f[list(sorted(sup))], axis=0)
            assert (syn_m == syn_f).all()
            lib["entries"].append(dict(mechs=sorted(int(m) for m in sup), w=3,
                                       generator=tag, iteration=0,
                                       model=model, family="symmetric", code="bb72"))
            have.add(sup)
            new_supports.append(sorted(sup))
            added += 1
    lib["n"] = len(lib["entries"])
    lib["w_min"] = min(e["w"] for e in lib["entries"])
    lib_path.write_text(json.dumps(lib), encoding="utf-8")
    print(f"added {added} w=3 entries ({unmapped} unmappable); library n={lib['n']}, "
          f"w_min={lib['w_min']}")

    # sanity: current headline decoders on the new entries (full-sym device decoder)
    for name, cfg in [("baseline", rmc.DEC_CFG), ("ghw", GHW_CFG), ("heavy", HEAVY_CFG)]:
        x = ctx.fails(ctx.decoder(cfg), new_supports)
        print(f"  {name:9s} on the {added} new w=3 entries: fixes {int((~x).sum())}, "
              f"fails {int(x.sum())}")


if __name__ == "__main__":
    main()
