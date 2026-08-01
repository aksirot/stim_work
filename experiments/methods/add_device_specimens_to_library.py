"""Fold rodan device-top-up failure specimens into the decoder-loop library.

The device-dir top-up (onset_topup_72 with EMC_CALIB=device caches) records failing
mechanism configurations per sub-model. Under the device convention every sub-model
mechanism exists in the full-symmetric DEM with identical (detector, observable)
footprint, so specimens map exactly (add_w3_harvest_to_library pattern) and are
full-model failures of the SAME device decoder — no re-verification required, though
we verify the syndrome identity and report current decoders' scores on the additions.

Selection: all specimens at w <= 4, plus up to CAP5 per model at w = 5 (heavier
specimens add bench cost without risk value — the library's mid/high band is rich).
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from decoder_loop import Ctx, GHW_CFG, OUT
import run_error_model_comparison as rmc
from importance_sampling import _parse_dem

DEV = pathlib.Path("runs/error_model_comparison_18_4_4_device_baseline18_ghw72")
MODELS = {"CZ_only": "CZ only", "meas_only": "meas only", "prep_only": "prep only",
          "gate_idle": "gate idle", "meas_idle": "meas idle"}
CAP5 = 10
NB_PRE_X2 = dict(GHW_CFG); NB_PRE_X2["pre_iter"] = 640


def main():
    ctx = Ctx()
    _, det_f, obs_f = _parse_dem(ctx.circ)
    foot = {(det_f[i].tobytes(), obs_f[i].tobytes()): i for i in range(det_f.shape[0])}

    lib_path = OUT / "library.json"
    lib = json.loads(lib_path.read_text(encoding="utf-8"))
    have = {frozenset(e["mechs"]) for e in lib["entries"]}

    added, unmapped, new_supports = 0, 0, []
    for slug_name, model in MODELS.items():
        f = DEV / f"tech1_72__{slug_name}.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))["result"]["spectrum"]
        fc = s.get("failure_configs", {})
        if not fc:
            continue
        circ_m = rmc.make_circuit72(model, rmc.P_REF)
        _, det_m, obs_m = _parse_dem(circ_m)
        tag = f"device_topup_{slug_name}"
        n_model = 0
        for w_str, cfgs in sorted(fc.items(), key=lambda kv: int(kv[0])):
            w = int(w_str)
            if w > 5:
                continue
            take = cfgs if w <= 4 else cfgs[:CAP5]
            for cfg in take:
                mapped = [foot.get((det_m[m].tobytes(), obs_m[m].tobytes())) for m in cfg]
                if any(m is None for m in mapped):
                    unmapped += 1
                    continue
                sup = frozenset(mapped)
                if len(sup) != len(cfg) or sup in have:
                    continue
                syn_m = np.bitwise_xor.reduce(det_m[list(cfg)], axis=0)
                syn_f = np.bitwise_xor.reduce(det_f[sorted(sup)], axis=0)
                assert (syn_m == syn_f).all()
                lib["entries"].append(dict(mechs=sorted(int(m) for m in sup), w=w,
                                           generator=tag, iteration=0))
                have.add(sup)
                new_supports.append(sorted(sup))
                added += 1; n_model += 1
        print(f"{slug_name:12s}: +{n_model}")
    lib["n"] = len(lib["entries"])
    lib["w_min"] = min(e["w"] for e in lib["entries"])
    lib_path.write_text(json.dumps(lib), encoding="utf-8")
    print(f"added {added} specimens ({unmapped} unmappable); "
          f"library n={lib['n']}, w_min={lib['w_min']}")

    for name, cfg in [("ghw", GHW_CFG), ("nb_pre_x2", NB_PRE_X2)]:
        x = ctx.fails(ctx.decoder(cfg), new_supports)
        print(f"  {name:10s} on the {added} additions: fixes {int((~x).sum())}, "
              f"fails {int(x.sum())}")


if __name__ == "__main__":
    main()
