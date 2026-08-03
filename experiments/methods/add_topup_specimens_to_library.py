"""Fold onset-top-up failure specimens into the decoder-loop library.

onset_topup_72.py records up to 200 failing configs per weight (w=2..10) into each
spectrum's `failure_configs` block. Those are GHW-decoder failures in the sub-onset
band — material the loop's own descent cannot generate (its nc1 fast generator fixes
every w=3 config, so it never harvests one). This is the `add_w3_harvest_to_library.py`
pattern applied to the top-up feed.

Mechanism indices live in each spectrum's OWN DEM. Under the device-calibration
convention every sub-model mechanism exists in the full-symmetric DEM with the identical
(detector, observable) footprint, so each config is mapped by footprint matching and
verified to reproduce the same syndrome before it joins the library.

    python experiments/methods/add_topup_specimens_to_library.py --dry-run
    python experiments/methods/add_topup_specimens_to_library.py
    python experiments/methods/add_topup_specimens_to_library.py --dir runs/some_other_cache
    python experiments/methods/add_topup_specimens_to_library.py --max-weight 4   # sub-onset only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from decoder_loop import Ctx, OUT
import onset_topup_72 as T
from importance_sampling import _parse_dem

DEFAULT_DIR = "runs/error_model_comparison_18_4_4_sys_baseline18_ghw72"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=DEFAULT_DIR, help="results dir holding the topped-up spectra")
    ap.add_argument("--max-weight", type=int, default=10, help="ignore specimens above this weight")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write library.json")
    a = ap.parse_args(argv)

    src = pathlib.Path(a.dir)
    if not src.is_dir():
        raise SystemExit(f"no such results dir: {src}")

    ctx = Ctx()
    _, det_f, obs_f = _parse_dem(ctx.circ)          # full-symmetric DEM at P_REF
    foot = {(det_f[i].tobytes(), obs_f[i].tobytes()): i for i in range(det_f.shape[0])}

    lib_path = OUT / "library.json"
    lib = json.loads(lib_path.read_text(encoding="utf-8"))
    have = {frozenset(e["mechs"]) for e in lib["entries"]}
    n_before, w_before = len(lib["entries"]), lib.get("w_min")

    added_by_w: dict[int, int] = {}
    total_added = dup = unmapped = 0
    scanned = 0

    for name in T.SPECTRA:
        p = src / f"{name}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        spec = j["result"]["spectrum"]
        fc = spec.get("failure_configs") or {}
        n_spec = sum(len(v) for v in fc.values())
        scanned += 1
        if not n_spec:
            print(f"[skip] {name}: no specimens recorded")
            continue

        # the spectrum's own sampling circuit, under ITS cached calibration convention
        device = j["config"]["decoder"].get("calib") == "device"
        circ_m, _ = T.build(name, device=device)
        _, det_m, obs_m = _parse_dem(circ_m)

        added_here = 0
        for w_str, cfgs in sorted(fc.items(), key=lambda kv: int(kv[0])):
            w = int(w_str)
            if w > a.max_weight:
                continue
            for mechs in cfgs:
                mapped = [foot.get((det_m[m].tobytes(), obs_m[m].tobytes())) for m in mechs]
                if any(m is None for m in mapped):
                    unmapped += 1
                    continue
                sup = frozenset(mapped)
                if len(sup) != len(mechs):       # collapsed under the mapping
                    unmapped += 1
                    continue
                if sup in have:
                    dup += 1
                    continue
                # verify the mapped config reproduces the same syndrome
                syn_m = np.bitwise_xor.reduce(det_m[list(mechs)], axis=0)
                syn_f = np.bitwise_xor.reduce(det_f[sorted(sup)], axis=0)
                assert (syn_m == syn_f).all(), f"{name} w={w}: syndrome mismatch after mapping"
                # provenance is load-bearing: an entry is only meaningful against
                # the decoder of ITS device family (symmetric models share the
                # full-symmetric device decoder; the x5 asym ray is a different
                # device with its own priors).
                fam = "asym" if name.startswith("asym__") else "symmetric"
                lib["entries"].append(dict(mechs=sorted(int(m) for m in sup), w=w,
                                           generator=f"topup_{name}", iteration=0,
                                           model=name, family=fam, code="bb72"))
                have.add(sup)
                added_by_w[w] = added_by_w.get(w, 0) + 1
                added_here += 1
                total_added += 1
        print(f"[fold] {name}: {n_spec} specimens -> {added_here} new entries")

    if not scanned:
        raise SystemExit(f"no known spectra found under {src} (checked {len(T.SPECTRA)} names)")

    lib["n"] = len(lib["entries"])
    lib["w_min"] = min(e["w"] for e in lib["entries"])
    print(f"\nlibrary {n_before} -> {lib['n']} entries (+{total_added}); "
          f"w_min {w_before} -> {lib['w_min']}")
    print(f"  added by weight: {dict(sorted(added_by_w.items()))}")
    print(f"  {dup} already present, {unmapped} unmappable")

    if a.dry_run:
        print("\n[dry-run] library.json NOT written")
        return
    if not total_added:
        print("\nnothing to add; library.json left untouched")
        return
    bak = OUT / "library_pre_topup.json.bak"
    if not bak.exists():
        bak.write_text(json.dumps(json.loads(lib_path.read_text(encoding="utf-8"))),
                       encoding="utf-8")
        print(f"  backed up prior library to {bak.name}")
    lib_path.write_text(json.dumps(lib), encoding="utf-8")
    print(f"  wrote {lib_path}")


if __name__ == "__main__":
    main()
