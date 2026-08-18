"""Stochastic hunt for ghw_deep's sub-onset CZ-only failures: harvest, strip, escape.

Motivation (2026-08-05): the deep campaign measured ONE w=4 failure per 3.01M shots on
the CZ-only channel — sub-onset there, since CZ's circuit distance is 10 (onset 5) —
and the specimen's mechanism config was not recorded. The Technique-II subset
construction (tech2_subset_seeds --model "CZ only") reaches only configs that are
subsets of weight-D logicals; ghw_deep passes all of those, so the residual failure
mode lives OFF that manifold. This driver searches the space the construction cannot:

  H  harvest  ghw_deep-failing configs at weights where f(w) is measurable (w~8-16)
  S  strip    greedy failure-preserving descent to locally minimal failing sets
              (splitting_crosscheck.strip_configs — the add/remove asymmetry)
  E  escape   perturb the lightest survivors (add 1-2 random mechanisms, re-strip):
              plateau escapes, repeated until 3 consecutive rounds produce nothing
              lighter than the achieved floor

Everything at w <= 6 is decoded once more by all three decoder generations for
context, footprint-mapped into the FULL-SYMMETRIC DEM (the library's index space,
same convention as add_device_specimens_to_library, syndrome-asserted), and appended
to runs/decoder_loop/library.json with provenance.

    python experiments/methods/cz_subonset_hunt.py
    python experiments/methods/cz_subonset_hunt.py --targets 8 10 12 16 --rounds 12
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

for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS", "EMC_CALIB"):
    os.environ.pop(_k, None)
import run_error_model_comparison as rmc
from repo_paths import REPO_ROOT
from importance_sampling import _parse_dem, _expand
from splitting_crosscheck import strip_configs

LIB = REPO_ROOT / "runs" / "decoder_loop"
D_CZ = 10                       # tech2_72__CZ_only: circuit distance (BP-OSD bound)


def decode_cfgs(dec, det, obs, c2m, cfgs):
    """Failure mask for expanded-column configs."""
    syn = np.zeros((len(cfgs), det.shape[1]), dtype=bool)
    tru = np.zeros((len(cfgs), obs.shape[1]), dtype=bool)
    for i, cfg in enumerate(cfgs):
        syn[i] = np.bitwise_xor.reduce(det[c2m[list(cfg)]], axis=0)
        tru[i] = np.bitwise_xor.reduce(obs[c2m[list(cfg)]], axis=0)
    return np.any(dec.decode_batch(syn) != tru, axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", type=int, nargs="+", default=[8, 10, 12, 16])
    ap.add_argument("--per-weight", type=int, default=12)
    ap.add_argument("--shots-cap", type=int, default=120_000)
    ap.add_argument("--rounds", type=int, default=12, help="max escape rounds")
    ap.add_argument("--stall", type=int, default=3, help="stop after this many rounds with no lighter find")
    ap.add_argument("--decoder", default="ghw_deep")
    ap.add_argument("--no-add", action="store_true")
    a = ap.parse_args(argv)

    circ = rmc.make_circuit72("CZ only", rmc.P_REF)
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
    dec = rmc.DEC(calib, dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS[a.decoder]))
    dec.setup(circ)
    probs, det, obs = _parse_dem(circ)
    c2m, _, _ = _expand(probs, None)
    N_exp = c2m.shape[0]
    rng = np.random.default_rng(2026)
    print(f"[ctx] CZ-only: {det.shape[0]} mechs ({N_exp} expanded), decoder={a.decoder} "
          f"(device calib)", flush=True)

    # --- H: harvest ---
    configs = []
    for w in a.targets:
        got, shots, t0 = 0, 0, time.time()
        while got < a.per_weight and shots < a.shots_cap:
            idx = rng.integers(0, N_exp, size=(2000, w))
            s_ = np.sort(idx, axis=1)
            bad_rows = (s_[:, 1:] == s_[:, :-1]).any(axis=1)
            idx = idx[~bad_rows]
            syn = np.bitwise_xor.reduce(det[c2m[idx]], axis=1)
            tru = np.bitwise_xor.reduce(obs[c2m[idx]], axis=1)
            bad = np.any(dec.decode_batch(syn) != tru, axis=1)
            for row in idx[bad][: a.per_weight - got]:
                configs.append(frozenset(int(c) for c in row))
                got += 1
            shots += len(idx)
        print(f"[harvest] w={w}: {got}/{a.per_weight} in {shots:,} shots "
              f"({time.time()-t0:.0f}s)", flush=True)
    if not configs:
        raise SystemExit("harvest empty — raise --shots-cap or lower --targets")

    # --- S: strip ---
    stripped = strip_configs(det, obs, c2m, dec, sorted(configs, key=len), rng,
                             decode_cap=40_000)
    pool = {frozenset(c) for c in stripped}
    floor = min(len(c) for c in pool)

    # --- E: escape rounds ---
    stall = 0
    for rnd in range(a.rounds):
        lightest = sorted(pool, key=len)[:8]
        perturbed = []
        for cfg in lightest:
            for _ in range(6):
                k = int(rng.integers(1, 3))          # add 1 or 2 mechanisms
                add = set()
                while len(add) < k:
                    c = int(rng.integers(0, N_exp))
                    if c not in cfg:
                        add.add(c)
                perturbed.append(frozenset(cfg | add))
        bad = decode_cfgs(dec, det, obs, c2m, perturbed)
        still = [p for p, b in zip(perturbed, bad) if b]
        if still:
            re = strip_configs(det, obs, c2m, dec, still, rng, decode_cap=25_000)
            new = {frozenset(c) for c in re} - pool
            pool |= new
            new_floor = min(len(c) for c in pool)
            print(f"[escape {rnd}] {len(still)}/{len(perturbed)} kept failing -> "
                  f"{len(new)} new minimal cores; floor {floor} -> {new_floor}", flush=True)
            stall = stall + 1 if new_floor >= floor else 0
            floor = new_floor
        else:
            print(f"[escape {rnd}] no perturbation kept failing", flush=True)
            stall += 1
        if stall >= a.stall:
            print(f"[escape] floor stable at {floor} for {a.stall} rounds — stop", flush=True)
            break

    census = {}
    for cfg in pool:
        census[len(cfg)] = census.get(len(cfg), 0) + 1
    print("[census] " + "  ".join(f"w={w}:{n}" for w, n in sorted(census.items())), flush=True)

    # --- low-weight survivors: cross-decoder context + library ---
    low = sorted((c for c in pool if len(c) <= 6), key=len)
    if not low:
        print("[result] no cores at w<=6 — the floor did not descend into the "
              "sub-onset band this run")
    ctx_rows = {}
    if low:
        for name in ("baseline", "ghw", "ghw_deep"):
            d2 = rmc.DEC(calib, dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS[name]))
            d2.setup(circ)
            bad = decode_cfgs(d2, det, obs, c2m, low)
            ctx_rows[name] = int(bad.sum())
            print(f"[context] {name:10s}: fails {int(bad.sum())}/{len(low)} of the "
                  f"w<=6 cores", flush=True)

    added = unmapped = 0
    if low and not a.no_add:
        circ_f = rmc.make_circuit72("full symmetric", rmc.P_REF)
        _, det_f, obs_f = _parse_dem(circ_f)
        foot = {(det_f[i].tobytes(), obs_f[i].tobytes()): i for i in range(det_f.shape[0])}
        lib_path = LIB / "library.json"
        lib = json.loads(lib_path.read_text(encoding="utf-8")) if lib_path.exists() \
            else {"entries": []}
        have = {frozenset(e["mechs"]) for e in lib["entries"]}
        for cfg in low:
            mechs = sorted({int(c2m[c]) for c in cfg})
            if len(mechs) != len(cfg):
                unmapped += 1
                continue
            mapped = [foot.get((det[m].tobytes(), obs[m].tobytes())) for m in mechs]
            if any(m is None for m in mapped):
                unmapped += 1
                continue
            fs = frozenset(int(m) for m in mapped)
            if len(fs) != len(mechs) or fs in have:
                continue
            syn_m = np.bitwise_xor.reduce(det[mechs], axis=0)
            syn_f = np.bitwise_xor.reduce(det_f[sorted(fs)], axis=0)
            assert (syn_m == syn_f).all()
            lib["entries"].append(dict(mechs=sorted(fs), w=len(fs),
                                       generator="cz_hunt_strip_escape", iteration=0,
                                       model="CZ only", family="symmetric", code="bb72"))
            have.add(fs)
            added += 1
        ws = [e["w"] for e in lib["entries"]]
        lib["n"] = len(ws)
        lib["w_min"] = min(ws)
        lib_path.write_text(json.dumps(lib), encoding="utf-8")
        print(f"[lib] +{added} entries ({unmapped} unmappable) -> n={lib['n']}, "
              f"w_min={lib['w_min']}", flush=True)

    out = LIB / "cz_subonset_hunt.json"
    out.write_text(json.dumps(dict(
        decoder=a.decoder, targets=a.targets, floor=floor,
        census={str(w): n for w, n in sorted(census.items())},
        n_low=len(low), context=ctx_rows, added=added, unmapped=unmapped),
        indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
