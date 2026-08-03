"""Technique-II seeded failure candidates: subsets of circuit logicals.

Stochastic descent finds failures from ABOVE and plateaus (72: reached the true onset;
144: stuck at w=8 after 100+ escape rounds). This builds candidates from CODE
STRUCTURE instead, which is where the onset actually comes from.

The construction: let L be a circuit logical of weight D (trivial syndrome, non-trivial
logical action). For any subset S of L, syndrome(S) == syndrome(L\\S), so the decoder
must choose between explaining the syndrome with S or with its complement — and
choosing the complement IS the logical error. Hence, per weight w = |S|:

  w <  D/2   complement is HEAVIER: a min-weight decoder wins. A failure here is an
             unambiguous DECODER DEFECT (sub-onset floor) — the interesting case.
  w == D/2   (even D) exact tie: even a perfect decoder fails a fraction — this is
             Technique II's f0*, not a defect.
  w >  D/2   complement is LIGHTER: a min-weight decoder FAILS by construction. For
             odd D this includes w0 = (D+1)/2, which is why the onset sits there.

Subsets are enumerated EXHAUSTIVELY when C(D, w) is small (C(8,4)=70, C(10,5)=252),
so for a given logical the sub-onset question is answered completely rather than
sampled. Failures are appended to the failure library with full provenance.

    python experiments/methods/tech2_subset_seeds.py --code 72
    python experiments/methods/tech2_subset_seeds.py --code 72 --search 200  # more logicals
    python experiments/methods/tech2_subset_seeds.py --code 144 --search 400
"""
from __future__ import annotations

import argparse
import itertools
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
from min_weight import find_min_weight_logicals

MAX_SUBSETS = 400          # per (logical, weight): enumerate all if C(D,w) <= this


def build_ctx(code):
    if code == "72":
        circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
        calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
        decs = {
            "baseline": dict(rmc.DEC_CFG),
            "ghw": dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"]),
            "ghw_deep": dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_deep"]),
        }
        make = lambda cfg: rmc.DEC(calib, cfg)
        lib_dir = REPO_ROOT / "runs" / "decoder_loop"
        onset_json = REPO_ROOT / "runs" / "splitting_crosscheck" / "72_full_ghw_onset.json"
        meta = dict(model="full symmetric", family="symmetric", code="bb72")
    else:
        from experiment_runner import load_config, build_circuit, make_decoder
        from bb_code_sim import RelayBPDecoder
        cfg_y = load_config(str(REPO_ROOT / "experiments" / "configs" / "gross_lpu_idle.yaml"))
        circ = build_circuit(cfg_y)
        PAPER = dict(rmc.DEC_CFG)
        PAPER.update(num_sets=600, pre_iter=80, set_max_iter=60, stop_nconv=5,
                     gamma0=0.79, gamma_dist_interval=(0.34, 1.24))
        decs = {"campaign": "campaign",
                "ghw": dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"]),
                "paper_T7": PAPER}
        def make(cfg):
            if cfg == "campaign":
                return make_decoder(cfg_y)
            return RelayBPDecoder(**{k: (tuple(v) if isinstance(v, list) else v)
                                     for k, v in cfg.items()})
        lib_dir = REPO_ROOT / "runs" / "decoder_loop144"
        onset_json = REPO_ROOT / "runs" / "splitting_crosscheck" / "lpu_idle_camp.json"
        meta = dict(model="gross idle (bare memory)", family="idle", code="bb144")
    return circ, decs, make, lib_dir, onset_json, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", choices=["72", "144"], default="72")
    ap.add_argument("--search", type=int, default=0,
                    help="BP-OSD trials for ADDITIONAL logicals (0 = cached only)")
    ap.add_argument("--distance", type=int, default=None, help="override D")
    ap.add_argument("--no-add", action="store_true", help="report only, do not touch library")
    a = ap.parse_args(argv)

    circ, dec_cfgs, make, lib_dir, onset_json, meta = build_ctx(a.code)
    probs, det, obs = _parse_dem(circ)
    c2m, q_base, _ = _expand(probs, None)
    m2c = {}
    for col, m in enumerate(c2m):
        m2c.setdefault(int(m), col)
    print(f"[ctx] code={a.code}  mechs={det.shape[0]}  detectors={det.shape[1]}", flush=True)

    # --- logicals: cached onset-hunt result, plus optional BP-OSD search ---
    logicals = []
    if onset_json.exists():
        j = json.loads(onset_json.read_text(encoding="utf-8"))
        for key in ("best_logical_mechs", "min_config_mechs"):
            v = j.get(key)
            if v:
                logicals.append(frozenset(int(m) for m in v))
    if a.search:
        D = a.distance or (len(min(logicals, key=len)) if logicals else None)
        t0 = time.time()
        print(f"[search] BP-OSD for weight-{D} logicals, {a.search} trials ...", flush=True)
        found = find_min_weight_logicals(circ, D, max_trials=a.search, seed=7)
        print(f"[search] +{len(found)} logicals in {time.time()-t0:.0f}s", flush=True)
        logicals += [frozenset(int(m) for m in s) for s in found]
    # keep genuinely zero-syndrome, non-trivial ones only
    keep = []
    for L in dict.fromkeys(logicals):
        cols = [m2c[m] for m in L if m in m2c]
        if len(cols) != len(L):
            continue
        syn = np.bitwise_xor.reduce(det[sorted(L)], axis=0)
        tru = np.bitwise_xor.reduce(obs[sorted(L)], axis=0)
        if not syn.any() and tru.any():
            keep.append(sorted(L))
    if not keep:
        print("[!] no verified circuit logicals available — run with --search")
        return 1
    Ds = sorted({len(L) for L in keep})
    D = a.distance or min(Ds)
    w0 = (D + 1) // 2
    print(f"[logicals] {len(keep)} verified (weights {Ds}); using D={D} -> w0={w0}", flush=True)

    # --- subsets per weight ---
    rng = np.random.default_rng(2026)
    weights = [w for w in range(max(2, w0 - 2), w0 + 1)]
    cands = {}                       # w -> list of mech-support tuples
    for L in keep:
        n = len(L)
        for w in weights:
            if w > n:
                continue
            combos = list(itertools.combinations(L, w))
            if len(combos) > MAX_SUBSETS:
                idx = rng.choice(len(combos), size=MAX_SUBSETS, replace=False)
                combos = [combos[i] for i in idx]
            cands.setdefault(w, set()).update(combos)
    for w in weights:
        print(f"[cands] w={w}: {len(cands.get(w, ()))} distinct subsets "
              f"({'complement heavier -> decoder SHOULD win' if 2*w < D else 'tie' if 2*w == D else 'complement lighter -> perfect decoder FAILS'})",
              flush=True)

    # --- decode ---
    results = {}
    new_fail_supports = []
    for name, cfg in dec_cfgs.items():
        dec = make(cfg)
        dec.setup(circ)
        row = {}
        for w in weights:
            supports = sorted(cands.get(w, ()))
            if not supports:
                continue
            syn = np.zeros((len(supports), det.shape[1]), dtype=bool)
            tru = np.zeros((len(supports), obs.shape[1]), dtype=bool)
            for i, S in enumerate(supports):
                syn[i] = np.bitwise_xor.reduce(det[list(S)], axis=0)
                tru[i] = np.bitwise_xor.reduce(obs[list(S)], axis=0)
            t0 = time.time()
            bad = np.any(dec.decode_batch(syn) != tru, axis=1)
            row[w] = (int(bad.sum()), len(supports))
            tag = "DEFECT" if 2 * w < D else ("tie" if 2 * w == D else "expected")
            print(f"  {name:10s} w={w}: {int(bad.sum()):4d}/{len(supports):4d} fail "
                  f"[{tag}]  ({len(supports)/max(time.time()-t0,1e-9):.1f} dec/s)", flush=True)
            if 2 * w <= D:
                new_fail_supports += [supports[i] for i in np.nonzero(bad)[0]]
        results[name] = row

    # --- library ---
    added = 0
    if not a.no_add and new_fail_supports:
        lib_path = lib_dir / "library.json"
        lib = json.loads(lib_path.read_text(encoding="utf-8")) if lib_path.exists() \
            else {"entries": []}
        have = {frozenset(e["mechs"]) for e in lib["entries"]}
        for S in new_fail_supports:
            fs = frozenset(int(m) for m in S)
            if fs not in have:
                # regime matters: below D/2 a failure is a decoder DEFECT; at exactly
                # D/2 the true fault and its complement tie, so even a perfect decoder
                # fails a fraction — tagged so benches never read a tie as a defect.
                gen = "tech2_subset_defect" if 2 * len(fs) < D else "tech2_subset_tie"
                lib["entries"].append(dict(mechs=sorted(fs), w=len(fs),
                                           generator=gen, iteration=0, **meta))
                have.add(fs); added += 1
        ws = [e["w"] for e in lib["entries"]]
        lib["n"] = len(ws); lib["w_min"] = min(ws)
        lib_path.write_text(json.dumps(lib), encoding="utf-8")
        print(f"[lib] +{added} entries -> n={lib['n']}, w_min={lib['w_min']}", flush=True)

    out = lib_dir / "tech2_subset_seeds.json"
    out.write_text(json.dumps(dict(code=a.code, D=D, w0=w0, n_logicals=len(keep),
                                   weights=weights, added=added,
                                   results={k: {str(w): v for w, v in r.items()}
                                            for k, r in results.items()}), indent=1),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
