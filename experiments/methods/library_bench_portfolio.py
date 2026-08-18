"""Bench baseline / ghw / ghw_deep / min-weight portfolio on the full failure library.

Motivation (2026-08-06 confirmed): the CZ w=4 specimen [2482,5202,5289,5338] fails
ghw and ghw_deep deterministically while baseline decodes it — failure sets are
non-nested, which is exactly the condition where a portfolio beats every member.
PortfolioRelay's ML selection (highest prior log-likelihood among syndrome-validating
corrections) provably picks the right member on one-sided defects in either direction.

Scoring is tag-aware: entries tagged *_tie are perfect-decoder ties — no decoder can
be expected to fix them, so they are scored SEPARATELY and never counted against a
decoder. Everything else is defect-class: fixable in principle.

All entries live in the FULL-SYMMETRIC DEM index space (the library convention);
syndromes/truths are reconstituted from that DEM and decoded by decoders set up on the
full-symmetric circuit, device-calibrated at DECODER_P.

Output: runs/decoder_loop/library_bench_portfolio.json (+ stdout table).
"""
from __future__ import annotations

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
from importance_sampling import _parse_dem
from portfolio_relay import PortfolioRelay

LIB = REPO_ROOT / "runs" / "decoder_loop" / "library.json"
OUT = REPO_ROOT / "runs" / "decoder_loop" / "library_bench_portfolio.json"
CHUNK = 64
SPECIMEN = frozenset([2482, 5202, 5289, 5338])   # the confirmed CZ w=4 defect


def main():
    lib = json.loads(LIB.read_text(encoding="utf-8"))
    entries = [e for e in lib["entries"] if e.get("code", "bb72") == "bb72"]
    ties = [e for e in entries if e.get("generator", "").endswith("_tie")]
    defects = [e for e in entries if not e.get("generator", "").endswith("_tie")]
    print(f"[lib] {len(entries)} bb72 entries: {len(defects)} defect-class, "
          f"{len(ties)} tie-class", flush=True)

    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
    probs, det, obs = _parse_dem(circ)
    calib_probs, _, _ = _parse_dem(calib)

    base = rmc.CalibratedRelayBP(calib, **rmc.DEC_CFG); base.setup(circ)
    ghw = rmc.CalibratedRelayBP(calib, **dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"]))
    ghw.setup(circ)
    deep = rmc.CalibratedRelayBP(calib, **dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_deep"]))
    deep.setup(circ)
    # primary FIRST: its prediction is the fallback when no member validates
    pf = PortfolioRelay({"ghw_deep": deep, "baseline": base}, det, obs, calib_probs)
    DECS = [("baseline", base), ("ghw", ghw), ("ghw_deep", deep), ("PORTFOLIO", pf)]

    def bench(group, label):
        syn = np.zeros((len(group), det.shape[1]), dtype=bool)
        tru = np.zeros((len(group), obs.shape[1]), dtype=bool)
        for i, e in enumerate(group):
            syn[i] = np.bitwise_xor.reduce(det[e["mechs"]], axis=0)
            tru[i] = np.bitwise_xor.reduce(obs[e["mechs"]], axis=0)
        res = {}
        for name, d in DECS:
            fails = np.zeros(len(group), dtype=bool)
            t0 = time.time()
            for lo in range(0, len(group), CHUNK):
                hi = min(lo + CHUNK, len(group))
                pred = d.decode_batch(syn[lo:hi])
                fails[lo:hi] = np.any(pred.astype(bool) != tru[lo:hi], axis=1)
                if (lo // CHUNK) % 4 == 0:
                    print(f"    [{label}] {name}: {hi}/{len(group)} "
                          f"({hi/(time.time()-t0):.1f}/s)", flush=True)
            res[name] = fails
            print(f"  [{label}] {name:10s}: fails {int(fails.sum()):4d}/{len(group)} "
                  f"({len(group)/(time.time()-t0):.1f}/s)", flush=True)
        return res, syn, tru

    t_all = time.time()
    res_d, _, _ = bench(defects, "defect")
    res_t, _, _ = bench(ties, "tie")

    # --- report ---
    def by_key(group, fails, key):
        agg = {}
        for e, f in zip(group, fails):
            k = str(e.get(key))
            a = agg.setdefault(k, [0, 0])
            a[1] += 1
            a[0] += int(f)
        return agg

    print("\n===== DEFECT-CLASS (fixable in principle; lower = better) =====")
    print(f"{'decoder':12s} {'fails':>6} / {len(defects)}")
    for name, _ in DECS:
        print(f"{name:12s} {int(res_d[name].sum()):6d}")
    print("\nby weight (fails/n):")
    ws = sorted({e['w'] for e in defects})
    hdr = f"{'decoder':12s} " + " ".join(f"{'w=' + str(w):>9s}" for w in ws)
    print(hdr)
    for name, _ in DECS:
        agg = by_key(defects, res_d[name], "w")
        print(f"{name:12s} " + " ".join(
            f"{agg.get(str(w), [0, 0])[0]:4d}/{agg.get(str(w), [0, 0])[1]:<4d}" for w in ws))
    print("\nby generator (fails/n):")
    gens = sorted({e['generator'] for e in defects})
    for name, _ in DECS:
        agg = by_key(defects, res_d[name], "generator")
        row = "  ".join(f"{g}:{agg[g][0]}/{agg[g][1]}" for g in gens if g in agg)
        print(f"{name:12s} {row}")

    idx_spec = next((i for i, e in enumerate(defects)
                     if frozenset(e["mechs"]) == SPECIMEN), None)
    if idx_spec is not None:
        print("\nthe confirmed CZ w=4 specimen: " + "  ".join(
            f"{name}={'FAIL' if res_d[name][idx_spec] else 'fix'}" for name, _ in DECS))

    print(f"\n===== TIE-CLASS (perfect-decoder ties; ~50% expected, not a defect) =====")
    for name, _ in DECS:
        print(f"{name:12s} fails {int(res_t[name].sum()):4d}/{len(ties)}")

    out = dict(
        n_defects=len(defects), n_ties=len(ties),
        defect_fails={n: int(res_d[n].sum()) for n, _ in DECS},
        tie_fails={n: int(res_t[n].sum()) for n, _ in DECS},
        defect_by_weight={n: by_key(defects, res_d[n], "w") for n, _ in DECS},
        defect_by_generator={n: by_key(defects, res_d[n], "generator") for n, _ in DECS},
        specimen={n: bool(res_d[n][idx_spec]) for n, _ in DECS} if idx_spec is not None else None,
        elapsed_s=time.time() - t_all,
    )
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}  ({out['elapsed_s']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
