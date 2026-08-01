"""Deadline-free w=3 certification: which decoder clears ALL sub-onset library entries?

The loop's target (user, 2026-08-01): a verified decoder with ZERO failures at
sub-onset weights on the enriched library. w=3 < w0=4, so every w=3 failure is pure
miscorrection — a perfect decoder fixes all of them; this sweep measures which relay
configs / portfolios come closest, including two-knob combos the one-knob neighborhood
cannot reach. Each candidate decodes the full w=3 set TWICE (relay is stochastic;
a config only certifies if BOTH passes are clean).

Output: runs/decoder_loop/w3_certify.json + .log
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from decoder_loop import Ctx, lib_load, GHW_CFG, OUT
import run_error_model_comparison as rmc
from portfolio_relay import PortfolioRelay
from importance_sampling import _parse_dem


def V(**kw):
    d = dict(GHW_CFG); d.update(kw); return d


CANDIDATES = {
    "ghw":            dict(GHW_CFG),
    "nb_pre_x2":      V(pre_iter=640),
    "heavy":          dict(rmc.DEC_CFG, gamma0=0.125, pre_iter=320, num_sets=200,
                           set_max_iter=120, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5),
    # two-knob combos (unreachable by one-knob neighborhood):
    "pre640_sets600": V(pre_iter=640, num_sets=600),
    "pre640_heavygamma": V(pre_iter=640, gamma0=0.125, gamma_dist_interval=(-0.24, 0.66)),
    "pre1280_sets600": V(pre_iter=1280, num_sets=600),
    "pre640_iter240": V(pre_iter=640, set_max_iter=240),
}
PORTFOLIOS = [("ghw+nb_pre_x2", ["ghw", "nb_pre_x2"]),
              ("ghw+heavy+nb_pre_x2", ["ghw", "heavy", "nb_pre_x2"]),
              ("nb_pre_x2+pre640_sets600", ["nb_pre_x2", "pre640_sets600"])]


def main():
    ctx = Ctx()
    lib = lib_load()
    w3 = [e for e in lib["entries"] if e["w"] == 3]
    supports = [sorted(e["mechs"]) for e in w3]
    gens = [e["generator"] for e in w3]
    syn, tru = ctx.syn_tru_of_mechs(supports)
    print(f"w3 certification: {len(w3)} sub-onset entries "
          f"({len(set(gens))} sources)", flush=True)

    probs, det, obs = _parse_dem(ctx.circ)
    calib_probs, _, _ = _parse_dem(ctx.calib)
    results = {}
    fails_of = {}
    for name, cfg in CANDIDATES.items():
        dec = ctx.decoder(cfg)
        bad1 = np.any(dec.decode_batch(syn) != tru, axis=1)
        bad2 = np.any(dec.decode_batch(syn) != tru, axis=1)
        bad = bad1 | bad2                      # certify only if BOTH passes clean
        fails_of[name] = bad
        by_gen = {}
        for g, b in zip(gens, bad):
            if b:
                by_gen[g] = by_gen.get(g, 0) + 1
        results[name] = dict(kind="single", fails=int(bad.sum()), n=len(w3),
                             flaky=int((bad1 != bad2).sum()), by_gen=by_gen)
        print(f"[single]    {name:22s} fails {int(bad.sum()):3d}/{len(w3)} "
              f"(flaky {int((bad1 != bad2).sum())})  {by_gen}", flush=True)

    for label, names in PORTFOLIOS:
        pf = PortfolioRelay({n: ctx.decoder(CANDIDATES[n]) for n in names},
                            det, obs, calib_probs)
        bad1 = np.any(pf.decode_batch(syn) != tru, axis=1)
        bad2 = np.any(pf.decode_batch(syn) != tru, axis=1)
        bad = bad1 | bad2
        union = np.logical_and.reduce([fails_of[n] for n in names])
        by_gen = {}
        for g, b in zip(gens, bad):
            if b:
                by_gen[g] = by_gen.get(g, 0) + 1
        results[label] = dict(kind="portfolio", fails=int(bad.sum()), n=len(w3),
                              flaky=int((bad1 != bad2).sum()),
                              oracle_fails=int(union.sum()), by_gen=by_gen)
        print(f"[portfolio] {label:22s} fails {int(bad.sum()):3d}/{len(w3)} "
              f"(oracle {int(union.sum())}, flaky {int((bad1 != bad2).sum())})  "
              f"{by_gen}", flush=True)

    out = OUT / "w3_certify.json"
    out.write_text(json.dumps(dict(n_w3=len(w3), results=results), indent=1),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
