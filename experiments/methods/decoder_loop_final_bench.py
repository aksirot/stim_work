"""Deadline-free closing bench for the decoder-selection loop.

The loop's iterations 4-6 hit a plateau (paired tests statistically even, promotions
on ~4% library-risk margins) and the boxed bench never scored the full roster on the
full library. This one-off scores the finalists + portfolio combos on ALL library
entries with real throughput probes — the recommendation table for the final report.
Writes runs/decoder_loop/final_bench.json.
"""
import json
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binom

from decoder_loop import Ctx, lib_load, GHW_CFG, OUT
import run_error_model_comparison as rmc
from portfolio_relay import PortfolioRelay
from importance_sampling import _parse_dem

FINALISTS = {
    "baseline":  dict(rmc.DEC_CFG),
    "ghw":       dict(GHW_CFG),
    "heavy":     dict(rmc.DEC_CFG, gamma0=0.125, pre_iter=320, num_sets=200,
                      set_max_iter=120, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5),
    "nb_pre_x2": dict(GHW_CFG, pre_iter=640),
    "nb_g0_x2":  dict(GHW_CFG, pre_iter=640, gamma0=0.125),
}
PORTFOLIOS = [("ghw+nb_g0_x2", ["ghw", "nb_g0_x2"]),
              ("heavy+nb_g0_x2", ["heavy", "nb_g0_x2"]),
              ("ghw+heavy+nb_g0_x2", ["ghw", "heavy", "nb_g0_x2"])]


def main():
    ctx = Ctx()
    lib = lib_load()
    supports = [sorted(e["mechs"]) for e in lib["entries"]]
    ws = np.array([e["w"] for e in lib["entries"]])
    syn, tru = ctx.syn_tru_of_mechs(supports)
    print(f"final bench: library n={len(supports)}, weights {ws.min()}..{ws.max()}", flush=True)

    q_star = ctx.q_base * (rmc.DECODER_P / rmc.P_REF)
    pw = binom.pmf(ws, ctx.N_exp, q_star)
    n_of_w = {int(w): int((ws == w).sum()) for w in set(ws.tolist())}
    ent_wt = pw / np.array([n_of_w[int(w)] for w in ws])

    rng = np.random.default_rng(707)
    _, psyn, _ = ctx.sample(rng, 12, 400)          # throughput probe at a mid weight

    rows, fails_of = [], {}
    for name, cfg in FINALISTS.items():
        dec = ctx.decoder(cfg)
        bad = np.any(dec.decode_batch(syn) != tru, axis=1)
        fails_of[name] = bad
        t0 = time.time(); dec.decode_batch(psyn); rate = len(psyn) / (time.time() - t0)
        w3 = int(bad[ws == 3].sum())
        rows.append(dict(kind="single", name=name, fixes=int((~bad).sum()),
                         n=len(supports), risk=float(ent_wt[bad].sum()),
                         w3_fails=w3, w3_n=int((ws == 3).sum()),
                         onset_ub=int(ws[bad].min()) if bad.any() else None,
                         rate=round(rate, 1)))
        print(f"[single]    {name:20s} fixes {rows[-1]['fixes']:3d}/{len(supports)}  "
              f"risk {rows[-1]['risk']:.3e}  w3 {w3}/{rows[-1]['w3_n']}  "
              f"rate {rate:.1f}/s", flush=True)

    probs, det, obs = _parse_dem(ctx.circ)
    calib_probs, _, _ = _parse_dem(ctx.calib)
    for label, names in PORTFOLIOS:
        pf = PortfolioRelay({n: ctx.decoder(FINALISTS[n]) for n in names},
                            det, obs, calib_probs)
        bad = np.any(pf.decode_batch(syn) != tru, axis=1)
        t0 = time.time(); pf.decode_batch(psyn); rate = len(psyn) / (time.time() - t0)
        union = np.logical_and.reduce([fails_of[n] for n in names])
        w3 = int(bad[ws == 3].sum())
        rows.append(dict(kind="portfolio", name=label, fixes=int((~bad).sum()),
                         n=len(supports), risk=float(ent_wt[bad].sum()),
                         w3_fails=w3, w3_n=int((ws == 3).sum()),
                         onset_ub=int(ws[bad].min()) if bad.any() else None,
                         rate=round(rate, 1), oracle_fixes=int((~union).sum())))
        print(f"[portfolio] {label:20s} fixes {rows[-1]['fixes']:3d}/{len(supports)}  "
              f"risk {rows[-1]['risk']:.3e}  w3 {w3}/{rows[-1]['w3_n']}  "
              f"rate {rate:.1f}/s  (oracle {rows[-1]['oracle_fixes']})", flush=True)

    out = OUT / "final_bench.json"
    out.write_text(json.dumps(dict(library_n=len(supports), q_star=float(q_star),
                                   rows=rows), indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
