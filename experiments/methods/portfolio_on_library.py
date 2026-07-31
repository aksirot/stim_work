"""Bench PortfolioRelay combinations against the decoder-loop failure library.

Standalone read-only check (the loop's v1 bench is singles-only by design): does a
portfolio of the library's complementary leaders beat every single config?
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from decoder_loop import Ctx, lib_load, GHW_CFG
import run_error_model_comparison as rmc
from portfolio_relay import PortfolioRelay
from importance_sampling import _parse_dem

HEAVY_CFG = dict(rmc.DEC_CFG, gamma0=0.125, pre_iter=320, num_sets=200,
                 set_max_iter=120, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
NB_PRE_X2 = dict(GHW_CFG, pre_iter=640)

def main():
    ctx = Ctx()
    lib = lib_load()
    entries = lib["entries"]
    supports = [sorted(e["mechs"]) for e in entries]
    ws = [e["w"] for e in entries]
    syn, tru = ctx.syn_tru_of_mechs(supports)
    print(f"library n={len(entries)}, weights {min(ws)}..{max(ws)}")

    probs, det, obs = _parse_dem(ctx.circ)
    calib_probs, _, _ = _parse_dem(ctx.calib)

    members = {
        "baseline": ctx.decoder(rmc.DEC_CFG),
        "ghw": ctx.decoder(GHW_CFG),
        "heavy": ctx.decoder(HEAVY_CFG),
        "nb_pre_x2": ctx.decoder(NB_PRE_X2),
    }
    combos = [("baseline+ghw", ["baseline", "ghw"]),
              ("heavy+nb_pre_x2", ["heavy", "nb_pre_x2"]),
              ("heavy+nb+ghw", ["heavy", "nb_pre_x2", "ghw"])]

    # singles first (also warms the member caches)
    fails_of = {}
    for name, dec in members.items():
        x = np.any(dec.decode_batch(syn) != tru, axis=1)
        fails_of[name] = x
        ub = min((w for w, f in zip(ws, x) if f), default=None)
        print(f"  single {name:12s}: fixes {int((~x).sum()):3d}/{len(entries)}  onset_ub={ub}")

    for label, names in combos:
        pf = PortfolioRelay({n: members[n] for n in names}, det, obs, calib_probs)
        x = np.any(pf.decode_batch(syn) != tru, axis=1)
        ub = min((w for w, f in zip(ws, x) if f), default=None)
        # oracle union: how many entries at least ONE member fixes (portfolio ceiling)
        u = np.logical_and.reduce([fails_of[n] for n in names])
        print(f"  PORTFOLIO {label:16s}: fixes {int((~x).sum()):3d}/{len(entries)}  "
              f"onset_ub={ub}   (oracle-union ceiling {int((~u).sum())})")

if __name__ == "__main__":
    main()
