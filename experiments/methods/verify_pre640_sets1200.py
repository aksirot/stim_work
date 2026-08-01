"""Verification of pre640_sets1200 — the 0/90 sub-onset certified candidate.

(1) Paired McNemar vs nb_pre_x2 (the closing-bench leader) at w=10 and w=14,
    5k fresh shots each: promotion-grade evidence that the sub-onset win does not
    cost above-onset performance.
(2) Full 480-entry library score + risk + throughput probe (closing-table row).
Writes runs/decoder_loop/verify_pre640_sets1200.json
"""
import json
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import binom, binomtest

from decoder_loop import Ctx, lib_load, GHW_CFG, OUT
import run_error_model_comparison as rmc

def V(**kw):
    d = dict(GHW_CFG); d.update(kw); return d

CAND = V(pre_iter=640, num_sets=1200)
RIVAL = V(pre_iter=640)                      # nb_pre_x2


def main():
    ctx = Ctx()
    cand, rival = ctx.decoder(CAND), ctx.decoder(RIVAL)
    rng = np.random.default_rng(1111)

    # (1) paired verify
    b01 = b10 = f_c = f_r = n = 0
    for w in (10, 14):
        done = 0
        while done < 5_000:
            B = 1_000
            _, syn, tru = ctx.sample(rng, w, B)
            xr = np.any(rival.decode_batch(syn) != tru, axis=1)
            xc = np.any(cand.decode_batch(syn) != tru, axis=1)
            b01 += int((xr & ~xc).sum()); b10 += int((~xr & xc).sum())
            f_r += int(xr.sum()); f_c += int(xc.sum())
            done += B; n += B
            print(f"  [verify] w={w}: {done}/5000  cum b01={b01} b10={b10}", flush=True)
    p = binomtest(b10, b10 + b01, 0.5, alternative="greater").pvalue if (b10 + b01) else 1.0
    ok = p > 0.05
    print(f"[verify] paired {n}: rival {f_r} vs cand {f_c}  b01={b01} b10={b10} "
          f"p={p:.3f} -> {'PASS' if ok else 'REJECT'}", flush=True)

    # (2) full-library row
    lib = lib_load()
    supports = [sorted(e["mechs"]) for e in lib["entries"]]
    ws = np.array([e["w"] for e in lib["entries"]])
    syn, tru = ctx.syn_tru_of_mechs(supports)
    bad = np.any(cand.decode_batch(syn) != tru, axis=1)
    q_star = ctx.q_base * (rmc.DECODER_P / rmc.P_REF)
    pw = binom.pmf(ws, ctx.N_exp, q_star)
    n_of_w = {int(w): int((ws == w).sum()) for w in set(ws.tolist())}
    ent_wt = pw / np.array([n_of_w[int(w)] for w in ws])
    risk = float(ent_wt[bad].sum())
    _, psyn, _ = ctx.sample(rng, 12, 300)
    t0 = time.time(); cand.decode_batch(psyn); rate = 300 / (time.time() - t0)
    w3_fails = int(bad[ws == 3].sum())
    print(f"[library] fixes {int((~bad).sum())}/{len(supports)}  risk {risk:.3e}  "
          f"w3 {w3_fails}/{int((ws == 3).sum())}  rate {rate:.1f}/s", flush=True)

    out = dict(candidate="pre640_sets1200",
               cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in CAND.items()},
               verify=dict(shots=n, rival="nb_pre_x2", rival_fails=f_r, cand_fails=f_c,
                           b01=b01, b10=b10, p_regress=float(p), passes=bool(ok)),
               library=dict(n=len(supports), fixes=int((~bad).sum()), risk=risk,
                            w3_fails=w3_fails, rate=round(rate, 1)))
    (OUT / "verify_pre640_sets1200.json").write_text(json.dumps(out, indent=1),
                                                     encoding="utf-8")
    print("wrote verify_pre640_sets1200.json", flush=True)


if __name__ == "__main__":
    main()
