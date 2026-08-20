"""Reproduce fail-fast Table 4 bins for the Y1 in-module measurement.

Target (paper p.35, Table 4): w=100 -> F/T = 94/1826 = 5.15e-2, delineated
F_mem=72, F_out=38, F_both=16 (F = mem + out - both). Also w=90: 100/7844,
w=110: 88/426 for the shoulder shape.

Circuit: gross_lpu_y1_match.yaml (C=6, d_init=3, flat per-round idle) — the
closest realizable match to the paper's Table-1 fingerprint (79,470 vs 79,591
columns; caveats: our layering isn't their graph-coloring schedule, expansion
ratio 3.80 vs 5.03, K=12 minimal framing vs their K=24 Bell harness).

Delineation: obs 0 = the Y1 outcome; obs 1..11 = memory logicals.
  F_out  = shots where obs 0 mispredicted (with or without memory errors)
  F_mem  = shots where any of obs 1..11 mispredicted
  F_both = both;  F = F_mem + F_out - F_both  (the paper's identity)

    BIN_WS=100 BIN_SHOTS=2000 BIN_DECODER=deep600 python experiments/methods/y1_table4_bin.py
    BIN_WS=90,100,110 BIN_SHOTS=3000,2000,1000 BIN_DECODER=campaign ...

Output: runs/framework/bb144/y1_table4_<decoder>.json (incremental per weight).
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from experiment_runner import load_config, build_circuit, make_decoder
from bb_code_sim import RelayBPDecoder
from importance_sampling import _parse_dem, _expand
from subonset_relay_sweep import sample_batch
from repo_paths import REPO_ROOT

DEEP600 = dict(gamma0=0.125, pre_iter=80, num_sets=600, set_max_iter=60,
               gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
WS = [int(x) for x in os.environ.get("BIN_WS", "100").split(",")]
SHOTS = [int(x) for x in os.environ.get("BIN_SHOTS", "2000").split(",")]
DEC = os.environ.get("BIN_DECODER", "deep600")
CHUNK = 64
OUT = REPO_ROOT / "runs" / "framework" / "bb144" / f"y1_table4_{DEC}.json"
PAPER = {90: (7844, 100, 53, 67, 20), 100: (1826, 94, 72, 38, 16),
         110: (426, 88, 80, 29, 21)}   # w: (T, F, Fmem, Fout, Fboth)


def main():
    cfg = load_config(str(REPO_ROOT / "experiments/configs/gross_lpu_y1_match.yaml"))
    circ = build_circuit(cfg)
    dec = RelayBPDecoder(**DEEP600) if DEC == "deep600" else make_decoder(cfg)
    dec.setup(circ)
    probs, det, obs = _parse_dem(circ)
    c2m, qb, _ = _expand(probs, None)
    print(f"[y1t4] comp={det.shape[0]:,} K={obs.shape[1]} mu={c2m.shape[0]*qb:.0f} "
          f"decoder={DEC}", flush=True)
    rng = np.random.default_rng(2026)

    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for w, T in zip(WS, SHOTS):
        if str(w) in results:
            print(f"[y1t4] w={w}: cached, skip", flush=True)
            continue
        f_out = f_mem = f_both = fails = 0
        done, t0, dec_t = 0, time.time(), 0.0
        while done < T:
            B = min(CHUNK, T - done)
            _, syn, tru = sample_batch(rng, c2m, det, obs, w, B)
            td = time.time()
            pred = dec.decode_batch(syn)
            dec_t += time.time() - td
            bad = pred.astype(bool) != tru.astype(bool)      # per-shot, per-obs
            out_err = bad[:, 0]
            mem_err = bad[:, 1:].any(axis=1)
            fails += int((out_err | mem_err).sum())
            f_out += int(out_err.sum())
            f_mem += int(mem_err.sum())
            f_both += int((out_err & mem_err).sum())
            done += B
            print(f"  [w={w}] {fails}F ({f_mem}m/{f_out}o/{f_both}b) / {done} "
                  f"({done/(time.time()-t0):.2f}/s, {dec_t/done:.2f}s/dec)", flush=True)
        row = dict(w=w, trials=done, F=fails, F_mem=f_mem, F_out=f_out,
                   F_both=f_both, f=fails/done, tdec_mean=dec_t/done,
                   decoder=DEC, paper=dict(zip(("T", "F", "F_mem", "F_out", "F_both"),
                                               PAPER.get(w, ()))))
        results[str(w)] = row
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        pf = PAPER.get(w)
        cmp_ = (f"  paper: {pf[1]}/{pf[0]} = {pf[1]/pf[0]:.3e} "
                f"({pf[2]}m/{pf[3]}o/{pf[4]}b)" if pf else "")
        print(f"[y1t4] w={w}: OURS {fails}/{done} = {fails/done:.3e} "
              f"({f_mem}m/{f_out}o/{f_both}b){cmp_}", flush=True)


if __name__ == "__main__":
    main()
