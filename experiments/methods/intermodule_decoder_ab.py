"""Decoder A/B on the inter-module (r1) circuit: cheap campaign relay vs paper-grade.

Motivation (2026-08-18): our inter-module joint-parity LER at p=1e-3 is 0.186 (fair
couplers, K=1, a=1/2) vs Tour de Gross Fig 7's ~2e-3 for the SAME operation under a
harsher any-logical-failure count — a real ~100x gap. Two suspects: the framework
campaign's deliberately cheap relay (num_sets=20) vs the paper's production config,
or the adapter construction itself (lost deformed-code distance). This paired A/B
separates them: sample fixed-weight fault configs, decode the IDENTICAL configs with
both decoders, compare f(w) and the McNemar discordants.

  paper-grade f(w) drops ~100x  -> the gap is decoder quality; construction is fine
  paper-grade f(w) barely moves -> the merge itself is defective; audit the adapter

    AB_WEIGHTS=60,100,150 AB_SHOTS=2000 python experiments/methods/intermodule_decoder_ab.py

Output: runs/framework/bb144/inter_module_ab.json (incremental per weight) + stdout.
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

YAML = REPO_ROOT / "experiments" / "configs" / "gross_intermodule_r1.yaml"
OUT = REPO_ROOT / "runs" / "framework" / "bb144" / "inter_module_ab.json"
WEIGHTS = [int(x) for x in os.environ.get("AB_WEIGHTS", "60,100,150").split(",")]
SHOTS = int(os.environ.get("AB_SHOTS", "2000"))
CHUNK = 128

# "paper-grade" = 600 relay sets with the CAMPAIGN's validated gammas (deep600).
# The literal published-gamma transcription (gamma0=0.79, interval (0.34,1.24)) is
# catastrophically wrong under our wrapper — 2026-08-18 factorization on idle w=100:
# paper-gammas 85/128 fails at 20 sets vs cheap-gammas 11/128; cheap-gammas at 600
# sets: 2/128. Set count is the production axis; those gammas are a convention trap.
PAPER = dict(gamma0=0.125, pre_iter=80, num_sets=600, set_max_iter=60,
             gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)


def decode_masked(dec, syn, tru, label):
    fails = np.zeros(len(syn), dtype=bool)
    t0 = time.time()
    for lo in range(0, len(syn), CHUNK):
        hi = min(lo + CHUNK, len(syn))
        pred = dec.decode_batch(syn[lo:hi])
        fails[lo:hi] = np.any(pred.astype(bool) != tru[lo:hi].astype(bool), axis=1)
        el = time.time() - t0
        print(f"    [{label}] {hi}/{len(syn)} ({hi/el:.2f}/s, {int(fails[:hi].sum())} fails)",
              flush=True)
    return fails


def main():
    cfg = load_config(str(YAML))
    circ = build_circuit(cfg)
    cheap = make_decoder(cfg)
    cheap.setup(circ)
    paper = RelayBPDecoder(**PAPER)
    paper.setup(circ)
    probs, det, obs = _parse_dem(circ)
    c2m, _, _ = _expand(probs, None)
    rng = np.random.default_rng(2026)
    print(f"[ab] inter_module r1: {det.shape[0]:,} mechs ({c2m.shape[0]:,} expanded); "
          f"cheap=num_sets20  paper=num_sets600", flush=True)

    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
    for w in WEIGHTS:
        if str(w) in results:
            print(f"[ab] w={w}: cached, skip", flush=True)
            continue
        print(f"[ab] w={w}: sampling {SHOTS} configs", flush=True)
        _, syn, tru = sample_batch(rng, c2m, det, obs, w, SHOTS)
        fc = decode_masked(cheap, syn, tru, f"w={w} cheap")
        fp = decode_masked(paper, syn, tru, f"w={w} paper")
        b01 = int((fc & ~fp).sum())     # cheap fails, paper fixes
        b10 = int((~fc & fp).sum())     # paper fails, cheap fixes
        row = dict(shots=SHOTS, cheap_fails=int(fc.sum()), paper_fails=int(fp.sum()),
                   b01=b01, b10=b10,
                   f_cheap=float(fc.mean()), f_paper=float(fp.mean()),
                   ratio=(float(fc.sum()) / fp.sum() if fp.sum() else None))
        results[str(w)] = row
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"[ab] w={w}: cheap {row['cheap_fails']}/{SHOTS} vs paper "
              f"{row['paper_fails']}/{SHOTS}  (b01={b01} b10={b10})  -> wrote", flush=True)
    print("[ab] done", flush=True)


if __name__ == "__main__":
    main()
