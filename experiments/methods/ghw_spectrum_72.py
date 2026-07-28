#!/usr/bin/env python
"""GHW-decoder failure spectra for the [[72,4,8]] code, weights 3..10 (rodan run).

Samples the onset-region bins w=3..10 with the glo_heavy_wide Relay-BP config
(pre_iter=320, gamma0=0.0625, num_sets=200, set_max_iter=120, gamma interval (-0.5,1.0),
stop_nconv=5; priors calibrated at DECODER_P — the report's device convention) using the
SAME adaptive protocol as the baseline onset top-up (onset_topup_72.py): per-weight cap
3e6 shots by default (zero-bin bound 1e-6; raise via GHW_SHOTS_MAX when budget allows —
a deepening run pools with an earlier one only under a fresh out-dir + distinct
GHW_SEED_BASE), early stop at TARGET failures, 50k chunks. Output goes to its OWN
directory
(runs/ghw_topup_72/<name>.json) — the report cache is never touched, so cached baseline
bins vs these files is a clean decoder A/B at matched budgets. decode-only and sampling
wall time are recorded per bin (the baseline top-up did not store timing; its throughput
is known from local benches).

Spectrum list and circuit builders are shared with onset_topup_72.py (same array order,
same DECODER_P assertion path). One spectrum per invocation; resumable per weight.

    python experiments/methods/ghw_spectrum_72.py --index 1        # tech1_72__full_symmetric
    python experiments/methods/ghw_spectrum_72.py tech1_72__gate_idle
    python experiments/methods/ghw_spectrum_72.py --list

Env knobs: GHW_SHOTS_MAX (1e8), GHW_TARGET (20), GHW_CHUNK (50000), GHW_WEIGHTS ("3..10"
or "3,5,7"), GHW_OUT_DIR (runs/ghw_topup_72), GHW_SAVE_EVERY (checkpoint every N chunks,
default 20 = 1M shots), GHW_MAX_CHUNKS_PER_RUN (graceful time-boxing/sharding: exit 0
after N chunks with a checkpoint; rerun continues).

COST GUIDANCE (local 24-thread box, decode-dominated): only bins whose true f(w) is
below ~TARGET/cap = 2e-7 burn the full 1e8 cap; everything else early-stops at TARGET
failures (shots ≈ TARGET/f). But a CAPPED bin is now multi-day: measured GHW throughput
at w=3 — full symmetric ~200/s (1e8 ≈ 6 days/bin), meas/prep ~60/s (~19 days/bin), gate
idle ~25/s (~46 days/bin). Intra-weight checkpoints (per-chunk seeding) make kills
harmless, but budget the wall time deliberately: start with full symmetric, and consider
GHW_WEIGHTS/GHW_SHOTS_MAX overrides per channel rather than 1e8 across the board.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_error_model_comparison as R
from onset_topup_72 import SPECTRA, build
from importance_sampling import _parse_dem, _expand

# Decoder variant (GHW_VARIANT env): any key of R.DECODER_VARIANTS — "ghw" (nc5) or
# "ghw_nc1" (first-converged stopping; 3.5-1500x faster, harvest-equivalent). Applied on
# top of the runner's DEC_CFG (which itself honors EMC_DECODER — leave that unset here).
VARIANT = os.environ.get("GHW_VARIANT", "ghw")
GHW_CFG = dict(R.DEC_CFG, **R.DECODER_VARIANTS[VARIANT])

SHOTS_MAX = int(float(os.environ.get("GHW_SHOTS_MAX", "3e6")))
TARGET = int(os.environ.get("GHW_TARGET", "20"))
CHUNK = int(os.environ.get("GHW_CHUNK", "50000"))
SAVE_EVERY = int(os.environ.get("GHW_SAVE_EVERY", "20"))    # checkpoint every N chunks (1M shots)
MAX_CHUNKS_PER_RUN = int(float(os.environ.get("GHW_MAX_CHUNKS_PER_RUN", "inf"))
                         if os.environ.get("GHW_MAX_CHUNKS_PER_RUN") else 1 << 62)
# Distinct from campaign (4) and baseline top-up (104). Override (GHW_SEED_BASE) for a
# SECOND deepening run into a different out-dir whose bins will be POOLED with this one —
# same base would replay identical per-chunk draws and double-count them.
SEED_BASE = int(os.environ.get("GHW_SEED_BASE", "204"))
OUT_DIR = pathlib.Path(os.environ.get("GHW_OUT_DIR", "runs/ghw_topup_72"))


def parse_weights(spec: str):
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


WEIGHTS = parse_weights(os.environ.get("GHW_WEIGHTS", "3..10"))


def sample_chunk(rng, col_to_mech, det_mat, obs_mat, w, B):
    """B uniform weight-w fault sets in the expanded representation (with-replacement +
    duplicate-row rejection; matches subonset_relay_sweep.sample_batch)."""
    N_exp = col_to_mech.shape[0]
    idx = rng.integers(0, N_exp, size=(B, w))
    if w > 1:
        while True:
            s = np.sort(idx, axis=1)
            bad = (s[:, 1:] == s[:, :-1]).any(axis=1)
            if not bad.any():
                break
            idx[bad] = rng.integers(0, N_exp, size=(int(bad.sum()), w))
    mechs = col_to_mech[idx]
    return (np.bitwise_xor.reduce(det_mat[mechs], axis=1),
            np.bitwise_xor.reduce(obs_mat[mechs], axis=1))


def run(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    cfg_json = {k: list(v) if isinstance(v, tuple) else v for k, v in GHW_CFG.items()}
    if path.exists():
        j = json.loads(path.read_text(encoding="utf-8"))
        # Resume guard (mirrors the campaign convention): never resume against different
        # decoder settings or budgets — the pooled bins would mix estimators.
        assert j["ghw_cfg"] == cfg_json, f"{name}: existing file has different ghw_cfg"
        assert j["shots_max"] == SHOTS_MAX and j["target"] == TARGET, \
            f"{name}: existing file has different budgets (shots_max/target)"
    else:
        j = dict(name=name, decoder=VARIANT, ghw_cfg=cfg_json,
                 decoder_p=R.DECODER_P, p_ref=R.P_REF, seed_base=SEED_BASE,
                 shots_max=SHOTS_MAX, target=TARGET, bins={})

    circ, calib = build(name)
    probs, det_mat, obs_mat = _parse_dem(circ)
    col_to_mech, q_base, _ = _expand(probs, None)
    j["n_expanded"], j["q_base"] = int(col_to_mech.shape[0]), float(q_base)

    dec = R.CalibratedRelayBP(calib, **GHW_CFG)
    dec.setup(circ)

    # Seeding is PER CHUNK (rng = f(seed_base, w, name, chunk_index)), not per weight, so a
    # bin interrupted mid-flight resumes from its partial checkpoint with independent,
    # non-overlapping draws — at a 1e8 cap a capped bin is a multi-day decode and MUST NOT
    # redo from scratch. `partial` holds in-flight bins; a bin moves to `bins` when done.
    name_h = abs(hash(name)) % (1 << 31)
    chunks_this_run = 0
    partial = j.setdefault("partial", {})

    def _seeded_chunk(w, k, B):
        rng = np.random.default_rng([SEED_BASE, w, name_h, k])
        return sample_chunk(rng, col_to_mech, det_mat, obs_mat, w, B)

    for w in WEIGHTS:
        if str(w) in j["bins"]:
            print(f"[{name}] w={w}: done ({j['bins'][str(w)]}); skip", flush=True)
            continue
        p = partial.get(str(w), dict(chunk=0, trials=0, fails=0, decode_s=0.0))
        k, T, F, t_dec = p["chunk"], p["trials"], p["fails"], float(p["decode_s"])
        if T:
            print(f"[{name}] w={w}: resuming at chunk {k} ({T} trials, {F} fails)", flush=True)
        t0 = time.perf_counter()
        while F < TARGET and T < SHOTS_MAX:
            if chunks_this_run >= MAX_CHUNKS_PER_RUN:
                partial[str(w)] = dict(chunk=k, trials=T, fails=F, decode_s=round(t_dec, 1))
                path.write_text(json.dumps(j, indent=1), encoding="utf-8")
                print(f"[{name}] w={w}: chunk budget for this run reached at {T} trials — "
                      f"checkpointed, rerun to continue", flush=True)
                return
            B = min(CHUNK, SHOTS_MAX - T)
            syn, tru = _seeded_chunk(w, k, B)
            td = time.perf_counter()
            pred = dec.decode_batch(syn)
            t_dec += time.perf_counter() - td
            F += int(np.any(pred != tru, axis=1).sum())
            T += B
            k += 1
            chunks_this_run += 1
            if k % SAVE_EVERY == 0:
                partial[str(w)] = dict(chunk=k, trials=T, fails=F, decode_s=round(t_dec, 1))
                path.write_text(json.dumps(j, indent=1), encoding="utf-8")
                print(f"[{name}] w={w}: {T}/{SHOTS_MAX}, fails={F} "
                      f"({T/max(t_dec,1e-9):,.0f} dec/s) [checkpoint]", flush=True)
        j["bins"][str(w)] = dict(trials=T, fails=F, decode_s=round(t_dec, 1),
                                 wall_s=round(time.perf_counter() - t0, 1))
        partial.pop(str(w), None)
        path.write_text(json.dumps(j, indent=1), encoding="utf-8")   # bin complete
        print(f"[{name}] w={w}: DONE {F}/{T} (f={F/max(T,1):.2e}) "
              f"decode {t_dec:,.0f}s ({T/max(t_dec,1e-9):,.0f}/s)", flush=True)
    print(f"[{name}] all weights done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", help="spectrum task name (e.g. tech1_72__full_symmetric)")
    ap.add_argument("--index", type=int, help="SPECTRA[index] (same order as onset_topup_72)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for i, n in enumerate(SPECTRA):
            print(f"{i:2d}  {n}")
        return
    name = args.name or (SPECTRA[args.index] if args.index is not None else None)
    if name is None:
        ap.error("give a spectrum name, --index N, or --list")
    print(f"GHW spectrum: {name}  (w={WEIGHTS}, cap={SHOTS_MAX}, target={TARGET}, "
          f"decoder_p={R.DECODER_P}, out={OUT_DIR})", flush=True)
    run(name)


if __name__ == "__main__":
    main()
