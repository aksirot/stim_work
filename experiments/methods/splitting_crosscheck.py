"""Technique-III (replica-exchange / Metropolis splitting) cross-check runs.

IS cannot resolve the small low-weight bins on the bigger codes; splitting is
decoder-in-the-loop and weight-agnostic, so its P_fail(p) ladder cross-checks the
reweighted-IS/ansatz LER curves — and would expose a sub-onset decoder floor the IS
window cannot see. Targets (the [[18,4,4]] ladders are already cached as tech3__*):

    python experiments/methods/splitting_crosscheck.py --target 72_full_ghw
    python experiments/methods/splitting_crosscheck.py --target lpu_idle_camp
    python experiments/methods/splitting_crosscheck.py --target lpu_idle_ghw
    python experiments/methods/splitting_crosscheck.py --smoke      # 18-code regression

Output: runs/splitting_crosscheck/<target>.json — tech3 schema (sp/sP/sP_se ascending in
p) plus the FULL diagnostics (per-pair swap_accept, per-rung mean_weight) and the
honesty gate: a ladder point is quotable only while min(swap_accept) above it exceeds
SWAP_GATE and mean_weight decreases monotonically down-ladder; `gate_rung` marks the
deepest quotable rung and `p_quotable_min` its p. The idle ladders are expected to gate
above p_low (their bottom rungs cannot be seeded near onset without L(D), which was
never persisted) — that is a stated quotable-depth, not a failure.

Seeding per target:
  72_full_ghw   : internal min-weight search (K=4 -> 15 systematic + <=400 random
                  BP-OSD decodes, minutes) with distance=8 pinned.
  lpu_idle_*    : NO search (it would be 4095 serial BP-OSD decodes at 70k columns).
                  Instead: harvest failing configs by fixed-weight sampling at the
                  lowest weights the cached idle spectrum shows measurable f(w),
                  mapped to mechanism supports; gap_weights inflate from there.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

# The 72-code target composes its decoder explicitly — the EMC env switches must not
# leak into the imported runner module.
for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS"):
    os.environ.pop(_k, None)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_error_model_comparison as rmc
from subonset_relay_sweep import sample_batch

from bb_code_sim import RelayBPDecoder
from importance_sampling import _parse_dem, _expand
from splitting import replica_exchange_estimate
from repo_paths import REPO_ROOT

OUT_DIR = REPO_ROOT / "runs" / "splitting_crosscheck"
SWAP_GATE = 0.05
GHW_CFG = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"])


def gate(diag):
    """(gate_rung, reason): deepest rung quotable under the swap/mean-weight gate.

    Rung 0 = p_high. Rung i is quotable if every adjacent swap pair above it accepts
    > SWAP_GATE and mean_weight is non-increasing (2% tolerance) down to it.
    """
    sw = list(diag["swap_accept"])
    mw = list(diag["mean_weight"])
    rung = len(mw) - 1
    for i in range(len(sw)):
        if sw[i] <= SWAP_GATE:
            return i, f"swap_accept[{i}]={sw[i]:.3f} <= {SWAP_GATE}"
        if mw[i + 1] > mw[i] * 1.02:
            return i, f"mean_weight rises at rung {i + 1} ({mw[i]:.1f} -> {mw[i + 1]:.1f})"
    return rung, "full ladder quotable"


def save(target, temper, diag, extra, elapsed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g, why = gate(diag)
    sp = list(np.asarray(temper.p_ladder, float))          # descending (p_high..p_low)
    out = dict(
        target=target,
        sp=sp[::-1], sP=list(np.asarray(temper.P_logical, float))[::-1],
        sP_se=list(np.asarray(temper.P_logical_se, float))[::-1],
        swap_accept=list(map(float, diag["swap_accept"])),
        mean_weight=list(map(float, diag["mean_weight"])),
        P_high=float(temper.P_high), P_high_se=float(temper.P_high_se),
        n_pool=int(diag["n_pool"]), n_collect=int(diag["n_collect"]),
        gate_rung=int(g), gate_reason=why,
        p_quotable_min=float(sp[g]),
        swap_gate=SWAP_GATE, elapsed_s=round(elapsed, 1), **extra)
    path = OUT_DIR / f"{target}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[{target}] gate: quotable down to p={sp[g]:.3e} (rung {g}/{len(sp)-1}) — {why}")
    print(f"[{target}] swap_accept {min(out['swap_accept']):.2f}..{max(out['swap_accept']):.2f}   "
          f"P(p_low)={out['sP'][0]:.3e} +-{out['sP_se'][0]:.1e}")
    print(f"[{target}] wrote {path} ({elapsed:,.0f}s)")


def run_72_full_ghw():
    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
    dec = rmc.CalibratedRelayBP(calib, **GHW_CFG)
    t0 = time.time()
    temper, diag = replica_exchange_estimate(
        circ, dec, p_ref=rmc.P_REF, p_high=0.008, p_low=1e-4, n_levels=16,
        n_walkers=8, local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000,
        distance=8, mw_supports=None, seed_p_grid=[3e-3, 1e-3],
        gap_weights=[10, 16, 26], seed=3, single_sector=False, verbose=True)
    save("72_full_ghw", temper, diag,
         dict(code="BB_72_4_8", model="full symmetric", decoder="ghw",
              decoder_cfg={k: list(v) if isinstance(v, tuple) else v for k, v in GHW_CFG.items()},
              calibrated_at=rmc.DECODER_P, p_ref=rmc.P_REF), time.time() - t0)


def strip_configs(det, obs, c2m, dec, configs, rng, max_passes=24, decode_cap=20_000):
    """Greedy failure-preserving stripping: descend each failing config toward a locally
    minimal failing set.

    Exploits the add/remove asymmetry: adding faults keeps failure, removing usually
    breaks it — so locally minimal failing sets are rare and valuable cold-rung seeds.
    Per pass, batch-decode every single-fault removal of a config and keep a random one
    that still fails; stop when none do (local minimum) or budgets run out. Cost ~= sum
    of config sizes per pass, capped at decode_cap decodes total.
    """
    out = []
    spent = 0
    for cfg0 in configs:
        cur = list(cfg0)
        for _ in range(max_passes):
            if spent >= decode_cap or len(cur) <= 2:
                break
            cand = np.array([[c for c in cur if c != drop] for drop in cur], dtype=np.int64)
            syn = np.bitwise_xor.reduce(det[c2m[cand]], axis=1)
            tru = np.bitwise_xor.reduce(obs[c2m[cand]], axis=1)
            bad = np.any(dec.decode_batch(syn) != tru, axis=1)
            spent += len(cur)
            if not bad.any():
                break                                  # locally minimal failing set
            cur = [c for c in cur if c != cur[int(rng.choice(np.nonzero(bad)[0]))]]
        out.append(frozenset(cur))
    sizes0 = sorted(len(c) for c in configs)
    sizes1 = sorted(len(c) for c in out)
    print(f"[strip] {len(configs)} configs: weights {sizes0[0]}..{sizes0[-1]} -> "
          f"{sizes1[0]}..{sizes1[-1]} (min {sizes1[0]}; {spent} decodes)", flush=True)
    return out


def harvest_idle_seeds(circ, dec, rng, per_weight=15, shots_cap=60_000, batch=2_000):
    """Failing-config mechanism supports from fixed-weight sampling, guided by the
    cached idle spectrum (harvest only where f(w) was measurable)."""
    spec = json.loads((REPO_ROOT / "runs" / "framework" / "bb144" / "lpu_idle" /
                       "spectrum.json").read_text(encoding="utf-8"))
    byw = {int(w): int(f) for w, f in spec["failures_by_weight"].items()}
    tbyw = spec.get("trials_by_weight", {})
    lows = sorted(w for w, f in byw.items() if f >= 2)
    if not lows:
        raise SystemExit("idle spectrum has no measurable low-weight bins to harvest at")
    w_lo = lows[0]
    targets = sorted({w_lo, 2 * w_lo, 4 * w_lo, min(8 * w_lo, max(byw))})
    print(f"[harvest] lowest measurable f(w) at w={w_lo} "
          f"({byw[w_lo]}/{tbyw.get(str(w_lo), '?')}); harvest weights {targets}")
    probs, det, obs = _parse_dem(circ)
    c2m, _, _ = _expand(probs, None)
    N_exp = c2m.shape[0]
    configs = []                               # EXPANDED-column frozensets (for stripping)
    for w in targets:
        got, shots = 0, 0
        t0 = time.time()
        while got < per_weight and shots < shots_cap:
            idx = rng.integers(0, N_exp, size=(batch, w))
            if w > 1:                          # duplicate-row rejection (sample_batch style)
                while True:
                    s_ = np.sort(idx, axis=1)
                    bad_rows = (s_[:, 1:] == s_[:, :-1]).any(axis=1)
                    if not bad_rows.any():
                        break
                    idx[bad_rows] = rng.integers(0, N_exp, size=(int(bad_rows.sum()), w))
            syn = np.bitwise_xor.reduce(det[c2m[idx]], axis=1)
            tru = np.bitwise_xor.reduce(obs[c2m[idx]], axis=1)
            bad = np.any(dec.decode_batch(syn) != tru, axis=1)
            for row in idx[bad][: per_weight - got]:
                configs.append(frozenset(int(c) for c in row))
                got += 1
            shots += batch
        print(f"[harvest] w={w}: {got} failing configs in {shots} shots "
              f"({time.time()-t0:,.0f}s)", flush=True)
        if got == 0:
            print(f"[harvest] w={w}: nothing in budget — dropped")
    if not configs:
        raise SystemExit("harvest found no failing configs — cannot seed the ladder")
    # Stripping: descend the lightest harvested class toward locally minimal failing
    # sets — the cold rungs' seeds. Keep the un-stripped configs too (mid rungs).
    lightest = sorted(configs, key=len)[: 2 * per_weight]
    stripped = strip_configs(det, obs, c2m, dec, lightest, rng)
    all_cfgs = list({*configs, *stripped})
    supports = [frozenset(int(c2m[c]) for c in cfg) for cfg in all_cfgs]
    return supports, targets, min(len(c) for c in stripped)


def run_lpu_idle(which):
    from experiment_runner import load_config, build_circuit, make_decoder
    cfg = load_config(str(REPO_ROOT / "experiments" / "configs" / "gross_lpu_idle.yaml"))
    circ = build_circuit(cfg)
    if which == "camp":
        dec = make_decoder(cfg)                    # campaign relay: num_sets=20, nconv=5
        dcfg = dict(num_sets=cfg.relay_num_sets, stop_nconv=cfg.relay_stop_nconv,
                    gamma0=cfg.relay_gamma0, pre_iter=cfg.relay_pre_iter,
                    set_max_iter=cfg.relay_set_max_iter)
    else:
        dec = RelayBPDecoder(**GHW_CFG)            # ghw knobs, same from-circuit priors
        dcfg = {k: list(v) if isinstance(v, tuple) else v for k, v in GHW_CFG.items()}
    dec.setup(circ)
    rng = np.random.default_rng(17 if which == "camp" else 18)
    supports, targets, w_strip_min = harvest_idle_seeds(circ, dec, rng)
    gapw = sorted({int(t * 1.5) for t in targets[:-1]} | {targets[-1] * 2})
    t0 = time.time()
    temper, diag = replica_exchange_estimate(
        circ, dec, p_ref=cfg.p_ref, p_high=5e-3, p_low=1e-4, n_levels=16,
        n_walkers=8, local_steps=5, n_sweeps=120, burn_in=30, anchor_shots=3000,
        distance=None, mw_supports=supports, seed_p_grid=[2e-3, 8e-4],
        gap_weights=gapw, seed=4 if which == "camp" else 5,
        single_sector=False, verbose=True)
    save(f"lpu_idle_{which}", temper, diag,
         dict(code="bb144_lpu_idle", decoder=("campaign" if which == "camp" else "ghw"),
              decoder_cfg=dcfg, prior_convention="from p_ref circuit (campaign style)",
              harvest_weights=targets, n_seeds=len(supports),
              w_stripped_min=int(w_strip_min), p_ref=cfg.p_ref),
         time.time() - t0)


def run_smoke():
    """18-code full symmetric, tiny budgets, vs the cached tech3 ladder."""
    base = rmc.RESULTS
    t2 = json.loads((base / "tech2__full_symmetric.json").read_text(encoding="utf-8"))["result"]
    t3 = json.loads((base / "tech3__full_symmetric.json").read_text(encoding="utf-8"))["result"]
    circ = rmc.make_circuit("full symmetric", rmc.P_REF)
    calib = rmc.make_circuit("full symmetric", rmc.DECODER_P)
    dec = rmc.CalibratedRelayBP(calib, **rmc.DEC_CFG)
    t0 = time.time()
    # Full tech3 budgets (still seconds at 18-code scale): a tiny-budget ladder is
    # NOT a valid regression — under-equilibrated rungs bias every log-ratio low and
    # the shortfall compounds down-ladder (measured ~8x at p_low with sweeps=20).
    temper, diag = replica_exchange_estimate(
        circ, dec, p_ref=rmc.P_REF, p_high=0.015, p_low=1e-4, n_levels=16,
        n_walkers=8, local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000,
        distance=t2["D"], mw_supports=[frozenset(s) for s in t2["LD"]],
        seed=2, single_sector=False, verbose=False)
    sp = np.asarray(temper.p_ladder)[::-1]
    sP = np.asarray(temper.P_logical)[::-1]
    se = np.asarray(temper.P_logical_se)[::-1]
    ref_p = np.asarray(t3["sp"]); ref_P = np.asarray(t3["sP"]); ref_se = np.asarray(t3["sP_se"])
    print(f"smoke ({time.time()-t0:.0f}s): ladder vs cached tech3 (must agree within ~3 combined SE)")
    bad = 0
    for i in range(len(sp)):
        j = int(np.argmin(np.abs(np.log(ref_p) - np.log(sp[i]))))
        z = abs(np.log(max(sP[i], 1e-300)) - np.log(max(ref_P[j], 1e-300))) / \
            max(np.hypot(se[i] / max(sP[i], 1e-300), ref_se[j] / max(ref_P[j], 1e-300)), 1e-9)
        flag = "" if z < 3 else "  <-- DISAGREES"
        bad += z >= 3
        print(f"  p={sp[i]:.3e}: smoke {sP[i]:.3e}  cached {ref_P[j]:.3e}  z={z:.1f}{flag}")
    print(f"smoke verdict: {len(sp)-bad}/{len(sp)} points within 3 sigma "
          f"(swap {min(diag['swap_accept']):.2f}..{max(diag['swap_accept']):.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["72_full_ghw", "lpu_idle_camp", "lpu_idle_ghw"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run_smoke(); return
    if not a.target:
        ap.error("--target or --smoke")
    if (OUT_DIR / f"{a.target}.json").exists() and not a.force:
        raise SystemExit(f"{a.target}.json exists — use --force to redo")
    if a.target == "72_full_ghw":
        run_72_full_ghw()
    else:
        run_lpu_idle(a.target.split("_")[-1])


if __name__ == "__main__":
    main()
