"""PortfolioRelay prototype: heterogeneous Relay-BP ensemble with cross-config
maximum-likelihood selection.

Relay-BP already selects the best valid solution among randomized legs of ONE config;
this generalizes the selection ACROSS hyperparameter families. Motivation (measured in
this repo): gate-idle-class failures yield only to deep-pre/low-gamma0 knobs, the
stubborn mixed-prior traps only to wide-interval legs, and the wide interval's own
pathology ([[18,4,4]] meas_idle single, w=3 inflation) is confined to cases the stock
config solves trivially -- so best-of-portfolio-by-prior should take the union of fixes
while each member's pathologies are outvoted.

Selection rule per shot: among members whose correction REPRODUCES the syndrome
(validity checked against the DEM), pick the correction with the highest prior
log-likelihood sum(log(p_i/(1-p_i)) over flipped mechanisms); fall back to the first
member's observable prediction if none validate.

Evaluation battery (all data on disk from the tuning arc):
    --part quick     71-failure 72-code harvest + 18-code regression singles + 18 w=3
    --part midweight paired 72-code mid-weight bins vs both members (the nc1 killer)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS"):
    os.environ.pop(_k, None)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_error_model_comparison as rmc
from subonset_relay_sweep import sample_batch
from importance_sampling import _parse_dem, _expand

GHW_CFG = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"])


class PortfolioRelay:
    """Ensemble of set-up Relay decoders over the SAME circuit, ML-selected.

    Members must share the circuit's DEM (identical mechanism indexing). Scoring priors
    are taken from the calibration circuit (the same convention the members decode
    with). decode_batch matches the standard Decoder interface; decode_batch_detail
    additionally reports which member won each shot and validity counts.
    """

    def __init__(self, members: dict, det_mat: np.ndarray, obs_mat: np.ndarray,
                 priors: np.ndarray):
        self.members = members
        self.det = det_mat.astype(np.uint8)              # (M mechanisms, D detectors)
        self.obs = obs_mat.astype(np.uint8)
        p = np.clip(np.asarray(priors, float), 1e-12, 1 - 1e-12)
        self.logit = np.log(p / (1.0 - p))               # per-mech log prior odds (<0)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        return self.decode_batch_detail(dets)[0]

    def decode_batch_detail(self, dets: np.ndarray):
        dets8 = dets.astype(np.uint8)
        B = dets8.shape[0]
        best_score = np.full(B, -np.inf)
        best_obs = None
        winner = np.full(B, -1, dtype=np.int32)
        any_valid = np.zeros(B, dtype=bool)
        fallback_obs = None
        for mi, (name, dec) in enumerate(self.members.items()):
            corr = np.asarray(dec.decode_corrections_batch(dets8), dtype=np.uint8)
            syn = (corr.astype(np.uint32) @ self.det.astype(np.uint32)) % 2
            valid = np.all(syn == dets8, axis=1)
            score = corr.astype(np.float64) @ self.logit
            obs_pred = (corr.astype(np.uint32) @ self.obs.astype(np.uint32)) % 2
            if fallback_obs is None:
                fallback_obs = obs_pred
            take = valid & (score > best_score)
            if best_obs is None:
                best_obs = obs_pred.copy()
            best_obs[take] = obs_pred[take]
            best_score[take] = score[take]
            winner[take] = mi
            any_valid |= valid
        best_obs[~any_valid] = fallback_obs[~any_valid]
        return best_obs.astype(np.uint8), winner, any_valid


def make_portfolio(circ_ref, circ_calib, is72):
    base = rmc.CalibratedRelayBP(circ_calib, **rmc.DEC_CFG); base.setup(circ_ref)
    ghw = rmc.CalibratedRelayBP(circ_calib, **GHW_CFG); ghw.setup(circ_ref)
    probs, det, obs = _parse_dem(circ_ref)
    calib_probs, _, _ = _parse_dem(circ_calib)           # scoring priors = calib convention
    pf = PortfolioRelay({"baseline": base, "ghw": ghw}, det, obs, calib_probs)
    return pf, base, ghw


def part_quick():
    j = json.loads(open("runs/subonset_tune_72/hunt_and_bench_w3.json",
                        encoding="utf-8").read())
    print("=== 1. the 71-failure [[72,4,8]] harvest (sub-onset) ===")
    tot = {"portfolio": 0, "n": 0}
    windetail = {0: 0, 1: 0}
    for model in j["harvest"]:
        h = j["harvest"][model]
        if h["fails"] == 0:
            continue
        syn = np.array(h["syndromes"], dtype=bool)
        tru = np.array(h["truths"], dtype=bool)
        pf, base, ghw = make_portfolio(rmc.make_circuit72(model, rmc.P_REF),
                                       rmc.make_circuit72(model, rmc.DECODER_P), True)
        pred, winner, valid = pf.decode_batch_detail(syn)
        still = np.any(pred != tru, axis=1)
        for w in winner:
            if w in windetail: windetail[w] += 1
        tot["portfolio"] += int((~still).sum()); tot["n"] += h["fails"]
        print(f"  {model:16s}: portfolio fixes {int((~still).sum()):3d}/{h['fails']:3d}  "
              f"(all-valid {int(valid.sum())}/{len(valid)})", flush=True)
    print(f"  TOTAL: {tot['portfolio']}/{tot['n']}   winner counts baseline/ghw: "
          f"{windetail[0]}/{windetail[1]}")

    print("\n=== 2. [[18,4,4]] regression singles (the wide-interval pathology) ===")
    for model in ("meas idle", "full symmetric", "gate idle"):
        circ = rmc.make_circuit(model, rmc.P_REF)
        pf, base, ghw = make_portfolio(circ, rmc.make_circuit(model, rmc.DECODER_P), False)
        probs, det, obs = _parse_dem(circ)
        for name, d in (("baseline", base), ("ghw", ghw), ("PORTFOLIO", pf)):
            bad = np.any(d.decode_batch(det) != obs, axis=1)
            print(f"  {model:16s} {name:10s}: {int(bad.sum())}/{det.shape[0]} singles fail",
                  flush=True)

    print("\n=== 3. [[18,4,4]] full symmetric w=3 paired (the +38% inflation) ===")
    circ = rmc.make_circuit("full symmetric", rmc.P_REF)
    pf, base, ghw = make_portfolio(circ, rmc.make_circuit("full symmetric", rmc.DECODER_P), False)
    probs, det, obs = _parse_dem(circ)
    c2m, _, _ = _expand(probs, None)
    rng = np.random.default_rng(61)
    _, syn, tru = sample_batch(rng, c2m, det, obs, 3, 20_000)
    for name, d in (("baseline", base), ("ghw", ghw), ("PORTFOLIO", pf)):
        t0 = time.time()
        bad = np.any(d.decode_batch(syn) != tru, axis=1)
        print(f"  {name:10s}: f(3) = {int(bad.sum()):4d}/20000 = {bad.mean():.4f}  "
              f"({20_000/(time.time()-t0):,.0f} dec/s)", flush=True)


def part_midweight():
    print("=== 4. [[72,4,8]] mid-weight paired (where nc1 died) ===")
    circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
    pf, base, ghw = make_portfolio(circ, rmc.make_circuit72("full symmetric", rmc.DECODER_P), True)
    probs, det, obs = _parse_dem(circ)
    c2m, _, _ = _expand(probs, None)
    rng = np.random.default_rng(71)
    for w, T in [(8, 15_000), (10, 15_000), (12, 10_000)]:
        f = {"baseline": 0, "ghw": 0, "PORTFOLIO": 0}
        s = {k: 0.0 for k in f}
        done = 0
        while done < T:
            B = min(5_000, T - done)
            _, syn, tru = sample_batch(rng, c2m, det, obs, w, B)
            for name, d in (("baseline", base), ("ghw", ghw), ("PORTFOLIO", pf)):
                t0 = time.time()
                bad = np.any(d.decode_batch(syn) != tru, axis=1)
                s[name] += time.time() - t0
                f[name] += int(bad.sum())
            done += B
        print(f"  w={w}: " + "   ".join(f"{k} {f[k]:4d} ({T/s[k]:6,.0f}/s)" for k in f),
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["quick", "midweight"], required=True)
    a = ap.parse_args()
    (part_quick if a.part == "quick" else part_midweight)()


if __name__ == "__main__":
    main()
