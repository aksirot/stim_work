"""Relay-BP hyperparameter sweep on the sub-onset weight bins.

Question: can Relay-BP *tuning* remove the sub-onset decoder floor — the f(w) > 0 bins at
w < w0 = ceil(D/2) where a perfect min-weight decoder is provably exact (any explanation
no heavier than a sub-onset fault sits in the correct coset), so every failure there is a
decoder artifact (BP misconvergence or a prior-landscape misjudgment)?

Design: sample each weight bin ONCE, then decode the identical fault sets with every
hyperparameter config — a paired comparison, so config differences resolve at the
per-disagreement level (McNemar counts vs the baseline) instead of needing two
independent ~1/f(w) samples. With sub-onset rates around 1e-6, plan ~1e7 trials per bin
across shards; each shard writes one JSON (sum `trials`/`fails` and the McNemar counts
across shards to aggregate). Failing fault sets (mechanism indices) are kept per config
(capped) for anatomy afterwards.

Everything (circuits, calibration convention, baseline config) is imported from
run_error_model_comparison.py so the sweep tests exactly the report's decoder.

Usage (single shard, 72-code full mix, sub-onset bins):
    python subonset_relay_sweep.py --code 72 --shots 1000000 --seed 0 \\
        --out runs/subonset_relay_sweep/full72_s0.json

Cluster array (shard by seed; aggregate by summing counts):
    python subonset_relay_sweep.py --code 72 --shots 2000000 --seed $SLURM_ARRAY_TASK_ID \\
        --out runs/subonset_relay_sweep/full72_s$SLURM_ARRAY_TASK_ID.json

Config subset / listing:
    python subonset_relay_sweep.py --list-configs
    python subonset_relay_sweep.py --code 18 --configs baseline sets600_paper pre320
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import run_error_model_comparison as rmc
from importance_sampling import _parse_dem, _expand
from repo_paths import REPO_ROOT

# ---------------------------------------------------------------------------
# Hyperparameter grid. baseline = the report's DEC_CFG; each variant moves ONE knob
# (except the named composites) so effects attribute cleanly. Prior findings to test
# against at scale: pre_iter mattered more than num_sets for the BP symmetry traps;
# num_sets=20/stop_nconv=5 killed the symmetric w=1 fails; sets=100 added nothing on
# the stubborn x5-asym singles (pre prior-calibration).
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline":      dict(rmc.DEC_CFG),
    "sets5":         dict(rmc.DEC_CFG, num_sets=5),
    "sets60":        dict(rmc.DEC_CFG, num_sets=60),
    "sets200":       dict(rmc.DEC_CFG, num_sets=200),
    "sets600_paper": dict(rmc.DEC_CFG, num_sets=600),
    "pre160":        dict(rmc.DEC_CFG, pre_iter=160),
    "pre320":        dict(rmc.DEC_CFG, pre_iter=320),
    "iter120":       dict(rmc.DEC_CFG, set_max_iter=120),
    "nconv1":        dict(rmc.DEC_CFG, stop_nconv=1),
    "nconv9":        dict(rmc.DEC_CFG, stop_nconv=9),
    "gamma_lo":      dict(rmc.DEC_CFG, gamma0=0.0625),
    "gamma_hi":      dict(rmc.DEC_CFG, gamma0=0.25),
    "wide_interval": dict(rmc.DEC_CFG, gamma_dist_interval=(-0.5, 1.0)),
    "heavy":         dict(rmc.DEC_CFG, num_sets=200, pre_iter=320, set_max_iter=120),
}

DEFAULT_WEIGHTS = {"18": [1, 2], "72": [1, 2, 3, 4]}   # sub-onset + onset (w0: 18->2, 72->4)


def build_circuits(code, model, asym, p_sample, p_calib):
    """(sampling circuit at p_sample, calibration circuit at p_calib) via the runner's builders."""
    if asym:
        cp, rr = (rmc.P, rmc.ROUNDS) if code == "18" else (rmc.BB_72_4_8, rmc.ROUNDS72)
        if model != "full symmetric":
            raise SystemExit("--asym supports only the full mix (matching the report's asym tasks)")
        return rmc.make_full_asym(cp, rr, p_sample), rmc.make_full_asym(cp, rr, p_calib)
    make = rmc.make_circuit if code == "18" else rmc.make_circuit72
    return make(model, p_sample), make(model, p_calib)


def sample_batch(rng, col_to_mech, det_mat, obs_mat, w, B):
    """B uniform weight-w fault sets in the expanded representation (without replacement).

    Sampled with replacement + rejection of rows with duplicate EXPANDED columns (w is
    tiny vs N_exp, so rejections are ~w^2/N_exp rare). Distinct expanded columns of the
    same mechanism are kept — they XOR to zero, identical to the dense expansion
    (matches importance_sampling._sample_failures_at_weight).
    """
    N_exp = col_to_mech.shape[0]
    idx = rng.integers(0, N_exp, size=(B, w))
    if w > 1:
        while True:
            s = np.sort(idx, axis=1)
            bad = (s[:, 1:] == s[:, :-1]).any(axis=1)
            if not bad.any():
                break
            idx[bad] = rng.integers(0, N_exp, size=(int(bad.sum()), w))
    mechs = col_to_mech[idx]                                   # (B, w) mechanism indices
    syndromes = np.bitwise_xor.reduce(det_mat[mechs], axis=1)  # (B, n_det) bool
    truths = np.bitwise_xor.reduce(obs_mat[mechs], axis=1)     # (B, K) bool
    return mechs, syndromes, truths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", choices=["18", "72"], default="72")
    ap.add_argument("--model", default="full symmetric", choices=list(rmc.MODELS))
    ap.add_argument("--asym", action="store_true", help="x5 meas/meas-idle ray (full mix only)")
    ap.add_argument("--weights", type=int, nargs="+", default=None,
                    help=f"fault weights to sample (default {DEFAULT_WEIGHTS})")
    ap.add_argument("--shots", type=int, default=100_000, help="trials per weight (this shard)")
    ap.add_argument("--batch", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0, help="shard seed (array-job index)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    help="config names to run (baseline is always prepended)")
    ap.add_argument("--decoder-p", type=float, default=rmc.DECODER_P,
                    help="calibration prior for ALL configs (the report's device convention)")
    ap.add_argument("--keep-fails", type=int, default=100,
                    help="failing fault sets kept per (config, weight) for anatomy")
    ap.add_argument("--out", default=None, help="output JSON (default under runs/subonset_relay_sweep/)")
    ap.add_argument("--list-configs", action="store_true")
    args = ap.parse_args()

    if args.list_configs:
        for k, v in CONFIGS.items():
            print(f"{k:14s} {v}")
        return

    names = list(dict.fromkeys(["baseline"] + [c for c in args.configs]))
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; --list-configs shows the grid")
    weights = args.weights or DEFAULT_WEIGHTS[args.code]

    circ, calib = build_circuits(args.code, args.model, args.asym, rmc.P_REF, args.decoder_p)
    probs, det_mat, obs_mat = _parse_dem(circ)
    col_to_mech, q_base, _ = _expand(probs, None)
    print(f"code={args.code} model={args.model!r} asym={args.asym}  "
          f"n_mech={probs.shape[0]} N_exp={col_to_mech.shape[0]} n_det={det_mat.shape[1]}")
    print(f"weights={weights} shots/weight={args.shots} seed={args.seed} "
          f"decoder_p={args.decoder_p}  configs={names}")

    decoders = {}
    for name in names:
        d = rmc.CalibratedRelayBP(calib, **CONFIGS[name])
        d.setup(circ)                      # freezes priors from the calib circuit
        decoders[name] = d

    rng = np.random.default_rng(args.seed)
    results = {w: {n: dict(trials=0, fails=0, b01=0, b10=0, fail_mechs=[]) for n in names}
               for w in weights}
    t0 = time.time()
    for w in weights:
        done = 0
        while done < args.shots:
            B = min(args.batch, args.shots - done)
            mechs, syndromes, truths = sample_batch(rng, col_to_mech, det_mat, obs_mat, w, B)
            base_bad = None
            for name in names:
                pred = decoders[name].decode_batch(syndromes)
                bad = np.any(pred != truths, axis=1)
                r = results[w][name]
                r["trials"] += B
                r["fails"] += int(bad.sum())
                if name == "baseline":
                    base_bad = bad
                else:                       # McNemar counts vs baseline on the SAME faults
                    r["b01"] += int((base_bad & ~bad).sum())   # baseline fails, variant ok
                    r["b10"] += int((~base_bad & bad).sum())   # baseline ok, variant fails
                keep = args.keep_fails - len(r["fail_mechs"])
                if keep > 0 and bad.any():
                    r["fail_mechs"] += [sorted(map(int, row)) for row in mechs[bad][:keep]]
            done += B
        el = time.time() - t0
        print(f"w={w}: done {args.shots} trials/config in {el:,.0f}s cumulative")
        for name in names:
            r = results[w][name]
            note = "" if name == "baseline" else f"   fixes(b01)={r['b01']} breaks(b10)={r['b10']}"
            print(f"   {name:14s} f({w}) = {r['fails']:6d}/{r['trials']}"
                  f" = {r['fails']/max(r['trials'],1):.3e}{note}")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "runs" / "subonset_relay_sweep" /
        f"{args.code}_{'asym' if args.asym else 'sym'}_{args.model.replace(' ', '_')}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        args=vars(args), configs={n: CONFIGS[n] for n in names}, p_ref=rmc.P_REF,
        q_base=q_base, n_expanded=int(col_to_mech.shape[0]),
        results={str(w): results[w] for w in weights},
        elapsed_s=time.time() - t0,
    ), indent=1, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
