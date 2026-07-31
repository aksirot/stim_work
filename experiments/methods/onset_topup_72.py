#!/usr/bin/env python
"""Onset top-up for the [[72,4,8]] failure spectra.

Resample fault weights 2..10 at a 3e6-shot cap (adaptive: stop early once TARGET failures are
seen) and ADD the samples into the cached spectrum bins, tightening the onset region from the
~6e-5 rule-of-three floor (50k shots) toward ~1e-6, without touching the mid/high weights.

Correctness:
  * Same circuit + same LOW-p-calibrated decoder as the campaign (run_error_model_comparison's
    builders + DEC): the isolated/ablated spectra use per-model priors from the model circuit at
    DECODER_P; the ASYMMETRIC (x5-ray) spectra use priors from the x5 circuit at DECODER_P
    (make_*_asym scales meas/meas_idle x5), i.e. the 5x-ray priors — verified before sampling.
  * Importance sampling at fixed weight is unbiased, so pooling (trials += , failures += ) an
    independent top-up run into the existing bins is a valid larger-sample estimate. A distinct
    seed (per spectrum+weight, offset by any already-added trials) keeps the draws independent.

One spectrum per invocation. Resumable per weight (a completed weight is recorded in the JSON's
`onset_topup` block and skipped on rerun); an interrupted weight redoes from scratch (nothing was
merged for it yet), so no double-counting.

    python experiments/methods/onset_topup_72.py --index 0        # SPECTRA[0]
    python experiments/methods/onset_topup_72.py tech1_72__meas_only
    python experiments/methods/onset_topup_72.py --list
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_error_model_comparison as R
from importance_sampling import _parse_dem, _expand, _sample_failures_at_weight

ONSET_WEIGHTS = list(range(2, 11))                              # w = 2..10
SHOTS_MAX = int(os.environ.get("ONSET_SHOTS_MAX", "3000000"))   # per-weight hard cap
TARGET = int(os.environ.get("ONSET_TARGET", "20"))             # stop a weight once TARGET failures seen
CHUNK = int(os.environ.get("ONSET_CHUNK", "50000"))            # sampling batch size
TOPUP_SEED = 104                                               # base seed, distinct from campaign seed=4
# ONSET_RESULTS_DIR overrides the cache dir (for tests); default = the campaign's run_dir.
RESULTS = pathlib.Path(os.environ["ONSET_RESULTS_DIR"]) if os.environ.get("ONSET_RESULTS_DIR") else R.RESULTS

# Stable array-index -> spectrum-name order (models, then ablations, then x5 asym).
SPECTRA = ([f"tech1_72__{R.slug(n)}" for n in R.MODELS]
           + [f"tech1_72_abl__{R.slug(n)}" for n in R.ABLATED]
           + ["asym__full_72"] + [f"asym__{R.slug(n)}_72" for n in R.ABLATED])

_SLUG2MODEL = {R.slug(n): n for n in R.MODELS}
_SLUG2ABL = {R.slug(n): n for n in R.ABLATED}


def build(name: str, device: bool = False):
    """(sampling circuit @ P_REF, calibration circuit @ DECODER_P) for a 72-code spectrum name.
    device=True: EMC_CALIB=device convention — one decoder per device (full symmetric for
    the symmetric family, full asym for the asym rays), matching the cached spectrum."""
    if name.startswith("tech1_72_abl__"):
        a = _SLUG2ABL[name[len("tech1_72_abl__"):]]
        calib = (R.make_circuit72("full symmetric", R.DECODER_P) if device
                 else R.make_ablated_circuit72(a, R.DECODER_P))
        return R.make_ablated_circuit72(a, R.P_REF), calib
    if name.startswith("tech1_72__"):
        m = _SLUG2MODEL[name[len("tech1_72__"):]]
        calib = (R.make_circuit72("full symmetric", R.DECODER_P) if device
                 else R.make_circuit72(m, R.DECODER_P))
        return R.make_circuit72(m, R.P_REF), calib
    if name.startswith("asym__") and name.endswith("_72"):
        x = name[len("asym__"):-len("_72")]
        if x == "full":
            return (R.make_full_asym(R.BB_72_4_8, R.ROUNDS72, R.P_REF),
                    R.make_full_asym(R.BB_72_4_8, R.ROUNDS72, R.DECODER_P))
        ch = R.ABLATED[_SLUG2ABL[x]]
        calib = (R.make_full_asym(R.BB_72_4_8, R.ROUNDS72, R.DECODER_P) if device
                 else R.make_abl_asym(R.BB_72_4_8, R.ROUNDS72, ch, R.DECODER_P))
        return R.make_abl_asym(R.BB_72_4_8, R.ROUNDS72, ch, R.P_REF), calib
    raise ValueError(f"not a 72-code spectrum name: {name!r}")


def topup(name: str) -> None:
    import json
    path = RESULTS / f"{name}.json"
    j = json.loads(path.read_text(encoding="utf-8"))
    dcfg = j["config"]["decoder"]
    assert dcfg.get("calibrated_at") == R.DECODER_P, \
        f"{name}: cached decoder calibrated_at={dcfg.get('calibrated_at')} != DECODER_P={R.DECODER_P}"
    spec = j["result"]["spectrum"]

    # calibration convention comes from the CACHED spectrum (merging must match the
    # decoder that produced the existing bins), not from the environment
    circ, calib = build(name, device=dcfg.get("calib") == "device")
    probs, det_mat, obs_mat = _parse_dem(circ)
    col_to_mech, q_base, _ = _expand(probs, None)
    n_expanded = int(col_to_mech.shape[0])
    # The pooled merge is only valid over the SAME expanded representation.
    assert n_expanded == int(spec["n_expanded"]) and abs(q_base - float(spec["q_base"])) < 1e-12, \
        (f"{name}: expansion mismatch — cached (N={spec['n_expanded']}, q={spec['q_base']}) vs "
         f"rebuilt (N={n_expanded}, q={q_base}); wrong circuit/p_ref?")

    dec = R.DEC(calib)
    dec.setup(circ)                          # CalibratedRelayBP: freezes priors from `calib` (DECODER_P)

    byw = {int(w): [int(t), int(f)] for w, t, f in
           zip(spec["weights"], spec["trials"], spec["failures"])}
    tu = spec.setdefault("onset_topup", {})  # {str(w): {added_trials, added_fails}} — done weights

    def save():
        ws = sorted(byw)
        spec["weights"] = ws
        spec["trials"] = [byw[w][0] for w in ws]
        spec["failures"] = [byw[w][1] for w in ws]
        path.write_text(json.dumps(j, indent=1), encoding="utf-8")

    fc = spec.setdefault("failure_configs", {})   # {str(w): [[mech,...], ...]} specimens
    FC_CAP = 200                                  # per weight, bounds file size

    def _sample_keep(w, T, rng):
        """Same sampling as importance_sampling._sample_failures_at_weight (same rng
        call pattern -> same distribution), but KEEPS the failing configurations'
        mechanism lists. Purely observational: counts are computed identically."""
        N_exp = col_to_mech.shape[0]
        syndromes = np.zeros((T, det_mat.shape[1]), dtype=bool)
        truths = np.zeros((T, obs_mat.shape[1]), dtype=bool)
        rows = []
        for t in range(T):
            mech_idxs = col_to_mech[rng.choice(N_exp, size=w, replace=False)]
            rows.append(mech_idxs)
            syndromes[t] = np.bitwise_xor.reduce(det_mat[mech_idxs], axis=0)
            truths[t] = np.bitwise_xor.reduce(obs_mat[mech_idxs], axis=0)
        fails = np.any(dec.decode_batch(syndromes) != truths, axis=1)
        cfgs = [sorted(int(m) for m in rows[i]) for i in np.nonzero(fails)[0]]
        return int(fails.sum()), cfgs

    for w in ONSET_WEIGHTS:
        if str(w) in tu:
            print(f"[{name}] w={w}: already topped up ({tu[str(w)]}); skip", flush=True)
            continue
        rng = np.random.default_rng([TOPUP_SEED, w, abs(hash(name)) % (1 << 31)])
        add_t = add_f = 0
        t0 = time.perf_counter()
        while add_f < TARGET and add_t < SHOTS_MAX:
            c = min(CHUNK, SHOTS_MAX - add_t)
            n_f, cfgs = _sample_keep(w, c, rng)
            add_f += n_f
            bucket = fc.setdefault(str(w), [])
            bucket.extend(cfgs[: max(0, FC_CAP - len(bucket))])
            add_t += c
        byw.setdefault(w, [0, 0])
        byw[w][0] += add_t
        byw[w][1] += add_f
        tu[str(w)] = {"added_trials": add_t, "added_fails": add_f,
                      "n_failure_configs": len(fc.get(str(w), []))}
        save()                               # atomic per weight: merge + mark done together
        f_hat = byw[w][1] / byw[w][0]
        print(f"[{name}] w={w}: +{add_t} shots +{add_f} fails -> bin {byw[w]} "
              f"(f={f_hat:.2e}, {len(fc.get(str(w), []))} specimens) "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    print(f"[{name}] onset top-up complete", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", help="spectrum task name (e.g. tech1_72__meas_only)")
    ap.add_argument("--index", type=int, help="SPECTRA[index] instead of a name (for SLURM arrays)")
    ap.add_argument("--list", action="store_true", help="print the indexed spectrum list and exit")
    args = ap.parse_args(argv)
    if args.list:
        for i, n in enumerate(SPECTRA):
            print(f"{i:2d}  {n}")
        return
    name = args.name or (SPECTRA[args.index] if args.index is not None else None)
    if name is None:
        ap.error("give a spectrum name, --index N, or --list")
    print(f"onset top-up: {name}  (w={ONSET_WEIGHTS}, cap={SHOTS_MAX}, target={TARGET}, "
          f"decoder calibrated_at={R.DECODER_P})", flush=True)
    topup(name)


if __name__ == "__main__":
    main()
