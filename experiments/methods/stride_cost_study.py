"""How coarse can the weight sweep stride before the reweighted LER moves?

Two things at once:

1. REGRESSION — the generalized `fill_spectrum` must reproduce the old hardcoded
   stride-2 behaviour bit-for-bit on every cached spectrum, so existing results are
   untouched.
2. MEASUREMENT — decimate a real DENSE spectrum to stride s, refill, reweight, and
   compare against the undecimated answer. That turns "how aggressive can we be" from a
   judgement call into a measured error budget, separately for a shoulder-preserving
   scheme (dense below w_dense, strided above) and a naive uniform stride.

    python experiments/methods/stride_cost_study.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from importance_sampling import FailureSpectrum, reweight_spectrum
from emc_report import fill_spectrum
from repo_paths import run_dir

P_STAR = 5e-4
P_HIGH = 5e-3          # where the tail's binomial mass actually matters


def old_fill(spec):
    """The pre-2026-08-03 hardcoded stride-2 rule, kept for the regression check."""
    W, T, F = list(spec.weights), list(spec.trials), list(spec.failures)
    w_out, t_out, f_out = [], [], []
    for i, (w, t, f) in enumerate(zip(W, T, F)):
        w_out.append(w); t_out.append(t); f_out.append(f)
        if i + 1 < len(W) and W[i + 1] == w + 2:
            w_out.append(w + 1); t_out.append(t + T[i + 1]); f_out.append(f + F[i + 1])
    return FailureSpectrum(weights=w_out, trials=t_out, failures=f_out,
                           n_expanded=spec.n_expanded, q_base=spec.q_base, p_ref=spec.p_ref)


def spec_of(path):
    r = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["result"]
    keep = set(FailureSpectrum.__dataclass_fields__)
    return FailureSpectrum(**{k: v for k, v in r["spectrum"].items() if k in keep})


def decimate(spec, stride, w_dense=None):
    """Keep every weight below w_dense, then every `stride`-th weight above it."""
    W, T, F = spec.weights, spec.trials, spec.failures
    keep = [i for i, w in enumerate(W)
            if (w_dense is not None and w < w_dense) or (w - W[0]) % stride == 0]
    if keep[-1] != len(W) - 1:
        keep.append(len(W) - 1)                      # always keep the last bin
    return FailureSpectrum(weights=[W[i] for i in keep], trials=[T[i] for i in keep],
                           failures=[F[i] for i in keep], n_expanded=spec.n_expanded,
                           q_base=spec.q_base, p_ref=spec.p_ref)


def L(spec, p):
    return float(reweight_spectrum(fill_spectrum(spec), [p]).P_logical[0])


def main():
    dirs = {
        "ghw (topped up)": run_dir("error_model_comparison_18_4_4_sys_baseline18_ghw72"),
        "baseline": run_dir("error_model_comparison_18_4_4"),
    }
    tasks = ["tech1_72__full_symmetric", "asym__full_72"]

    print("=" * 76)
    print("1. REGRESSION — generalized fill vs old stride-2 fill on cached spectra")
    bad = 0
    for label, d in dirs.items():
        for t in tasks:
            p = d / f"{t}.json"
            if not p.exists():
                continue
            s = spec_of(p)
            a, b = fill_spectrum(s), old_fill(s)
            same = (list(a.weights) == list(b.weights) and list(a.trials) == list(b.trials)
                    and list(a.failures) == list(b.failures))
            print(f"   {label:18s} {t:26s} {'IDENTICAL' if same else 'DIFFERS'}")
            bad += 0 if same else 1
    print(f"   -> {'no change to existing results' if not bad else str(bad) + ' MISMATCHES'}")

    print("=" * 76)
    print("2. MEASUREMENT — decimate a dense spectrum, refill, reweight")
    ref_path = dirs["ghw (topped up)"] / "tech1_72__full_symmetric.json"
    s = spec_of(ref_path)
    W = list(s.weights)
    print(f"   reference: {ref_path.name}, {len(W)} bins, w={W[0]}..{W[-1]}")
    for p in (P_STAR, P_HIGH):
        ref = L(s, p)
        print(f"\n   p = {p:.0e}   reference LER = {ref:.4e}")
        print(f"   {'scheme':34s} {'bins':>6} {'LER':>12} {'error':>9}")
        for w_dense in (None, 20, 40, 50):
            for stride in (2, 4, 8, 16, 32):
                d = decimate(s, stride, w_dense)
                v = L(d, p)
                tag = (f"uniform stride {stride}" if w_dense is None
                       else f"dense<{w_dense} then stride {stride}")
                print(f"   {tag:34s} {len(d.weights):6d} {v:12.4e} "
                      f"{(v/ref - 1)*100:+8.2f}%")
    print("=" * 76)
    print("Reading: the shoulder carries the LER at p*, the saturated tail carries mass")
    print("only at high p. A scheme is safe when BOTH rows stay within the statistical")
    print("error of the bins themselves (a few % here).")


if __name__ == "__main__":
    main()
