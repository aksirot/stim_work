"""f5-ansatz fits for every LPU-op failure spectrum, refitted consistently.

Each op's run directory carries a cached `ansatz_fit.json`, but they were produced at
different times with different sampling depth. This refits every available spectrum with
the same call and prints the parameters side by side, so the ops can be compared.

The f5 ansatz (arXiv:2511.15177 Eq. 10) has

    w0      onset weight — below it the ansatz is identically zero
    f0      failure fraction AT the onset, f(w0)
    gamma1  initial rise of f(w) above the onset
    gamma2  slower late rise
    wc      crossover weight between the two regimes

Reported alongside: the fit cost (residual; large = the ansatz does not describe these
bins), the measured onset (lightest weight with an observed failure) and, where they
disagree with w0, a warning — the ansatz is identically zero below its own w0, so a
measured failure beneath it is mass the ansatz structurally cannot represent.

    python experiments/methods/lpu_ansatz_fits.py
    python experiments/methods/lpu_ansatz_fits.py --refit      # ignore cached fits
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import numpy as np

from importance_sampling import FailureSpectrum, fit_failure_spectrum
from repo_paths import REPO_ROOT

SEARCH = [REPO_ROOT / "runs" / "framework" / "bb144",
          REPO_ROOT / "runs" / "cluster" / "framework" / "bb144" / "bb144"]


def load_spectrum(d):
    """FailureSpectrum from an LPU-op run directory (its own key names)."""
    j = json.loads((d / "spectrum.json").read_text(encoding="utf-8"))
    tw = j["trials_by_weight"]
    fw = j["failures_by_weight"]
    ws = sorted(int(w) for w in tw)
    return FailureSpectrum(
        weights=ws,
        trials=[int(tw[str(w)]) for w in ws],
        failures=[int(fw[str(w)]) for w in ws],
        n_expanded=int(j["n_expanded"]), q_base=float(j["q_base"]),
        p_ref=float(j["p_ref"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refit", action="store_true", help="ignore cached ansatz_fit.json")
    ap.add_argument("--K", type=int, default=12, help="logical qubits (gross code: 12)")
    a = ap.parse_args(argv)

    dirs = []
    for root in SEARCH:
        if root.is_dir():
            dirs += [pathlib.Path(p).parent for p in
                     glob.glob(str(root / "*" / "spectrum.json"))]
    dirs = sorted(set(dirs), key=lambda d: d.name)
    if not dirs:
        raise SystemExit("no LPU-op spectra found")

    print(f"{'operation':22s} {'w0':>6} {'f0':>11} {'gamma1':>8} {'gamma2':>8} "
          f"{'wc':>8} {'cost':>7} {'bins':>5} {'w_meas':>7}  source")
    rows = []
    for d in dirs:
        try:
            spec = load_spectrum(d)
        except Exception as e:                                   # noqa: BLE001
            print(f"{d.name:22s} spectrum unreadable: {e}")
            continue
        meas = [w for w, f in zip(spec.weights, spec.failures) if f > 0]
        w_meas = min(meas) if meas else None
        cached = d / "ansatz_fit.json"
        src = "cached"
        fit = None
        if cached.exists() and not a.refit:
            j = json.loads(cached.read_text(encoding="utf-8"))
            par, cost, npts = j.get("params", {}), j.get("cost"), j.get("n_points")
        else:
            src = "refit"
            try:
                fit = fit_failure_spectrum(spec, K=a.K, model="f5", w0=None, f0=None)
                par = {k: getattr(fit, k) for k in ("w0", "f0", "gamma1", "gamma2", "wc")
                       if hasattr(fit, k)}
                if not par and hasattr(fit, "params"):
                    par = dict(fit.params)
                cost = getattr(fit, "cost", None)
                npts = len(spec.weights)
            except Exception as e:                               # noqa: BLE001
                print(f"{d.name:22s} fit failed: {type(e).__name__}: {e}")
                continue
        g = lambda k: par.get(k, float("nan"))
        print(f"{d.name:22s} {g('w0'):6.1f} {g('f0'):11.3e} {g('gamma1'):8.2f} "
              f"{g('gamma2'):8.2f} {g('wc'):8.1f} "
              f"{(cost if cost is not None else float('nan')):7.1f} "
              f"{(npts or 0):5d} {(w_meas if w_meas is not None else -1):7d}  {src}")
        rows.append((d.name, par, cost, w_meas))

    print("\nw_meas = lightest weight with an OBSERVED failure.")
    bad = [(n, p.get("w0"), w) for n, p, _, w in rows
           if w is not None and p.get("w0") is not None and w < p["w0"] - 0.5]
    if bad:
        print("\nSUB-ANSATZ MASS — measured failures below the fitted w0 (the ansatz is")
        print("identically zero there, so this mass is outside the family it can express):")
        for n, w0, w in bad:
            print(f"   {n:22s} fitted w0={w0:.1f} but a failure was observed at w={w}")
    else:
        print("No operation has measured failures below its fitted w0.")


if __name__ == "__main__":
    main()
