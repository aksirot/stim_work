#!/usr/bin/env python3
"""Fit the Technique-I ansatz + plot LER(p) from a PARTIAL (in-progress) bb6 IS checkpoint.

Reads an outdir's bb6.spectrum.json (whatever weights are done so far), distance.json
(Technique-II w0/f0), and config.json; reweights the partial spectrum, fits the pinned
ansatz, and writes partial_fig10.png + partial_ansatz.json — WITHOUT touching the files the
running sweep writes. Safe to re-run while the sweep is still going (the checkpoint is written
atomically, so a read always sees a complete file).

Usage:
    python notebooks/bb_code/fit_partial.py --outdir notebooks/bb_code/bb6_fig10_out_500
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

import bb6_fig10_sweep as bb6  # noqa: E402
from importance_sampling import fit_failure_spectrum, logical_error_rate_from_ansatz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    ckpt = json.loads((args.outdir / "bb6.spectrum.json").read_text())
    cfg_d = json.loads((args.outdir / "config.json").read_text())
    dist_path = args.outdir / "distance.json"
    dist = json.loads(dist_path.read_text()) if dist_path.exists() else None

    spectrum = bb6._spectrum_from_checkpoint(ckpt)
    done = len(spectrum.weights)
    nz = [(w, f) for w, f in zip(spectrum.weights, spectrum.failures) if f > 0]
    print(f"partial spectrum: {done} weights done "
          f"({spectrum.weights[0]}..{spectrum.weights[-1]}), "
          f"{len(nz)} with failures, shots/weight={spectrum.trials[0]}, "
          f"N_expanded={spectrum.n_expanded}")
    if len(nz) < 2:
        print("not enough failing weights yet to fit the ansatz — wait for a few more weights.")
        return

    p = np.logspace(np.log10(cfg_d["p_lo"]), np.log10(cfg_d["p_hi"]), int(cfg_d["n_p"]))
    raw = bb6.reweight_spectrum(spectrum, p)

    w0 = f0 = None
    if dist is not None:
        w0, f0 = float(dist["onset"]), float(dist["onset_fraction"])
        print(f"pinning ansatz from Technique II: w0={w0}, f0={f0:.3e}")
    K = 12  # BB(6) = [[72,12,6]] → 12 logical observables
    fit = fit_failure_spectrum(spectrum, K=K, model=cfg_d["ansatz_model"], w0=w0, f0=f0)
    P_ext = logical_error_rate_from_ansatz(fit, list(p))

    print("ansatz params: " + ", ".join(f"{k}={v:.4g}" for k, v in fit.params.items())
          + f"   (n_points={fit.n_points}, cost={fit.cost:.3g})")
    print("extrapolated LER(p):")
    for pp, PP in zip(p[::4], P_ext[::4]):
        print(f"  p={pp:.2e}   LER={PP:.3e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: failure spectrum f(w) vs fault weight (what the ansatz actually fits) ---
    wq = np.asarray(spectrum.weights, dtype=float)
    T = np.asarray(spectrum.trials, dtype=float)
    F = np.asarray(spectrum.failures, dtype=float)
    fhat = F / T
    se = np.sqrt(np.maximum(fhat * (1.0 - fhat), 1.0 / T) / T)   # binomial SE, floored
    m = F > 0
    axL.errorbar(wq[m], fhat[m], yerr=se[m], fmt="o", color="steelblue", ms=4, capsize=2,
                 label=f"sampled $f(w)=F/T$ ({int(m.sum())} weights)")
    wgrid = np.arange(int(round(fit.params["w0"])), int(wq.max()) + 30)
    axL.plot(wgrid, fit.f(wgrid), "-", color="crimson", lw=2,
             label=fr"ansatz $f(w)$ ({cfg_d['ansatz_model']}, $\gamma$={fit.params.get('gamma', float('nan')):.2f})")
    if dist is not None:
        axL.plot([w0], [f0], "*", color="darkgreen", ms=15, zorder=5,
                 label=fr"Technique-II onset ($w_0$={int(w0)}, $f_0$={f0:.1e})")
    axL.axhline(fit.a, color="grey", ls="--", lw=0.8, label=fr"saturation $a=1-2^{{-K}}$={fit.a:.4f}")
    axL.set_yscale("log")
    axL.set_xlabel("Fault weight $w$"); axL.set_ylabel("Failure fraction $f(w)$")
    axL.set_title("Failure spectrum")
    axL.legend(fontsize=8, loc="lower right"); axL.grid(True, which="both", alpha=0.3)

    # --- Right panel: reweighted LER(p) + ansatz extrapolation ---
    lo = np.maximum(raw.P_logical - raw.P_logical_se, 1e-300)
    axR.fill_between(p, lo, raw.P_logical + raw.P_logical_se, color="steelblue", alpha=0.2)
    axR.plot(p, raw.P_logical, "o", color="steelblue", ms=4, label="IS reweighted (partial)")
    axR.plot(p, P_ext, "-", color="crimson", lw=2,
             label=f"Technique I ansatz ({cfg_d['ansatz_model']})")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("Physical error rate $p$"); axR.set_ylabel("Logical error rate")
    axR.set_title("LER vs $p$")
    axR.legend(fontsize=8); axR.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"BB(6) [[72,12,6]] Fig. 10 — PARTIAL ({done} weights, sweep in progress)")
    fig.tight_layout()
    out_png = args.outdir / "partial_fig10.png"
    fig.savefig(out_png, dpi=150)

    json.dump(
        {"weights_done": done, "weights": list(spectrum.weights),
         "failures": list(spectrum.failures), "shots_per_weight": int(spectrum.trials[0]),
         "params": {k: float(v) for k, v in fit.params.items()},
         "p": p.tolist(), "ansatz_P": P_ext.tolist(), "raw_P": raw.P_logical.tolist()},
        open(args.outdir / "partial_ansatz.json", "w"), indent=2,
    )
    print(f"wrote {out_png}")
    print(f"wrote {args.outdir / 'partial_ansatz.json'}")


if __name__ == "__main__":
    main()
