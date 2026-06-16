#!/usr/bin/env python3
"""Overlay our BB(6) ansatz LER(p) on the (approximately digitized) paper Fig-10 BB(6)-relay line.

arXiv:2511.15177 Fig. 10 (right panel) plots logical error rate vs physical error rate for the
distance-6/-12/-18 BB codes with the Relay decoder, fitting the FIVE-parameter ansatz (f5). This
script reads our in-progress IS checkpoint (same outdir as fit_partial.py), refits BOTH the f3
and f5 ansatz pinned by Technique II, and overlays them on points digitized by eye from the paper
figure's green BB(6)-relay curve.

⚠️  The paper points are an APPROXIMATE by-eye digitization off the published figure (~0.5 decade
uncertainty on a log axis), NOT the authors' numerical data (no public data repo was found). Use
the overlay to check we land on the same line, not for precise agreement.

Usage:
    python notebooks/bb_code/compare_vs_paper.py --outdir notebooks/bb_code/bb6_fig10_out_500
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

# Approx. digitization of the GREEN BB(6)-relay solid line from arXiv:2511.15177 Fig. 10 (right).
# Read by eye against the figure gridlines (x: 1e-5..1e-2, y: 1e0..1e-18). ~0.5 decade uncertainty.
PAPER_BB6_RELAY_P = np.array([1.0e-4, 3.0e-4, 1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3])
PAPER_BB6_RELAY_LER = np.array([3.0e-9, 3.0e-7, 3.0e-5, 1.0e-3, 2.0e-2, 4.0e-1])

# Approx. digitization of the GREEN BB(6)-relay failure spectrum f(w) from Fig. 10 (LEFT panel),
# log fault-weight axis. The green "BB(6) (bound)" star (Technique-II onset) sits near (w=3, ~3e-7)
# — consistent with our f0 (the right-panel LER overlay already pins that). ~0.5 decade uncertainty.
PAPER_BB6_SPEC_W = np.array([4.0, 6.0, 8.0, 12.0, 16.0, 22.0, 30.0, 45.0])
PAPER_BB6_SPEC_FW = np.array([3.0e-6, 1.0e-4, 2.0e-3, 4.0e-2, 2.0e-1, 6.0e-1, 9.0e-1, 9.9e-1])
PAPER_BB6_BOUND = (3.0, 3.0e-7)   # green "BB(6) (bound)" star (w0=D/2, f*(D/2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    ckpt = json.loads((args.outdir / "bb6.spectrum.json").read_text())
    cfg_d = json.loads((args.outdir / "config.json").read_text())
    dist = json.loads((args.outdir / "distance.json").read_text())

    spectrum = bb6._spectrum_from_checkpoint(ckpt)
    done = len(spectrum.weights)
    w0, f0 = float(dist["onset"]), float(dist["onset_fraction"])
    K = 12

    p = np.logspace(-5, -2, 60)                 # match the paper's x-range
    fits = {}
    for model in ("f3", "f5"):
        try:
            fit = fit_failure_spectrum(spectrum, K=K, model=model, w0=w0, f0=f0)
            fits[model] = (fit, logical_error_rate_from_ansatz(fit, list(p)))
            print(f"{model}: " + ", ".join(f"{k}={v:.3g}" for k, v in fit.params.items())
                  + f"  (cost={fit.cost:.3g}, n_points={fit.n_points})")
        except Exception as e:  # f5 may be under-constrained on partial data
            print(f"{model}: fit failed — {e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(PAPER_BB6_RELAY_P, PAPER_BB6_RELAY_LER, "k^--", ms=9, lw=1.5,
            label="paper Fig.10 BB(6)-relay (≈ digitized)")
    colors = {"f3": "crimson", "f5": "darkorange"}
    for model, (fit, P) in fits.items():
        ax.plot(p, P, "-", color=colors[model], lw=2,
                label=f"ours: Technique-I ansatz {model}")
    # our raw reweighted points (diagnostic; valid where they overlap the ansatz at higher p)
    raw = bb6.reweight_spectrum(spectrum, p)
    ax.plot(p, raw.P_logical, "o", color="steelblue", ms=3, alpha=0.5,
            label="ours: raw IS reweighted (partial)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.2e-2); ax.set_ylim(1e-18, 2.0)
    ax.set_xlabel("Physical error rate $p$"); ax.set_ylabel("Logical error rate $P(p)$")
    ax.set_title(f"BB(6) [[72,12,6]] Relay — ours vs paper Fig.10 ({done} weights, partial)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = args.outdir / "compare_vs_paper.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # ---- failure spectrum f(w) comparison (Fig. 10 LEFT panel) ----
    fig2, ax2 = plt.subplots(figsize=(8.5, 6))
    ax2.plot(PAPER_BB6_SPEC_W, PAPER_BB6_SPEC_FW, "k^--", ms=9, lw=1.5,
             label="paper Fig.10 BB(6)-relay $f(w)$ (≈ digitized)")
    ax2.plot([PAPER_BB6_BOUND[0]], [PAPER_BB6_BOUND[1]], "kX", ms=12,
             label="paper BB(6) bound (≈ digitized)")
    wq = np.asarray(spectrum.weights, dtype=float)
    T = np.asarray(spectrum.trials, dtype=float); Fc = np.asarray(spectrum.failures, dtype=float)
    fhat = Fc / T; se = np.sqrt(np.maximum(fhat * (1 - fhat), 1.0 / T) / T)
    msk = Fc > 0
    ax2.errorbar(wq[msk], fhat[msk], yerr=se[msk], fmt="o", color="steelblue", ms=4, capsize=2,
                 label=f"ours: sampled $f(w)$ ({int(msk.sum())} wts)")
    wgrid = np.arange(int(round(w0)), 64)
    for model, (fit, _P) in fits.items():
        ax2.plot(wgrid, fit.f(wgrid), "-", color=colors[model], lw=2, label=f"ours: ansatz {model}")
    ax2.plot([w0], [f0], "*", color="darkgreen", ms=16, zorder=5,
             label=fr"ours: Technique-II bound ($w_0$=3, $f_0$={f0:.1e})")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(1, 1e3); ax2.set_ylim(1e-9, 2.0)
    ax2.set_xlabel("Fault weight $w$"); ax2.set_ylabel("Failure spectrum $f(w)$")
    ax2.set_title(f"BB(6) failure spectrum — ours vs paper Fig.10 ({done} weights, partial)")
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    out2 = args.outdir / "compare_spectrum_vs_paper.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
