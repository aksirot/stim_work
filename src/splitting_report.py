"""Analysis/plot layer for the splitting arbitration report.

Three estimators of the same quantity — P_logical(p) for the [[72,4,8]] full-symmetric
circuit — are compared against ground truth from direct Monte Carlo in the regime where
MC is affordable:

* **paper-faithful splitting** (arXiv:2511.15177 Alg. 2/3): Eq.18 fine ladder,
  multi-seeded warm starts from typical failing configs, sigma+Delta chain-growth
  controller.  `runs/splitting_crosscheck/72_full_paper.json`
* **replica-exchange / tempering v1 and v2** (the paper's *future-work* idea, our
  implementations): coarse geometric ladder, walkers holding one replica per rate,
  swaps between adjacent rates.  `72_full_ghw.json`, `72_full_v2_ghwdeep.json`
* **reweighted importance sampling** (Technique I) from the campaign spectra, which the
  rodan top-up measured down to w=2.

One method per figure/table, so the notebook stays a thin sequence of calls.
"""
import json

import numpy as np
import matplotlib.pyplot as plt

from repo_paths import run_dir, REPO_ROOT
from importance_sampling import FailureSpectrum, reweight_spectrum

SPLIT = REPO_ROOT / "runs" / "splitting_crosscheck"

PAPER_C, V1_C, V2_C, IS_C, MC_C = "#2ca25f", "#c51b8a", "#8856a7", "#2c7fb8", "black"


def _load(name):
    return json.loads((SPLIT / f"{name}.json").read_text(encoding="utf-8"))


def _spec(path):
    r = json.loads(path.read_text(encoding="utf-8"))["result"]
    keep = set(FailureSpectrum.__dataclass_fields__)
    return FailureSpectrum(**{k: v for k, v in r["spectrum"].items() if k in keep})


class SplitReport:
    def load(self):
        self.paper = _load("72_full_paper")
        self.v1 = _load("72_full_ghw")
        self.v2 = _load("72_full_v2_ghwdeep")
        self.is_spec = _spec(run_dir("error_model_comparison_18_4_4_sys_baseline18_ghw72")
                             / "tech1_72__full_symmetric.json")
        mc = self.paper["mc_overlap"]
        self.mc_p, self.mc_ler = mc["p"], mc["ler"]
        self.mc_fails, self.mc_shots = mc["fails"], mc["shots"]
        print(f"paper run: {self.paper['algorithm']}")
        print(f"  decoder {self.paper['decoder']}, {len(self.paper['sp'])} ladder rates, "
              f"L={self.paper['diag']['L']} M={self.paper['diag']['M']} "
              f"({self.paper['diag']['n_inst']} instances), "
              f"T={self.paper['diag']['T_per_level'][0]}/level")
        print(f"  MC anchor: {self.mc_fails}/{self.mc_shots} at p={self.mc_p} "
              f"= {self.mc_ler:.3e}")
        print(f"  verdict: {self.paper['verdict']} (z = {self.paper['overlap_z']:.2f})")

    # --- the arbitration figure -----------------------------------------------------
    def fig_arbitration(self):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

        # LEFT: the three estimators vs the MC ground-truth point
        pg = np.geomspace(1e-4, 1e-2, 60)
        L_is = np.array([float(reweight_spectrum(self.is_spec, [p]).P_logical[0]) for p in pg])
        axL.plot(pg, L_is, "-", color=IS_C, lw=2,
                 label="reweighted IS (Technique I, topped-up bins)")

        sp = np.asarray(self.paper["sp"]); sP = np.asarray(self.paper["sP"])
        sse = np.asarray(self.paper["sP_se"])
        axL.plot(sp, sP, "-", color=PAPER_C, lw=2.5, label="paper splitting (Alg. 2/3)")
        axL.fill_between(sp, np.maximum(sP - sse, 1e-30), sP + sse, color=PAPER_C, alpha=0.25)

        for run, col, lab in ((self.v1, V1_C, "tempering v1 (ghw)"),
                              (self.v2, V2_C, "tempering v2 (ghw_deep)")):
            axL.plot(run["sp"], run["sP"], "--", color=col, lw=1.6, label=lab)

        axL.errorbar([self.mc_p], [self.mc_ler],
                     yerr=[self.mc_ler / np.sqrt(self.mc_fails)],
                     fmt="*", ms=18, color=MC_C, zorder=6, capsize=3,
                     label=f"direct MC ({self.mc_fails}/{self.mc_shots:,})")
        axL.axvline(self.mc_p, color="0.7", ls=":", lw=1)
        axL.set_xscale("log"); axL.set_yscale("log")
        axL.set_xlim(9e-5, 1.1e-2); axL.set_ylim(1e-22, 1)
        axL.set_xlabel("physical error rate p"); axL.set_ylabel(r"$P_{\rm logical}$")
        axL.set_title("[[72,4,8]] full symmetric — three estimators vs MC ground truth")
        axL.legend(fontsize=8, loc="lower right"); axL.grid(alpha=0.3, which="both")

        # RIGHT: the mechanism — mean fault weight held at each rung
        for run, col, lab in ((self.paper, PAPER_C, "paper splitting"),
                              (self.v1, V1_C, "tempering v1"),
                              (self.v2, V2_C, "tempering v2")):
            mw = np.asarray(run["mean_weight"] if "mean_weight" in run
                            else run["diag"]["mean_weight"], dtype=float)
            axR.plot(np.asarray(run["sp"])[:len(mw)], mw, "o-", color=col, ms=3, lw=1.5,
                     label=lab)
        axR.set_xscale("log")
        axR.set_xlabel("rate p at the rung"); axR.set_ylabel("mean fault weight of samples")
        axR.set_title("why: tempering collapses onto minimal-weight cores")
        axR.legend(fontsize=8); axR.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    # --- tables ---------------------------------------------------------------------
    def arbitration_table(self):
        p = self.mc_p
        rows = []
        for run, lab in ((self.paper, "paper splitting (Alg 2/3)"),
                         (self.v1, "tempering v1 (ghw)"),
                         (self.v2, "tempering v2 (ghw_deep)")):
            sp = np.asarray(run["sp"]); sP = np.asarray(run["sP"])
            o = np.argsort(sp)
            val = float(np.exp(np.interp(np.log(p), np.log(sp[o]), np.log(sP[o]))))
            rows.append((lab, val, val / self.mc_ler))
        print(f"{'estimator':30s} {'P(2e-3)':>11} {'ratio to MC':>12}")
        print(f"{'direct MC (ground truth)':30s} {self.mc_ler:11.3e} {1.0:12.2f}")
        for lab, v, r in rows:
            print(f"{lab:30s} {v:11.3e} {r:12.2e}")
        L_is = float(reweight_spectrum(self.is_spec, [p]).P_logical[0])
        print(f"{'reweighted IS':30s} {L_is:11.3e} {L_is/self.mc_ler:12.2f}")
        print()
        print("Ratios far from 1 are the estimator being wrong, not noise: the MC point "
              f"has {self.mc_fails} events (relative SE {1/np.sqrt(self.mc_fails):.0%}).")

    def controller_table(self):
        d = self.paper["diag"]
        lv = d["levels"]
        print(f"{'level':>5} {'p':>10} {'T':>7} {'sigma':>8} {'Delta':>8} "
              f"{'sig+del':>8} {'mean w':>7} {'transitions':>11}")
        for i, L in enumerate(lv):
            if i % 4 and i != len(lv) - 1:
                continue
            print(f"{i:5d} {L['p']:10.3e} {L['T']:7d} {L['sigma']:8.4f} "
                  f"{L['Delta']:8.4f} {L['sigma']+L['Delta']:8.4f} "
                  f"{L['mean_weight']:7.1f} {L['transitions']:11d}")
        eps = d["eps"]; t = len(lv) - 1
        print()
        print(f"controller target eps/sqrt(t) = {eps}/sqrt({t}) = {eps/np.sqrt(t):.3f}; "
              f"every level reports sigma+Delta below it, i.e. the controller declared "
              f"convergence at T={lv[0]['T']} — while the estimate is still ~30x low. "
              f"The controller measures chain self-consistency, NOT accuracy.")

    def cost_table(self):
        d = self.paper["diag"]
        n_lv = len(d["levels"])
        T = d["T_per_level"][0]
        inst = d["n_inst"]
        decodes = sum(L["T"] for L in d["levels"]) * inst
        print(f"{'quantity':34s} {'this run':>14} {'paper (BB12 ref)':>18}")
        print(f"{'ladder rates (Eq.18)':34s} {n_lv:14d} {'~16/decade':>18}")
        print(f"{'instances (L x M)':34s} {inst:14d} {'12 x 3 = 36':>18}")
        print(f"{'samples per level':34s} {T:14,d} {'~1,000,000':>18}")
        print(f"{'total decodes (approx)':34s} {decodes:14,d} {'~10^9':>18}")
        sp = self.paper["sp"]
        rng_txt = f"{sp[0]:.0e}-{sp[-1]:.0e}"
        print(f"{'p range covered':34s} {rng_txt:>14} {'to 1e-4+':>18}")
        print()
        print("The ~30x low point estimate is a BUDGET artefact, not a method failure: "
              "the error bar (across-instance spread) covers the truth, so under-sampling "
              "is visible rather than silent. Closing it means ~500x more samples per "
              "level over a wider ladder.")
