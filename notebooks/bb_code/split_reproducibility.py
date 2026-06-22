"""Reproducibility check for the replica-exchange (tempered) splitting estimate.

Runs `replica_exchange_estimate` N times with DIFFERENT random seeds but the SAME config as the
deep cross-check (ladder 6e-3 -> 1e-4, 48 levels, 8 walkers), then asks the honest-error-bar
question: does the run-to-run spread of P_logical match the per-run quoted +-SE?

  spread/quoted ~ 1  -> the quoted SE is honest
  spread/quoted >> 1 -> the SE underestimates the true variance (e.g. correlated walkers)

Writes bb6_fig10_curve/reproducibility.json. No brackets (this is about the tempered estimator).
"""
import sys, json, pathlib, time
import numpy as np
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from bb_code_sim import BB_72_12_6, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from min_weight import single_sector_dem, find_min_weight_logicals, build_circuit_translation_perms
from splitting import replica_exchange_estimate

P_REF, P_HIGH, P_LOW, NLEV = 0.003, 0.006, 1e-4, 48
SEEDS = [42, 7, 123]


def _dec():
    return RelayBPDecoder(gamma0=0.125, pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=6)


def main():
    circuit = build_bb_circuit(BB_72_12_6, ErrorModel(p_phys=P_REF, p_meas=P_REF), rounds=6, idle_noise=True)
    H, A, mult, probs, dc = single_sector_dem(circuit, detector_type=0)
    perms = build_circuit_translation_perms(None, H, det_coords=dc, verbose=False)
    print("min-weight seeds (systematic + symmetry) ...", flush=True); t0 = time.time()
    mw_supports = find_min_weight_logicals(None, 6, matrices=(H, A, probs), systematic=True,
                                           max_trials=0, symmetry_perms=perms, workers=20)
    print(f"  {len(mw_supports)} seeds ({time.time()-t0:.0f}s)", flush=True)

    runs = []
    for sd in SEEDS:
        print(f"replica-exchange seed={sd} ...", flush=True); t0 = time.time()
        temper, _ = replica_exchange_estimate(
            circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
            n_walkers=8, local_steps=8, n_sweeps=300, burn_in=100, anchor_shots=3000,
            distance=6, seed=sd, single_sector=True, sector=0, mw_supports=mw_supports, verbose=False)
        print(f"  seed={sd} done ({time.time()-t0:.0f}s)", flush=True)
        runs.append({"seed": sd, "p_ladder": np.asarray(temper.p_ladder).tolist(),
                     "P_logical": np.asarray(temper.P_logical).tolist(),
                     "P_logical_se": np.asarray(temper.P_logical_se).tolist()})

    P = np.array([r["P_logical"] for r in runs])        # (n_runs, L+1)
    SE = np.array([r["P_logical_se"] for r in runs])
    pl = np.array(runs[0]["p_ladder"])
    mean = P.mean(0)
    rel_spread = P.std(0, ddof=1) / np.maximum(mean, 1e-300)      # run-to-run scatter (relative)
    quoted_rel = (SE / np.maximum(P, 1e-300)).mean(0)            # mean per-run quoted relative SE
    print(f"\n  {len(SEEDS)} runs (seeds {SEEDS}); per-rung run-to-run spread vs quoted SE:")
    print("  p           mean P       run-spread   quoted-SE   spread/quoted")
    for k in range(0, len(pl), 4):
        print(f"  {pl[k]:.2e}   {mean[k]:.3e}    {rel_spread[k]:7.1%}     {quoted_rel[k]:7.1%}     "
              f"{rel_spread[k]/max(quoted_rel[k],1e-12):.2f}")
    # headline: median over rungs of spread/quoted, and the bottom rung
    ratio = rel_spread / np.maximum(quoted_rel, 1e-12)
    print(f"\n  median spread/quoted over rungs = {np.median(ratio):.2f}  "
          f"(~1 => honest SE; >>1 => SE too small)")
    print(f"  bottom rung p={pl[-1]:.1e}: P = {mean[-1]:.3e}, run-spread {rel_spread[-1]:.1%}, "
          f"quoted {quoted_rel[-1]:.1%}")
    out = {"seeds": SEEDS, "p_ladder": pl.tolist(), "P_logical_runs": P.tolist(),
           "P_logical_se_runs": SE.tolist(), "mean": mean.tolist(),
           "rel_spread": rel_spread.tolist(), "quoted_rel_se": quoted_rel.tolist(),
           "config": {"p_high": P_HIGH, "p_low": P_LOW, "n_levels": NLEV, "n_walkers": 8,
                      "local_steps": 8, "n_sweeps": 300, "burn_in": 100, "num_sets": 100}}
    (_HERE / "bb6_fig10_curve" / "reproducibility.json").write_text(json.dumps(out, indent=2))
    print("\nwrote bb6_fig10_curve/reproducibility.json")


if __name__ == "__main__":
    main()
