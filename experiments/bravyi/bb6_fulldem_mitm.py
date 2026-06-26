"""Exact full-DEM Table 2 via the optimized detector-pivot MITM (~70 min on 24 cores).

The full both-sector DEM has no closed-form enumeration, so run_technique_ii derives its Table-2
numbers by BP-OSD *search* (validated by search saturation). This computes the **bit-exact** counts
instead, with the now-tractable MITM, and writes distance_mitm.json next to the search distance.json
so the report can prefer/compare it (method="mitm_exact").

Must be run as a script (not imported): the MITM's process pool uses the Windows 'spawn' start
method, so the entry point is guarded by ``if __name__ == "__main__":``.
"""
import sys, json, time, pathlib

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from repo_paths import RUNS
_FULLDEM = RUNS / "bravyi" / "bb6" / "bb6_fulldem_curve"


def main():
    from bb_code_sim import BB_72_12_6, build_bb_circuit
    from surface_code_sim import ErrorModel
    from min_weight import (dem_check_action_matrices, exact_min_weight_logicals_mitm,
                            min_weight_fail_count, expanded_logical_count)
    from math import comb

    circuit = build_bb_circuit(BB_72_12_6, ErrorModel(p_phys=0.003, p_meas=0.003),
                               rounds=6, idle_noise=True)
    H, A, mult, probs = dem_check_action_matrices(circuit)        # sector=None -> full both-sector DEM
    det_coords = circuit.get_detector_coordinates()               # {det_id: (type, s, c)}, det_id == H row
    print(f"full DEM: H {H.shape}, A {A.shape} — running exact MITM (parallel) ...", flush=True)

    t0 = time.time()
    full = exact_min_weight_logicals_mitm(H, A, det_coords, l=6, m=6, weight=6, verbose=True)
    ld_comp = len(full)
    ld_exp = expanded_logical_count(full, mult)
    fails, n_exp = min_weight_fail_count(H, A, full, mult)
    half = 3
    f0 = fails / comb(n_exp, half)
    dt = time.time() - t0

    out = {"rounds": 6, "single_sector": False, "sector_type": None, "method": "mitm_exact",
           "distance": 6, "onset": half, "n_compressed": int(H.shape[1]), "n_expanded": int(n_exp),
           "n_min_logicals": int(ld_comp), "n_min_logicals_expanded": int(ld_exp),
           "fail_count": int(fails), "onset_fraction": float(f0), "mitm_seconds": round(dt, 1)}
    (_FULLDEM / "distance_mitm.json").write_text(json.dumps(out, indent=2))
    print(f"\nEXACT full-DEM Table 2 ({dt:.0f}s): D=6, w0=3, "
          f"|L(D)| comp={ld_comp} exp={ld_exp:.4g}, |F|={fails}, N={n_exp}, f0={f0:.6e}")
    print(f"wrote {_FULLDEM / 'distance_mitm.json'}")


if __name__ == "__main__":
    main()
