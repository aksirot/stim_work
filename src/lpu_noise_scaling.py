"""Asymmetric device-like noise on the built inter-module LPU circuit.

Shared by the IS runner (experiment_runner: Config.noise_scale_factors) and the direct-MC
driver (experiments/methods/lpu_direct_mc.py) so both paths scale the SAME instructions
the same way -- the decoders are then built on the scaled circuit, i.e. priors matched
to the asymmetric model at the circuit's p.

Audited 2026-09-22 on build_joint_x1x1_circuit (interleaved idle depth 12): measurement
flips are X_ERROR immediately before an M everywhere; the measure/reset dead-time idle
exists in two layouts -- the BARE rounds use the standard per-sub-layer layout (a p-rate
DEPOLARIZE1 anchored on the M/R stage, bb_code_sim's `meas_idle` predicate), while the
MERGED LPU rounds charge idle once per round as a lump DEPOLARIZE1(1-(1-p)^k) over k of
the 12 interleaved timesteps, one of which is the dead time. Do NOT apply bb_code_sim's
predicates blindly to LPU circuits: the merged rounds have no meas-idle-anchored
instruction at all.
"""
from __future__ import annotations

import math
from typing import Dict

import stim

from bb_code_sim import NOISE_CHANNEL_PREDICATES, scale_noise_channels

# device-like asymmetric point used for the 18/72-code error-model comparison
M5: Dict[str, float] = {"meas": 5.0, "meas_idle": 5.0}


def _datalike(q: int) -> bool:
    """Data + edge(bridge) qubits of either module: the qubits that sit idle through the
    ancilla measurement dead time (check/adapter ancillas are being measured then)."""
    import gross_code_lpu_tdg as tdg
    off = tdg.N_TOTAL_QUBITS
    if q >= 2 * off:
        return False
    l = q % off
    return l < tdg.N_DATA or tdg.EDGE_QUBIT_BASE <= l < tdg.VERTEX_QUBIT_BASE


def scale_im_noise(circ: stim.Circuit, p: float, factors: Dict[str, float]) -> stim.Circuit:
    """Rescale noise channels of a built inter-module circuit whose base rate is ``p``.

    'meas'      -> every measurement flip (X_ERROR immediately before an M) x f.
    'meas_idle' -> the measure/reset dead-time idle x f: (a) bare rounds' M/R-anchored
                   DEPOLARIZE1(p) x f directly; (b) merged LPU rounds' per-round idle
                   lump on data-like qubits gets (f-1) extra timestep-equivalents,
                   P' = 1 - (1-P)(1-p)^(f-1). Ancillas being measured are left alone.
    Other keys -> bb_code_sim.scale_noise_channels predicates ('cz', 'prep', 'gate_idle', ...).

    Validated on the built circuit: meas mass x5.00, legacy meas-idle x5, lumps +4 layers,
    every other channel x1.00, DEM still valid (max prob ~0.016 at p=1e-3).
    """
    f_mi = factors.get("meas_idle")
    out = stim.Circuit()
    if f_mi:
        # (b) first, on the untouched circuit: lumps are the DEPOLARIZE1 with k >= 2 layers
        for inst in circ.flattened():
            if inst.name == "DEPOLARIZE1":
                P = inst.gate_args_copy()[0]
                k = math.log1p(-P) / math.log1p(-p) if 0 < P < 1 else 1.0
                if k > 1.5:
                    tg = [t.value for t in inst.targets_copy()]
                    dl = [q for q in tg if _datalike(q)]
                    ot = [q for q in tg if not _datalike(q)]
                    if ot:
                        out.append("DEPOLARIZE1", ot, P)
                    if dl:
                        out.append("DEPOLARIZE1", dl, 1.0 - (1.0 - P) * (1.0 - p) ** (f_mi - 1))
                    continue
            out.append(inst)
    else:
        out = circ
    # (a) + everything else via the positional predicates; the legacy meas_idle layer is
    # exactly a p-rate DEPOLARIZE1, so restrict that predicate to |P - p| small (the lumps
    # handled above must not be scaled twice).
    std: Dict = {k: v for k, v in factors.items() if k != "meas_idle"}
    if f_mi:
        base = NOISE_CHANNEL_PREDICATES["meas_idle"]
        std[lambda i, pr, nx: base(i, pr, nx) and abs(i.gate_args_copy()[0] - p) < 1e-12] = f_mi
    return scale_noise_channels(out, std) if std else out
