"""Generate intermodule_circuit_view.ipynb — inspect the stim circuit the inter-module
DEMs are pulled from (analog of R.schedule_svg, for build_joint_x1x1_circuit).

The full circuit is 788 qubits x ~38 rounds — a timeline-svg of the whole thing is
unreadable, so (like schedule_svg's 7-qubit star) this slices to the bridge/adapter
neighbourhood: the inter-module coupling, which is the interesting part. Also shows
the circuit's per-round structure, a spatial timeslice, and the DEM summary.

NEEDS A STIM-CAPABLE KERNEL (the qec conda env, or the container) — it builds the
circuit, unlike the emc_report notebooks which only read cached JSON.
"""
import json
from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# The inter-module joint-measurement circuit, up close

The stim circuit behind the `inter_module` DEMs (X̄₁(A)⊗X̄₁(B), `build_joint_x1x1_circuit`).
Two gross modules (A at frame 0, B at +OFF) each run the X̄₁ half-LPU augmented with
11 bridge qubits, Bell-coupled by the adapter. Framing: |0⟩ init → noiseless MPP
reference → d_init noisy bare rounds → C merged rounds [A l-cycle, B l-cycle, adapter]
→ edge readout + return → d_init trailing bare → transversal readout.

Built here at a **small C/d_init** so the timeline is short — the per-round *structure*
is identical to production (C=10, d_init=12); only the number of repeats differs.""")

code(r'''import numpy as np
import stim
from IPython.display import SVG, display
import gross_code_lpu_tdg as tdg
from surface_code_sim import ErrorModel

# small window for a readable timeline; per-round structure == production
circ = tdg.build_joint_x1x1_circuit(ErrorModel(p_phys=1e-2, p_meas=1e-2),
                                    C=2, d_init=1, include_memory_observables=True,
                                    idle_noise=True)
print(f"qubits={circ.num_qubits}  measurements={circ.num_measurements}  "
      f"detectors={circ.num_detectors}  observables={circ.num_observables}  "
      f"TICKs={str(circ).count('TICK')}")
print(f"N_DATA={tdg.N_DATA}  N_TOTAL_QUBITS(OFF)={tdg.N_TOTAL_QUBITS}  "
      f"N_ADAPTER_ANC={tdg.N_ADAPTER_ANC}")''')

md(r"""## Per-round structure (op counts between TICKs)

One row per TICK segment — what each layer contains (gates, measurements, noise).""")

code(r'''from collections import Counter
seg, rows = Counter(), []
for inst in circ.flattened():
    if inst.name == "TICK":
        rows.append(dict(seg)); seg = Counter(); continue
    seg[inst.name] += len(inst.targets_copy()) // (2 if inst.name in
        ("CX","CZ","CY","DEPOLARIZE2") else 1)
rows.append(dict(seg))
keys = ["R","M","MPP","H","CX","CZ","CY","DEPOLARIZE1","DEPOLARIZE2","X_ERROR"]
print("seg " + " ".join(f"{k:>5}" for k in keys))
for i, s in enumerate(rows):
    if sum(s.values()):
        print(f"{i:3d} " + " ".join(f"{s.get(k,0):>5}" for k in keys))''')

md(r"""## Timeline-svg, sliced to the bridge / adapter neighbourhood

The inter-module coupling. Seed = the adapter ancilla qubits (the top `N_ADAPTER_ANC`
indices — the bridge Bell + U_B cross-check ancillas); we grow one hop along two-qubit
gates to pull in the bridge qubits and the V_l vertices they hang from, in BOTH
modules, then keep only gates fully inside that set. Rails are the real qubit indices
(A-module low, B-module ≈ +OFF, adapter ancillas highest).""")

code(r'''def neighbourhood(circ, seeds, hops=1):
    """Qubits reachable from `seeds` via <=hops two-qubit gates."""
    S = set(seeds)
    for _ in range(hops):
        add = set()
        for inst in circ.flattened():
            if inst.name in ("CX","CZ","CY"):
                t = [x.value for x in inst.targets_copy()]
                for a, b in zip(t[::2], t[1::2]):
                    if a in S or b in S: add.update((a, b))
        S |= add
    return S

def slice_to(circ, keep):
    """Sub-circuit of gates whose every qubit target is in `keep` (no detectors)."""
    out = stim.Circuit()
    for inst in circ.flattened():
        if inst.name in ("DETECTOR","OBSERVABLE_INCLUDE","SHIFT_COORDS","QUBIT_COORDS"):
            continue
        if inst.name == "TICK":
            out.append("TICK"); continue
        tg = inst.targets_copy()
        qs = [t.value for t in tg if t.is_qubit_target]
        if qs and all(q in keep for q in qs):
            out.append(inst.name, tg, inst.gate_args_copy())
    return out

adapter_anc = range(circ.num_qubits - tdg.N_ADAPTER_ANC, circ.num_qubits)
keep = neighbourhood(circ, adapter_anc, hops=1)
print(f"slice: {len(keep)} qubits (of {circ.num_qubits}): {sorted(keep)}")
sub = slice_to(circ, keep)
display(SVG(str(sub.diagram("timeline-svg"))))''')

md(r"""## Spatial timeslice of one merged round

The qubit layout with the gates active at a chosen TICK (spatial, not time-unrolled).
Adjust the tick index to scan through the merge.""")

code(r'''# a TICK inside the first merged round (after the d_init=1 bare round + framing)
display(SVG(str(circ.diagram("timeslice-svg", tick=6))))''')

md(r"""## The DEM these pull from

What Technique I / the decoders actually consume: the detector-error-model. Mechanism
count, and a few example errors with their detector + observable footprints.""")

code(r'''d = circ.detector_error_model(decompose_errors=False)
errs = [x for x in d if x.type == "error"]
print(f"DEM: {len(errs)} error mechanisms, {d.num_detectors} detectors, "
      f"{d.num_observables} observables")
print("\nexample mechanisms (prob | detectors | observables):")
for e in errs[:6]:
    ds = [t.val for t in e.targets_copy() if t.is_relative_detector_id()]
    os = [t.val for t in e.targets_copy() if t.is_logical_observable_id()]
    print(f"  p={e.args_copy()[0]:.2e}  dets={ds[:6]}{'...' if len(ds)>6 else ''}  obs={os}")''')

md(r"""## Reading it

* The **timeline slice** shows the adapter: bridge qubits from A and B Bell-coupled
  through the adapter ancillas, plus the V_l vertex checks they attach to — the
  physical inter-module link that makes the joint X̄₁⊗X̄₁ measurement.
* The **per-round table** distinguishes bare rounds (BB syndrome only) from merged
  rounds (A + B l-cycles + adapter), and shows where the idle DEPOLARIZE1 lands.
* This is the *small-window* build; production is C=10/d_init=12 (same per-round
  structure, more repeats). Swap `include_memory_observables`/`idle_noise` or the
  coupler rate to inspect variants.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "tour_de_gross" / "intermodule_circuit_view.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
