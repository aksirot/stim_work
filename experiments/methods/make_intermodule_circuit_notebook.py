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

md(r"""## Timeline-svg, sliced to ONE bridge coupling (compact, like schedule_svg's star)

The inter-module link, readably. Seed = the two adapter ancillas of bridge k=0 (the
first Bell-check pair), grown `hops` along two-qubit gates to pull in the A/B bridge
qubits and the V_l vertices they hang from. Then, like `schedule_svg`:
* **noise stripped** (structural view — the per-layer DEPOLARIZE1 boxes otherwise bury
  the gates; set `with_noise=True` to see them),
* **rails pruned** to qubits that take part in ≥1 two-qubit gate inside the slice
  (drops the reset/measure-only bystanders),
* **rails renumbered 0..N** — the mapping back to real qubit indices is printed.""")

code(r'''NOISE = {"DEPOLARIZE1","DEPOLARIZE2","X_ERROR","Z_ERROR","Y_ERROR",
         "PAULI_CHANNEL_1","PAULI_CHANNEL_2"}
TWOQ = {"CX","CZ","CY"}

def neighbourhood(circ, seeds, hops=1):
    """Qubits reachable from `seeds` via <=hops two-qubit gates."""
    S = set(seeds)
    for _ in range(hops):
        add = set()
        for inst in circ.flattened():
            if inst.name in TWOQ:
                t = [x.value for x in inst.targets_copy()]
                for a, b in zip(t[::2], t[1::2]):
                    if a in S or b in S: add.update((a, b))
        S |= add
    return S

def compact_timeline(circ, seeds, hops=2, with_noise=False):
    """Readable timeline-svg of the gate structure on a small qubit set.
    Returns (svg_text, rail_map) with rail_map[new] = real qubit index."""
    keep = neighbourhood(circ, seeds, hops)
    # prune to qubits that participate in a 2q gate fully inside the slice
    active = set()
    for inst in circ.flattened():
        if inst.name in TWOQ:
            t = [x.value for x in inst.targets_copy()]
            for a, b in zip(t[::2], t[1::2]):
                if a in keep and b in keep: active.update((a, b))
    keep = active
    rail = {q: i for i, q in enumerate(sorted(keep))}
    out = stim.Circuit()
    for inst in circ.flattened():
        if inst.name in ("DETECTOR","OBSERVABLE_INCLUDE","SHIFT_COORDS","QUBIT_COORDS"):
            continue
        if inst.name == "TICK":
            out.append("TICK"); continue
        if not with_noise and inst.name in NOISE:
            continue
        tg = inst.targets_copy()
        if not all(t.is_qubit_target for t in tg):
            continue
        # stim merges same-name gates into ONE instruction with many targets
        # (e.g. one CX with 1728 pairs) — split per gate before filtering
        qs = [t.value for t in tg]
        k = 2 if inst.name in TWOQ | {"DEPOLARIZE2","PAULI_CHANNEL_2","SWAP"} else 1
        sel = [g for g in (qs[i:i+k] for i in range(0, len(qs), k))
               if all(q in keep for q in g)]
        if sel:
            out.append(inst.name, [rail[q] for g in sel for q in g],
                       inst.gate_args_copy())
    return str(out.diagram("timeline-svg")), {i: q for q, i in rail.items()}

BASE = 2 * tdg.N_TOTAL_QUBITS                    # adapter ancillas start here
bell0 = [BASE, BASE + 1]                         # bridge k=0: (anc_A, anc_B)
svg, rails = compact_timeline(circ, bell0, hops=2)
print(f"{len(rails)} rails.  rail -> real qubit:  " +
      "  ".join(f"q{i}={q}" for i, q in rails.items()))
print("  (A-module: <378   B-module: 378..755   adapter ancillas: 756..787)")
display(SVG(svg))''')

md(r"""### The wide view (optional)

All 32 adapter ancillas + 1 hop — every bridge coupling at once (~70 rails). Busy by
construction; useful only to confirm the 11 bridges + 10 U_B cross-checks are all
wired the same way. Uncomment to render.""")

code(r'''# adapter_anc = range(circ.num_qubits - tdg.N_ADAPTER_ANC, circ.num_qubits)
# svg_w, rails_w = compact_timeline(circ, adapter_anc, hops=1)
# print(f"{len(rails_w)} rails"); display(SVG(svg_w))''')

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
