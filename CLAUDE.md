# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
pip install ldpc  # required for BB code BP-OSD decoder
```

Key packages: `stim` (quantum circuit simulation), `pymatching` (MWPM decoder), `ldpc` (BP-OSD decoder for BB codes).

## Running simulations

Simulations are run interactively via Jupyter notebooks or by importing the simulator modules directly:

```python
from surface_code_sim import SurfaceCodeSimulator, ErrorModel, CodeType, threshold_sweep
from bb_code_sim import BBCodeSimulator, BB_72_12_6, BB_144_12_12, BPOSDDecoder
```

Notebooks: `surface_code_exploration.ipynb`, `surface_code_explained.ipynb`, `bb_code_144_12_12.ipynb`.

## Architecture

This codebase simulates quantum error correction codes to estimate logical error rates.

**`surface_code_sim.py`** — Surface code (topological CSS code):
- `ErrorModel`: noise parameters (`p_phys` for gate depolarizing, `p_meas` for measurement bit-flip)
- `SurfaceCodeSimulator`: wraps `stim.Circuit.generated()` for rotated/unrotated surface codes; uses `PyMatchingDecoder` (MWPM) by default
- `threshold_sweep()`: sweeps distances × error rates to find the threshold crossing
- Decoder interface: `Decoder` base class with `setup(circuit)` + `decode_batch(events)`

**`bb_code_sim.py`** — Bivariate Bicycle (BB) codes (based on arXiv:2308.07915):
- `BBCodeParams`: defines a BB code via polynomial exponents `(l, m, a_exps, b_exps)`; pre-defined instances: `BB_72_12_6`, `BB_144_12_12`
- `build_parity_checks()`: constructs H_X, H_Z from the circulant polynomial structure
- `find_logical_ops()`: GF(2) linear algebra to find canonical logical Z/X operator pairs
- `build_bb_circuit()`: manually constructs the Stim circuit (6 CNOT layers per round matching the polynomial monomials); does not use `stim.Circuit.generated()`
- `BBCodeSimulator`: mirrors `SurfaceCodeSimulator` interface; defaults to `BPOSDDecoder` (belief propagation + ordered statistics) because MWPM cannot handle BB code hyperedge structure
- `BBPyMatchingDecoder`: MWPM fallback with `ignore_decomposition_failures=True`

**Key design note**: `bb_code_sim.py` imports `ErrorModel`, `Decoder`, `PyMatchingDecoder`, and `SimulationResult` from `surface_code_sim.py` — both simulators share the same result/noise/decoder abstractions.

**Simulation flow** (both codes):
1. Build noisy Stim circuit
2. Compile detector sampler, draw shots
3. Decode detection events → predicted observables
4. Compare predictions vs. actual flips → logical error rate ± binomial SE
