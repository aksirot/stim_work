# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

`relay_bp` requires Rust to build. Install Rust first, then install all packages:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install -r requirements.txt
```

Key packages: `stim` (quantum circuit simulation), `pymatching` (MWPM decoder), `ldpc` (BP-OSD decoder, fallback for BB codes), `relay_bp` (Rust-native relay-BP decoder, default for BB codes).

## Agentic workflow rules

These rules apply whenever Claude Code operates autonomously in this repo:

- **Branch**: always work on a feature branch, never commit directly to `main`. If no branch is active, create one before making changes.
- **Packages**: never install packages outside the `.venv` in the repo root. Always use `.venv/bin/pip install ...`. Never use `sudo pip`, `pip3`, or a system Python.
- **Commits**: commit only the files relevant to the task. Do not stage `__pycache__/`, `.venv/`, or notebook output (already gitignored). Do not amend published commits.
- **Notebooks**: do not execute notebook cells autonomously — edits to `.ipynb` files are source-only. Leave execution to the user.
- **Simulations**: do not run long simulations (more than ~30s) without explicit user instruction. Use small shot counts (≤50) for quick sanity checks only.
- **Scope**: do not modify `surface_code_sim.py` or `bb_code_sim.py` in ways that change existing public APIs without confirming with the user first.

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
