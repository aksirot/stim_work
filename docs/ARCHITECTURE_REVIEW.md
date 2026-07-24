# Architecture review — `src/` reorganization

Review of the `src/` library as of 2026-07-24, against the goal of a cleaner API: **circuit-building**,
**DEM analysis**, **one module per fail-fast technique (I/II/III)**, and an **analysis layer** (Λ work +
code/error-model comparison), with report code out of the library.

## TL;DR

The good news: the code already has a sound *latent* structure — two almost-disjoint trees (a
circuit/simulator tree and an estimator/analysis tree), **no import cycles**, and the technique
modules never reach up into circuit code. The problems are **four specific misplacements**, not a
tangle. None require a rewrite; all can be done incrementally behind re-export shims so the notebooks
(which can't be auto-run) keep importing the old names during migration.

## Current state

| Module | LOC | Role | Notable smell |
|---|---:|---|---|
| `repo_paths.py` | 20 | path anchors | — (leaf, fine) |
| `surface_code_sim.py` | 371 | surface sim **+ repo-wide `ErrorModel`/`Decoder`/`SimulationResult`** | contracts mis-homed here |
| `bb_code_sim.py` | 821 | BB algebra + `build_bb_circuit` + decoders + simulator | bundles builder + sim + decoders |
| `gross_code_lpu_tdg.py` | **2810** | LPU: algebra + layout + 5 circuit builders + driver | **god module**; dup `build_HX_HZ` |
| `lpu_graph.py` | 550 | LPU layout derivation/certification | dup `build_HX_HZ` with the above |
| `importance_sampling.py` | 707 | **Technique I** + private DEM `_parse_dem`/`_expand` | DEM plumbing hidden here |
| `min_weight.py` | **1456** | **Technique II** + `single_sector_dem`/action matrices | mixes DEM-construction w/ Tech II |
| `splitting.py` | 1254 | **Technique III** | depends on I + II (shared DEM plumbing) |
| `lambda_analysis.py` | 321 | Λ / error-budget analysis | analysis, but `runs/`-coupled |
| `bb6_report.py` | 547 | **matplotlib report figures for a notebook** | **report code inside the library** |
| `experiment_runner.py` | 909 | config-driven orchestrator (top of tree) | fine (lazy imports hide some coupling) |

Coupling (no cycles):

```
repo_paths, surface_code_sim        (leaves)
surface_code_sim → bb_code_sim → { gross_code_lpu_tdg, lpu_graph }
importance_sampling → { min_weight, splitting, lambda_analysis, bb6_report }
                      (splitting also → min_weight)
experiment_runner → bb_code_sim, surface_code_sim, importance_sampling, repo_paths
                    (+ gross_code_lpu_tdg, min_weight, splitting lazily)
```

## What's actually wrong (only four things)

1. **Repo-wide contracts live in the surface-code file.** `ErrorModel`, `Decoder`,
   `SimulationResult` are defined in `surface_code_sim.py`; `bb_code_sim`, `gross_code_lpu_tdg`, and
   `experiment_runner` all get them *transitively through `bb_code_sim`*. The BB/LPU stack imports the
   surface-code module just to reach shared types.
2. **DEM logic is scattered and half-hidden.** The low-level `_parse_dem`/`_expand` are underscore-
   privates in `importance_sampling` (Technique I), yet they're the shared substrate imported by
   Technique II *and* III. The higher-level `single_sector_dem` + action matrices are buried inside
   `min_weight` (Technique II) even though `splitting` and `experiment_runner` also use them. There is
   no "DEM API" — it's an implementation detail of two different technique files.
3. **`gross_code_lpu_tdg.py` is a 2810-line god module** — code constants, monomial algebra, graph/
   edge derivation, measurement tracking, five distinct circuit builders, and a `run_lpu` driver — and
   it re-defines `build_HX_HZ`, which also exists in `lpu_graph.py`.
4. **`bb6_report.py` is report/plotting code in the importable library.** It has `fig_*` plotters,
   hard-coded `runs/` paths, and a `main()`. **Nothing in `src/` imports it** — its only consumers are
   report notebooks and `experiments/bravyi/notebook_builders/`. It is pure notebook infrastructure.

## Target layout

```
src/
  repo_paths.py                 # unchanged
  core/                         # ← the keystone: neutral shared contracts, no code/technique deps
    noise.py                    #   ErrorModel                         (from surface_code_sim)
    decoder.py                  #   Decoder base + PyMatching/BPOSD/RelayBP/BBPyMatching
    result.py                   #   SimulationResult
    dem.py                      #   THE DEM API: parse_dem, expand, single_sector_dem,
                                #     action matrices, build_circuit_translation_perms
  circuits/                     # ← the "make stim circuits" API (pure builders)
    surface.py                  #   surface generation (+ SurfaceCodeSimulator wrapper)
    bb.py                       #   BB algebra, build_parity_checks, find_logical_ops,
                                #     build_bb_circuit, noise editing (+ BBCodeSimulator)
    lpu/
      algebra.py                #   monomials, build_HX_HZ  (single source; kills the dup)
      layout.py                 #   lpu_graph.py + edge/graph derivation
      builders.py               #   build_{x1,z1,idle,automorphism,joint_pauli}_circuit
  techniques/
    technique1_is.py            #   importance_sampling  (minus DEM plumbing → core.dem)
    technique2_minweight.py     #   min_weight           (minus DEM construction → core.dem)
    technique3_splitting.py     #   splitting
  analysis/
    lambda_budget.py            #   lambda_analysis
    compare.py                  #   reusable cross-code / error-model comparison
experiments/
  reporting/
    bb6_report.py               # ← moved out of the library
```

This is a direct expression of the stated goal: `circuits/` makes stim circuits, `core/dem.py` is the
DEM API, `techniques/` is one file per technique, `analysis/` holds the Λ work and comparisons, and the
report code leaves `src/`.

## Why `core/` is the keystone

Right now the technique tree and the circuit tree are only *accidentally* disjoint: they avoid each
other, but the shared types (`ErrorModel`, `Decoder`, `SimulationResult`) and shared DEM plumbing have
no neutral home, so they squat inside whichever module happened to define them first
(`surface_code_sim`, `importance_sampling`, `min_weight`). Extracting `core/` turns those into a real
shared foundation that **both** trees depend *down* into — after which `circuits/*` and `techniques/*`
become clean peers, each depending only on `core`, and the surface→BB→LPU import chain (which today
exists only to reach `ErrorModel`) dissolves.

## Migration plan — sequenced by risk × value

Do these **between campaigns, not while jobs are running** (`experiment_runner` and the YAML configs
reference module paths). Run `pytest` after each phase — the tests are the safety net.

| Phase | Move | Risk | Notes |
|---|---|---|---|
| **0** | `bb6_report.py` → `experiments/reporting/` | ~none | nothing in `src/` imports it; pure win, closes the flagged smell |
| **1** | extract `core/{noise,decoder,result}.py` | low | mechanical; leave re-export shims in `surface_code_sim`/`bb_code_sim` so old imports + notebooks keep working |
| **2** | extract `core/dem.py` (parse/expand + single_sector_dem + action matrices) | low–med | promotes the private DEM plumbing to a real API; techniques import `core.dem` |
| **3** | split `gross_code_lpu_tdg.py` → `circuits/lpu/{algebra,layout,builders}`; dedup `build_HX_HZ` with `lpu_graph` | med | large file but self-contained (only `experiment_runner` + tests import it) |
| **4** | rename techniques → `technique{1,2,3}_*`; `analysis/{lambda_budget,compare}` | low, churny | cosmetic + import-path churn; do last |

**Defer / discuss first:**
- **Full `qec.` namespace package** (`from qec.circuits.bb import ...`). Biggest churn — rewrites every
  import site *and every notebook*. Not worth it unless you also want the packaging benefits. A lighter
  alternative keeps flat top-level modules with the clearer names and skips the namespace.
- **Lifting `compare.py` out of `experiments/methods/run_error_model_comparison.py`.** The reusable
  comparison logic is real, but it's currently entangled with campaign-specific driver code (and
  `lambda_analysis` was *already* extracted from its sibling). Extract carefully, or leave the driver
  and only lift the genuinely code-agnostic helpers.

## Two design calls worth making explicitly

- **Pure builders vs. simulators.** Each circuit file today bundles a `*Simulator` convenience class and
  decoder glue with the pure circuit construction. If you want a clean "make a stim circuit" API, keep
  `circuits/*` as pure builders and either fold the `Simulator` wrappers into the same file (documented
  as convenience) or give them a thin `sim/` layer. Don't let simulator/decoder glue creep back into the
  builder API.
- **Is `lambda_analysis` library or driver?** It's clean analysis (no plotting) but couples to `runs/`
  JSON via `Run`/`load_run`. Split it: the pure Λ math stays in `analysis/lambda_budget.py`; the
  `runs/`-loading helpers move next to the drivers/reporting.
