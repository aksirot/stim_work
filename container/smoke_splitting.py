#!/usr/bin/env python
"""Container smoke for the SPLITTING job type: a tiny paper-faithful ladder.

`smoke_decode.py` proves the simulator + decoder run on a node. This proves the actual
production code path runs: `multi_seeded_split_estimate` (arXiv:2511.15177 Alg. 2/3 —
Eq.18 fine ladder, multi-seeded warm starts, sigma+Delta chain-growth controller), just
scaled to minutes — a short p range, 2 chain instances, 200 samples per level.

Prints one `SPLIT PASS ...` line and writes a per-task JSON under runs/_smoke_split/, so
a SLURM array proves N nodes can each run the workload AND write to shared storage.
Seed comes from $SLURM_ARRAY_TASK_ID (or $IDX) so array tasks are independent.

MEASURED cost (this laptop, 8 threads, ghw): the default settings below are ~1.3k
decodes ~= 3-4 min. Do not widen the p range casually — the ladder sits at high p where
the mean fault weight is ~29, so every decode is a hard one (~6 decodes/s/8-threads).
An earlier draft spanning 8e-3 -> 5e-3 with T=200 took 15 minutes.

    PYTHON_SCRIPT=stim_work/container/smoke_splitting.py     # from the parent dir
    SMOKE_DECODER=ghw_deep    # ~4x slower; good per-node calibration for the real run
    SMOKE_T=400 SMOKE_PLOW=5e-3   # scale up: doubles as a throughput benchmark

Needs `src/` importable (PYTHONPATH=<repo>/src if the image has no editable install).
"""
import json
import os
import socket
import sys
import time
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "methods"))
sys.path.insert(0, str(REPO / "src"))

import run_error_model_comparison as rmc            # noqa: E402
from splitting import multi_seeded_split_estimate   # noqa: E402

idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("IDX", "0")))
variant = os.environ.get("SMOKE_DECODER", "ghw")
if variant not in rmc.DECODER_VARIANTS:
    raise SystemExit(f"SMOKE_DECODER={variant!r} not in this checkout "
                     f"({sorted(rmc.DECODER_VARIANTS)}) — the tree is too old")

circ = rmc.make_circuit72("full symmetric", rmc.P_REF)
calib = rmc.make_circuit72("full symmetric", rmc.DECODER_P)
dec = rmc.DEC(calib, dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS[variant]))

T = int(os.environ.get("SMOKE_T", "80"))
p_low = float(os.environ.get("SMOKE_PLOW", "6.5e-3"))

t0 = time.time()
res, diag = multi_seeded_split_estimate(
    circ, dec, p_ref=rmc.P_REF, p_high=8e-3, p_low=p_low,
    L=2, M=1, T_init=T, T_cap=4 * T, eps=0.5,
    ladder="eq18", distance=8, anchor_shots=300, seed=100 + idx, verbose=False)
dt = time.time() - t0

host = socket.gethostname()
print(f"SPLIT PASS host={host} task={idx} decoder={variant} "
      f"levels={len(res.p_ladder)} P({res.p_ladder[-1]:.2e})={res.P_logical[-1]:.3e} "
      f"anchor={diag['P_high']:.3e} {dt:.0f}s", flush=True)

out_dir = REPO / "runs" / "_smoke_split"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{host}_{idx}.json").write_text(json.dumps(dict(
    host=host, task=idx, decoder=variant, elapsed_s=round(dt, 1),
    p=[float(x) for x in res.p_ladder], P=[float(x) for x in res.P_logical],
    P_high=float(diag["P_high"]), n_levels=len(res.p_ladder)), indent=1), encoding="utf-8")
