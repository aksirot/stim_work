"""Low-weight failure spectrum f(w) — the health check for a new LPU operation builder.

f(w) = fraction of weight-w fault configurations the decoder gets wrong. It is
**p-independent**, which is the whole point: a Monte-Carlo LER at p_ref cannot tell a
broken observable apart from a circuit operating far above threshold. Wave 6 learned
this the expensive way — the inter-module X1(A)xX1(B) circuit was recorded as having a
"floor" at LER~0.40 (p=5e-3) when in fact it sees ~106 expected faults/shot there and
the known-good Y1 baseline floors just as hard at the same point.

    ALWAYS compare a new builder against a VALIDATED baseline, never against zero.
    The validated Y1 baseline is itself nonzero at w=3,4,6 (decoder miscorrection at
    the cheap 20-set relay, not undetectable errors), so "matches the baseline" is the
    standard. This generalises the runbook's mandatory weight-1 degeneracy scan.

Resolution: with T samples per weight, f(w) resolves to ~1/T. A clean 0/T therefore
BOUNDS the floor at ~1/T rather than disproving one — to rule out the ~5e-4-class
decoder floor the campaign configs reference you need T >~ 10000, which is a cluster
job (chunked + checkpointed + resumable below, so it survives a walltime kill).

    # local smoke
    python experiments/tour_de_gross/failure_spectrum_probe.py --op y1 --weights 1 --T 40

    # cluster: the large-T confirmation, resumable
    python experiments/tour_de_gross/failure_spectrum_probe.py \
        --op inter_module --weights 1 2 3 4 5 6 --T 10000 --workers 48 \
        --out runs/framework/bb144/fw_inter_largeT

Re-running with a larger --T TOPS UP the existing checkpoint rather than restarting.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bb_code_sim import RelayBPDecoder
from gross_code_lpu_tdg import build_joint_pauli_circuit, build_joint_x1x1_circuit
from importance_sampling import _expand, _parse_dem, _sample_failures_at_weight
from surface_code_sim import ErrorModel

# Each op maps to (builder, kwargs). "y1" is the VALIDATED baseline to compare against.
OPS = {
    "y1": (build_joint_pauli_circuit, {"include_memory_observables": False}),
    "inter_module": (build_joint_x1x1_circuit, {}),
}

_W = {}          # per-process worker state (circuit/decoder are not picklable)


def build(op: str, C: int, d_init: int, p: float, close_cycles: bool, idle: bool):
    builder, kw = OPS[op]
    kw = dict(kw)
    if op == "inter_module":
        kw["close_cycles"] = close_cycles
    return builder(ErrorModel(p_phys=p, p_meas=p), C=C, d_init=d_init, idle_noise=idle, **kw)


def _init(op, C, d_init, p, close_cycles, idle, num_sets, stop_nconv):
    circuit = build(op, C, d_init, p, close_cycles, idle)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, _, _ = _expand(probs, None)
    dec = RelayBPDecoder(num_sets=num_sets, stop_nconv=stop_nconv)
    dec.setup(circuit)
    _W.update(det=det_mat, obs=obs_mat, c2m=col_to_mech, dec=dec)


def _chunk(task):
    """One (weight, chunk) unit of work -> failure count. Seed is per-chunk => reproducible."""
    w, n, seed = task
    rng = np.random.default_rng(seed)
    return _sample_failures_at_weight(_W["det"], _W["obs"], _W["c2m"], w, n, _W["dec"], rng)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", choices=sorted(OPS), default="inter_module")
    ap.add_argument("--weights", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--T", type=int, default=400, help="samples per weight (target total)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50, help="samples per work unit")
    ap.add_argument("--C", type=int, default=3)
    ap.add_argument("--d-init", type=int, default=2)
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--idle-noise", action="store_true")
    ap.add_argument("--no-close-cycles", dest="close_cycles", action="store_false",
                    help="inter_module only: reproduce the pre-closure circuit")
    ap.add_argument("--relay-num-sets", type=int, default=20)
    ap.add_argument("--relay-stop-nconv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="checkpoint dir; re-running with a larger --T tops up")
    args = ap.parse_args()

    key = dict(op=args.op, C=args.C, d_init=args.d_init, p=args.p, idle=args.idle_noise,
               close_cycles=args.close_cycles if args.op == "inter_module" else None,
               num_sets=args.relay_num_sets, stop_nconv=args.relay_stop_nconv, seed=args.seed)

    done: dict[int, dict] = {}
    ckpt = None
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        ckpt = args.out / "spectrum.json"
        if ckpt.exists():
            prev = json.loads(ckpt.read_text(encoding="utf-8"))
            if prev.get("key") != key:
                print(f"ERROR: {ckpt} was written with a different configuration.\n"
                      f"  on disk: {prev.get('key')}\n  now:     {key}\n"
                      f"Refusing to mix incomparable samples — use a different --out.",
                      file=sys.stderr)
                return 2
            done = {int(w): v for w, v in prev["results"].items()}
            print(f"resuming from {ckpt}: " +
                  ", ".join(f"f({w})={v['fail']}/{v['T']}" for w, v in sorted(done.items())),
                  flush=True)

    print(f"f(w) spectrum — op={args.op} C={args.C} d_init={args.d_init} p={args.p:g} "
          f"idle={args.idle_noise} close_cycles={key['close_cycles']} "
          f"relay={args.relay_num_sets}-set  target T={args.T}", flush=True)

    init_args = (args.op, args.C, args.d_init, args.p, args.close_cycles, args.idle_noise,
                 args.relay_num_sets, args.relay_stop_nconv)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=init_args) as pool:
        for w in args.weights:
            have = done.get(w, {"T": 0, "fail": 0})
            need = args.T - have["T"]
            if need <= 0:
                print(f"  f({w}) = {have['fail']/have['T']:.5f}   "
                      f"({have['fail']}/{have['T']}) [cached]", flush=True)
                continue
            t0 = time.time()
            # distinct seed per (weight, chunk-offset) so a top-up never repeats samples
            tasks = []
            off = have["T"]
            while off < args.T:
                n = min(args.chunk, args.T - off)
                tasks.append((w, n, args.seed * 1_000_003 + w * 10_007 + off))
                off += n
            fails = have["fail"] + sum(pool.map(_chunk, tasks))
            T = args.T
            f = fails / T
            se = float(np.sqrt(max(f * (1 - f), 1e-12) / T))
            done[w] = {"T": T, "fail": int(fails)}
            print(f"  f({w}) = {f:.5f} +- {se:.5f}   ({fails}/{T} fail) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if ckpt:      # checkpoint after EVERY weight — survives a walltime kill
                ckpt.write_text(json.dumps({"key": key, "results": done}, indent=2),
                                encoding="utf-8")

    nonzero = {w: v for w, v in done.items() if v["fail"]}
    print(f"\nresolution ~{1/args.T:.2g} per weight; "
          f"{'nonzero at w=' + ','.join(map(str, sorted(nonzero))) if nonzero else 'all bins clean'}",
          flush=True)
    print("REMINDER: compare against the validated y1 baseline, not against zero.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
