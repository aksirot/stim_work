"""Direct Monte Carlo of the LPU operations at high physical error rates.

Ground truth with no IS machinery: build each operation's circuit AT the target p
(config p_ref overridden, so p_meas/p_coupler scale with it faithfully), sample
stim detector data, decode with the deep600 relay (campaign gammas + 600 sets —
the validated paper-grade config; the published-gamma transcription is a trap,
see intermodule_decoder_ab.py), count any-observable failures. Empty-syndrome
shots are not decoded but an observable flip on one still counts (the fast-ladder
convention: undetected logicals are not free).

Anchors the IS spectra exactly like the 72-code fast-ladder MC checkpoints did,
and gives the interleaved inter-module leg its first end-to-end LER numbers.

    MC_OPS=inter_module_interleaved,inter_module_legacy MC_PS=1e-3,2e-3 \
      python experiments/methods/lpu_direct_mc.py

Env: MC_OPS (comma list or "all"), MC_PS (default "1e-3,2e-3"), MC_TARGET (30),
MC_SHOTS_MAX (4000), MC_DECODER ("deep600" | "campaign"), MC_BATCH (64).
Output: runs/framework/bb144/lpu_direct_mc.json (incremental per (op, p)) + stdout.
"""
import dataclasses
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from experiment_runner import load_config, build_circuit, make_decoder
from bb_code_sim import RelayBPDecoder
from repo_paths import REPO_ROOT

DEEP600 = dict(gamma0=0.125, pre_iter=80, num_sets=600, set_max_iter=60,
               gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
CFG_DIR = REPO_ROOT / "experiments" / "configs"
ALL_OPS = [("inter_module_interleaved", "gross_intermodule_r1_il"),
           ("inter_module_legacy", "gross_intermodule_r1"),
           # 2026-08-31 coupler-sensitivity study (interleaved idle + deep600):
           ("im_r1_deep600", "gross_intermodule_r1_il_deep600"),      # symmetric couplers
           ("im_r10_deep600", "gross_intermodule_r10_il_deep600"),    # 10x-worse Bell/coupler
           ("lpu_idle", "gross_lpu_idle"),
           ("automorphism", "gross_automorphism"),
           ("in_module_y1", "gross_lpu_y1")]
# MC_OUT lets concurrent containers (e.g. the r1 and r10 coupler cells) write to
# distinct files instead of racing the shared default.
OUT = pathlib.Path(os.environ.get(
    "MC_OUT", str(REPO_ROOT / "runs" / "framework" / "bb144" / "lpu_direct_mc.json")))

ops_env = os.environ.get("MC_OPS", "all")
OPS = ALL_OPS if ops_env == "all" else [o for o in ALL_OPS if o[0] in ops_env.split(",")]
PS = [float(x) for x in os.environ.get("MC_PS", "1e-3,2e-3").split(",")]
TARGET = int(os.environ.get("MC_TARGET", "30"))
SHOTS_MAX = int(os.environ.get("MC_SHOTS_MAX", "4000"))
DECODER = os.environ.get("MC_DECODER", "deep600")
BATCH = int(os.environ.get("MC_BATCH", "64"))


def main():
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, yml in OPS:
        for p in PS:
            key = f"{name}@{p:.0e}"
            if key in results and "ler" in results[key]:
                print(f"[{key}] cached, skip", flush=True)
                continue
            cfg = load_config(str(CFG_DIR / f"{yml}.yaml"))
            cfg = dataclasses.replace(cfg, p_ref=p)      # circuit AT the target p
            circ = build_circuit(cfg)
            dec = (RelayBPDecoder(**DEEP600) if DECODER == "deep600"
                   else make_decoder(cfg))
            t0 = time.time()
            dec.setup(circ)
            print(f"[{key}] qubits={circ.num_qubits} K={circ.num_observables} "
                  f"(setup {time.time()-t0:.0f}s) — target {TARGET} fails, "
                  f"cap {SHOTS_MAX}", flush=True)
            sampler = circ.compile_detector_sampler(seed=2026)
            fails = shots = decoded = 0
            t0 = time.time()
            while fails < TARGET and shots < SHOTS_MAX:
                dets, obs = sampler.sample(BATCH, separate_observables=True)
                nz = dets.any(axis=1)
                fails += int(obs[~nz].any(axis=1).sum())
                if nz.any():
                    pred = dec.decode_batch(dets[nz])
                    decoded += int(nz.sum())
                    fails += int((pred.astype(bool) != obs[nz].astype(bool))
                                 .any(axis=1).sum())
                shots += BATCH
                print(f"  [{key}] {fails} fails / {shots} shots "
                      f"({decoded/(max(time.time()-t0, 1e-9)):.2f} dec/s)", flush=True)
            ler = fails / shots
            results[key] = dict(op=name, p=p, fails=fails, shots=shots, ler=ler,
                                se_rel=(1/np.sqrt(fails) if fails else None),
                                K=int(circ.num_observables), decoder=DECODER,
                                elapsed_s=time.time() - t0)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"[{key}] LER = {fails}/{shots} = {ler:.3e}  -> saved", flush=True)

    print("\nsummary (per shot, per op's own K):")
    for k, r in sorted(results.items()):
        if "ler" in r:
            print(f"  {k:36s} LER = {r['ler']:.3e}  ({r['fails']}/{r['shots']}, "
                  f"K={r['K']}, {r['decoder']})")


if __name__ == "__main__":
    main()
