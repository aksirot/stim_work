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
from lpu_noise_scaling import M5, scale_im_noise      # shared with the IS runner (Config.noise_scale_factors)
from repo_paths import REPO_ROOT

DEEP600 = dict(gamma0=0.125, pre_iter=80, num_sets=600, set_max_iter=60,
               gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
CFG_DIR = REPO_ROOT / "experiments" / "configs"
# (op name, config yaml[, noise-scale factors applied to the built circuit])
ALL_OPS = [("inter_module_interleaved", "gross_intermodule_r1_il"),
           ("inter_module_legacy", "gross_intermodule_r1"),
           # 2026-08-31 coupler-sensitivity study (interleaved idle + deep600):
           ("im_r1_deep600", "gross_intermodule_r1_il_deep600"),      # symmetric couplers
           ("im_r10_deep600", "gross_intermodule_r10_il_deep600"),    # 10x-worse Bell/coupler
           # 2026-09-21 d_init sweep (merge vs idle attribution), campaign decoder
           # (MC_DECODER=campaign -> make_decoder(cfg) = the yaml's relay_num_sets=20):
           ("im_r1_fast", "gross_intermodule_r1_il_fast"),            # d_init=12 (MC cross-check of the IS)
           ("im_r10_fast", "gross_intermodule_r10_il_fast"),
           ("im_r1_fast_d6", "gross_intermodule_r1_il_fast_d6"),      # d_init=6: half the idle padding
           ("im_r10_fast_d6", "gross_intermodule_r10_il_fast_d6"),
           # 2026-09-22 Tour de Gross framing: NO padding (d_init=0) + noiseless return
           # cycle = the paper's per-instruction convention (Sec. 2.6). Paper: gross
           # inter-module meas. LER 10^-2.7 at p=1e-3 (Table 2).
           ("im_r1_fast_tdg", "gross_intermodule_r1_il_fast_tdg"),
           ("im_r10_fast_tdg", "gross_intermodule_r10_il_fast_tdg"),
           # 2026-09-22 asymmetric device point: meas x5 + meas-idle x5 (see scale_im_noise)
           ("im_r1_fast_m5", "gross_intermodule_r1_il_fast", M5),
           ("im_r10_fast_m5", "gross_intermodule_r10_il_fast", M5),
           ("im_r1_fast_d6_m5", "gross_intermodule_r1_il_fast_d6", M5),
           ("im_r10_fast_d6_m5", "gross_intermodule_r10_il_fast_d6", M5),
           ("im_r1_fast_tdg_m5", "gross_intermodule_r1_il_fast_tdg", M5),
           ("im_r10_fast_tdg_m5", "gross_intermodule_r10_il_fast_tdg", M5),
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
    for entry in OPS:
        name, yml = entry[0], entry[1]
        scale = entry[2] if len(entry) > 2 else None
        for p in PS:
            key = f"{name}@{p:.0e}"
            if key in results and "ler" in results[key]:
                print(f"[{key}] cached, skip", flush=True)
                continue
            cfg = load_config(str(CFG_DIR / f"{yml}.yaml"))
            cfg = dataclasses.replace(cfg, p_ref=p)      # circuit AT the target p
            circ = build_circuit(cfg)
            if scale:
                circ = scale_im_noise(circ, p, scale)
            dec = (RelayBPDecoder(**DEEP600) if DECODER == "deep600"
                   else make_decoder(cfg))
            t0 = time.time()
            dec.setup(circ)
            print(f"[{key}] qubits={circ.num_qubits} K={circ.num_observables} "
                  f"(setup {time.time()-t0:.0f}s) — target {TARGET} fails, "
                  f"cap {SHOTS_MAX}", flush=True)
            sampler = circ.compile_detector_sampler(seed=2026)
            # obs 0 = measurement OUTPUT; obs 1.. = preserved MEMORY logicals
            # (K=23 inter-module recipe). Split failures into out/mem/both — the
            # paper's F/F_mem/F_out/F_both. For K=1 circuits F_mem stays 0.
            K = int(circ.num_observables)
            fails = f_out = f_mem = f_both = shots = decoded = 0
            t0 = time.time()
            while fails < TARGET and shots < SHOTS_MAX:
                dets, obs = sampler.sample(BATCH, separate_observables=True)
                nz = dets.any(axis=1)
                resid = obs.astype(bool).copy()          # empty-syndrome: pred=0
                if nz.any():
                    pred = dec.decode_batch(dets[nz])
                    resid[nz] = pred.astype(bool) != obs[nz].astype(bool)
                    decoded += int(nz.sum())
                out_bad = resid[:, 0]
                mem_bad = resid[:, 1:].any(axis=1) if K > 1 else np.zeros(len(resid), bool)
                any_bad = out_bad | mem_bad
                fails += int(any_bad.sum())
                f_out += int(out_bad.sum())
                f_mem += int(mem_bad.sum())
                f_both += int((out_bad & mem_bad).sum())
                shots += BATCH
                print(f"  [{key}] {fails}F ({f_mem}m/{f_out}o/{f_both}b) / {shots} shots "
                      f"({decoded/(max(time.time()-t0, 1e-9)):.2f} dec/s)", flush=True)
            ler = fails / shots
            results[key] = dict(op=name, p=p, fails=fails, shots=shots, ler=ler,
                                f_out=f_out, f_mem=f_mem, f_both=f_both,
                                ler_out=f_out / shots, ler_mem=f_mem / shots,
                                se_rel=(1/np.sqrt(fails) if fails else None),
                                K=K, decoder=DECODER, elapsed_s=time.time() - t0,
                                # circuit-shape tags for the d_init-sweep notebook cell
                                d_init=cfg.lpu_d_init, C=cfg.lpu_C,
                                coupler_factor=cfg.p_coupler_factor,
                                noiseless_return=cfg.lpu_noiseless_return,
                                noise_scale=scale,
                                relay_num_sets=(None if DECODER == "deep600"
                                                else cfg.relay_num_sets))
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"[{key}] LER={fails}/{shots}={ler:.3e}  out={f_out} mem={f_mem} "
                  f"both={f_both}  -> saved", flush=True)

    print("\nsummary (per shot, per op's own K):")
    for k, r in sorted(results.items()):
        if "ler" in r:
            print(f"  {k:36s} LER = {r['ler']:.3e}  ({r['fails']}/{r['shots']}, "
                  f"K={r['K']}, {r['decoder']})")


if __name__ == "__main__":
    main()
