"""w=100 bin across ALL LPU operations, decoded with the deep600 relay.

Five rows: idle, automorphism, in-module (y1/joint), inter_module legacy,
inter_module interleaved. Same weight, same decoder — the closest single-bin
analogue of Fig 7's right panels. Compare at matched w/mu (printed), not raw w:
w=100 is 57% of the idle circuit's mean fault count but 6.6% of inter-module's.

deep600 = 600 relay sets with the CAMPAIGN's validated gammas. The literal
published-gamma transcription (gamma0=0.79, interval (0.34,1.24)) is a convention
trap under our wrapper: 2026-08-18 factorization on idle w=100 measured it at
85/128 fails (vs campaign 11/128) at 20 sets, while cheap-gammas+600sets scored
2/128. Set count is the production axis; those gammas are not portable.
Incremental JSON so partial results survive.

    BIN_W=100 BIN_SHOTS=512 python experiments/methods/lpu_w100_paper_sweep.py
"""
import json, os, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from experiment_runner import load_config, build_circuit
from bb_code_sim import RelayBPDecoder
from importance_sampling import _parse_dem, _expand
from subonset_relay_sweep import sample_batch
from repo_paths import REPO_ROOT

W = int(os.environ.get("BIN_W", "100"))
SHOTS = int(os.environ.get("BIN_SHOTS", "512"))
PAPER = dict(gamma0=0.125, pre_iter=80, num_sets=600, set_max_iter=60,
             gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)   # deep600
CFG_DIR = REPO_ROOT / "experiments" / "configs"
OPS = [("lpu_idle", "gross_lpu_idle"),
       ("automorphism", "gross_automorphism"),
       ("joint_pauli", "gross_lpu_y1"),          # in-module joint measurement leg
       ("inter_module_legacy", "gross_intermodule_r1"),
       ("inter_module_interleaved", "gross_intermodule_r1_il")]
OUT = str(REPO_ROOT / "runs" / "framework" / "bb144" / "w100_paper_sweep.json")

results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))
for name, yml in OPS:
    if name in results:
        print(f"[{name}] cached, skip", flush=True)
        continue
    try:
        cfg = load_config(str(CFG_DIR / f"{yml}.yaml"))
        if name == "joint_pauli" and cfg.experiment != "joint_pauli":
            print(f"[{name}] {yml} is experiment={cfg.experiment!r} — recording as that",
                  flush=True)
            name = f"in_module_{cfg.experiment}"
            if name in results:
                continue
        circ = build_circuit(cfg)
        dec = RelayBPDecoder(**PAPER)
        dec.setup(circ)
        probs, det, obs = _parse_dem(circ)
        c2m, q_base, _ = _expand(probs, None)
        mu = c2m.shape[0] * q_base
        print(f"[{name}] n_exp={c2m.shape[0]:,} mu={mu:.0f} — decoding {SHOTS} @ w={W}",
              flush=True)
        rng = np.random.default_rng(2026)
        fails, done, t0 = 0, 0, time.time()
        while done < SHOTS:
            B = min(128, SHOTS - done)
            _, syn, tru = sample_batch(rng, c2m, det, obs, W, B)
            fails += int(np.any(dec.decode_batch(syn) != tru, axis=1).sum())
            done += B
            print(f"  [{name}] {fails}/{done} ({done/(time.time()-t0):.2f}/s)", flush=True)
        results[name] = dict(w=W, fails=fails, trials=done, f=fails / done,
                             n_expanded=int(c2m.shape[0]), q_base=float(q_base),
                             mu_pref=float(mu), w_over_mu=W / mu,
                             decoder="deep600", elapsed_s=time.time() - t0)
        json.dump(results, open(OUT, "w"), indent=1)
        print(f"[{name}] f({W}) = {fails}/{done} = {fails/done:.3e}  -> saved", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
        results[name] = dict(error=str(e)[:200])
        json.dump(results, open(OUT, "w"), indent=1)

print("\nsummary:")
for n, r in results.items():
    if "f" in r:
        print(f"  {n:28s} f({r['w']}) = {r['fails']:4d}/{r['trials']}  = {r['f']:.3e}   "
              f"w/mu = {r['w_over_mu']:.3f}")
