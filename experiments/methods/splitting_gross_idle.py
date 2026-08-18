"""Paper-faithful splitting on the GROSS-code LPU idle — the paper's own system.

The fish production run (single-node, nohup-sharded). Context: at laptop budgets
(T_init=4k) the eq18 multi-seeded ladder FAILED arbitration on the [[72,4,8]] —
frozen weight distribution, 1000x low at 8e-4 — while the paper reports the method
working on the gross code at T_init=1e6, L=12, M=3 (~1e9 decodes). This driver asks
the sharp question at the paper's own operating point: does paper-scale budget
restore equilibration (watch mean-w in the per-level diagnostics), and does the
endpoint agree with the IS-reweighted gross-idle spectrum?

One process = one shard (its own warm-start harvest + M chains, distinct seed):

    GS_TAG=s0 GS_SEED=100 GS_TINIT=100000 python splitting_gross_idle.py
    ... repeat with distinct GS_TAG/GS_SEED across the node; the combiner
    (splitting_shard_combine.py, run locally) pools shards, paper-convention.

Env: GS_PHIGH (4e-3), GS_PLOW (3e-4), GS_L (1), GS_M (3), GS_TINIT (1e5),
GS_TCAP (4x TINIT), GS_ANCHOR (20000), GS_SEED (100), GS_TAG (""), SMOKE=1.
Output: runs/splitting_gross/ladder_<tag>.json
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from experiment_runner import load_config, build_circuit, make_decoder
from splitting import multi_seeded_split_estimate
from repo_paths import REPO_ROOT

OUT = REPO_ROOT / "runs" / "splitting_gross"
SMOKE = bool(int(os.environ.get("SMOKE", "0")))
YAML = REPO_ROOT / "experiments" / "configs" / "gross_lpu_idle.yaml"

P_HIGH = float(os.environ.get("GS_PHIGH", "4e-3"))
P_LOW = float(os.environ.get("GS_PLOW", "3e-4"))
SEED = int(os.environ.get("GS_SEED", "100"))
TAG = os.environ.get("GS_TAG", "")
D_GROSS = 10                       # gross-idle circuit distance (paper: w0 = 5)


def main():
    cfg = load_config(str(YAML))
    circ = build_circuit(cfg)
    dec = make_decoder(cfg)
    p_ref = float(cfg.p_ref)          # Config dataclass, attribute access
    kw = (dict(L=1, M=2, T_init=300, T_cap=600, anchor_shots=500) if SMOKE else
          dict(L=int(os.environ.get("GS_L", "1")),
               M=int(os.environ.get("GS_M", "3")),
               T_init=int(float(os.environ.get("GS_TINIT", "1e5"))),
               T_cap=int(float(os.environ.get("GS_TCAP",
                                              str(4 * float(os.environ.get("GS_TINIT", "1e5")))))),
               anchor_shots=int(float(os.environ.get("GS_ANCHOR", "20000")))))
    print(f"[gross-split] p_ref={p_ref}  {P_HIGH:.0e} -> {P_LOW:.0e}  {kw}  "
          f"seed={SEED} tag={TAG!r}", flush=True)
    t0 = time.time()
    res, diag = multi_seeded_split_estimate(
        circ, dec, p_ref=p_ref, p_high=P_HIGH, p_low=P_LOW,
        eps=0.3, ladder="eq18", distance=D_GROSS, seed=SEED, **kw)
    out = dict(target="gross_lpu_idle", algorithm="multi_seeded_split_estimate eq18",
               p_ref=p_ref, p_high=P_HIGH, p_low=P_LOW, budgets=kw,
               seed=SEED, tag=TAG, smoke=SMOKE,
               sp=np.asarray(res.p_ladder).tolist(),
               sP=np.asarray(res.P_logical).tolist(),
               sP_se=np.asarray(res.P_logical_se).tolist(),
               elapsed_s=time.time() - t0,
               diag={k: v for k, v in diag.items()
                     if isinstance(v, (int, float, str, list, dict))})
    OUT.mkdir(parents=True, exist_ok=True)
    fname = f"ladder_{TAG}.json" if TAG else "ladder.json"
    (OUT / fname).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[gross-split] S({P_LOW:.0e}) = {out['sP'][-1]:.3e}  "
          f"({len(out['sp'])} rungs, {out['elapsed_s'] / 3600:.2f} h)  wrote {fname}",
          flush=True)


if __name__ == "__main__":
    main()
