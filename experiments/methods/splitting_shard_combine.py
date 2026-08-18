"""Pool sharded splitting ladders and referee them against IS/MC.

Each fish shard writes ladder_<tag>.json (its own warm-start harvest + M chains,
distinct seed). This combiner treats each shard as one instance-group, pools in log
space (mean of ln S per rung; spread = std of ln S across shards — the paper's
across-instance error convention, one level up), and prints:

  * the pooled ladder with per-rung spread
  * mean-w drift per shard (the equilibration diagnostic: laptop failure mode was
    mean-w frozen ~15-17 over a 40x p range)
  * where referee data exists (72-code fast_ladder: stride-filled IS + MC
    checkpoints; gross: the lpu_idle IS spectrum), z per point

    python experiments/methods/splitting_shard_combine.py runs/splitting_gross
    python experiments/methods/splitting_shard_combine.py runs/splitting_crosscheck/fast_ladder
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from repo_paths import REPO_ROOT


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    d = (REPO_ROOT / argv[0]) if not pathlib.Path(argv[0]).is_absolute() \
        else pathlib.Path(argv[0])
    shards = sorted(d.glob("ladder_*.json"))
    if not shards:
        raise SystemExit(f"no ladder_*.json shards under {d}")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in shards]
    p0 = runs[0]["sp"]
    for r, f in zip(runs, shards):
        if r["sp"] != p0:
            raise SystemExit(f"{f.name}: rung grid differs — shards must share "
                             f"P_HIGH/P_LOW/budgets to pool")
    sp = np.asarray(p0, float)
    lnS = np.log(np.clip(np.asarray([r["sP"] for r in runs], float), 1e-300, None))
    pooled = np.exp(lnS.mean(axis=0))
    spread = lnS.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(sp)

    print(f"{len(runs)} shards from {d.name}: "
          + ", ".join(f.stem.replace('ladder_', '') for f in shards))
    print(f"budgets (shard 0): {runs[0]['budgets']}   range {sp.max():.1e} -> {sp.min():.1e} "
          f"({len(sp)} rungs)")
    print(f"\n{'p':>10}  {'pooled S':>11}  {'ln-spread':>9}")
    step = max(1, len(sp) // 12)
    for i in range(0, len(sp), step):
        print(f"{sp[i]:10.3e}  {pooled[i]:11.3e}  {spread[i]:9.2f}")
    if len(sp) % step != 1:
        print(f"{sp[-1]:10.3e}  {pooled[-1]:11.3e}  {spread[-1]:9.2f}")

    print("\nequilibration (mean-w at top rung -> bottom rung, per shard):")
    for r, f in zip(runs, shards):
        mw = r.get("diag", {}).get("mean_w_per_level") or r.get("diag", {}).get("mean_w")
        if isinstance(mw, list) and mw:
            print(f"  {f.stem:18s}: {float(mw[0]):5.1f} -> {float(mw[-1]):5.1f}")
        else:
            print(f"  {f.stem:18s}: (no per-level mean-w in diag — read the shard log)")

    # referees, where this directory has them
    fl = REPO_ROOT / "runs" / "splitting_crosscheck" / "fast_ladder"
    if d.resolve() == fl.resolve():
        for f in sorted(fl.glob("mc_*.json")):
            mc = json.loads(f.read_text(encoding="utf-8"))
            if not mc["fails"]:
                continue
            x = np.log(mc["p"])
            order = np.argsort(sp)
            S = float(np.exp(np.interp(x, np.log(sp[order]), np.log(pooled[order]))))
            rel = float(np.interp(x, np.log(sp[order]), spread[order]))
            z = abs(np.log(S) - np.log(mc["ler"])) / max(np.hypot(rel, mc["se_rel"]), 1e-9)
            print(f"  vs MC p={mc['p']:.1e}: pooled {S:.3e}  MC {mc['ler']:.3e}  "
                  f"z={z:.2f}  {'PASS' if z < 3 else 'FAIL'}")
        print("  (IS comparison: run the fast-ladder 'report' stage — it reads the "
              "pooled file if you save it as ladder.json)")
    else:
        spec_dir = REPO_ROOT / "runs" / "framework" / "bb144" / "lpu_idle"
        if (spec_dir / "spectrum.json").exists():
            from lpu_ansatz_fits import load_spectrum
            from importance_sampling import reweight_spectrum
            spec = load_spectrum(spec_dir)
            rw = reweight_spectrum(spec, sp.tolist())
            print(f"\nvs IS-reweighted gross-idle spectrum (measured bins, no ansatz):")
            print(f"{'p':>10}  {'pooled S':>11}  {'IS reweight':>11}  {'S/IS':>7}")
            for i in range(0, len(sp), step):
                P = float(rw.P_logical[i])
                if P > 0:
                    print(f"{sp[i]:10.3e}  {pooled[i]:11.3e}  {P:11.3e}  {pooled[i]/P:7.2f}")

    out = d / "pooled.json"
    out.write_text(json.dumps(dict(sp=sp.tolist(), pooled=pooled.tolist(),
                                   ln_spread=spread.tolist(),
                                   n_shards=len(runs)), indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
