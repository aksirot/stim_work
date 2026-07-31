"""Decoder-selection loop on [[72,4,8]] full symmetric: Metropolis-generated failure
library -> grid+neighborhood bench -> paired regression gate -> promote -> repeat.

One iteration ~1 h, time-boxed phases:
  G  generate: fast-generator (ghw_nc1) harvest + strip + descent-only escalation
     (plateau escapes; NO direct sampling at onset weights — extensible to codes where
     f(onset) is unmeasurable). HARD REQUIREMENT: library always holds >=10 failing
     configs at w <= w_hat+1 where w_hat is the library's own achieved minimum.
  B  bench: every candidate (named grid + one-knob neighborhood of the incumbent)
     decodes the FULL accumulated library; ranking by fixes, then speed. Side product:
     per-decoder certified onset upper bound w_onset_ub = min{w : fails a library entry}.
  V  verify: top-2 challengers vs incumbent, PAIRED fresh samples at w=10/14 (McNemar);
     promotion needs strictly more library fixes AND no significant b10 excess.
  R  report: iteration JSON + printed tables. The loop RECOMMENDS a decoder; promoting
     into campaign DECODER_VARIANTS stays a human decision.

    python experiments/methods/decoder_loop.py --smoke        # ~5 min micro-iteration
    python experiments/methods/decoder_loop.py --iterate      # one ~1 h iteration
    python experiments/methods/decoder_loop.py --status

State under runs/decoder_loop/: library.json (accumulating; entries never leave —
permanent regression tests), state.json (incumbent, iteration), iter_<N>.json.
Launch iterations DETACHED (cmd /c + log redirect), never as harness-child background.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

for _k in ("EMC_DECODER", "EMC_DECODER_18", "EMC_DECODER_72", "EMC_RESULTS"):
    os.environ.pop(_k, None)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_error_model_comparison as rmc
from subonset_relay_sweep import CONFIGS as NAMED_GRID
from importance_sampling import _parse_dem, _expand
from repo_paths import REPO_ROOT
from scipy.stats import binomtest

OUT = REPO_ROOT / "runs" / "decoder_loop"
GEN_CFG = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw_nc1"])   # fast generator
GHW_CFG = dict(rmc.DEC_CFG, **rmc.DECODER_VARIANTS["ghw"])       # iteration-1 incumbent
MODEL = "full symmetric"
CENSUS_MIN = 10          # required entries at w <= w_hat+1
STABLE_ROUNDS = 3        # escalation stops after this many rounds with no lighter find


# --------------------------------------------------------------------------- setup
class Ctx:
    def __init__(self):
        self.circ = rmc.make_circuit72(MODEL, rmc.P_REF)
        self.calib = rmc.make_circuit72(MODEL, rmc.DECODER_P)
        probs, self.det, self.obs = _parse_dem(self.circ)
        self.c2m, self.q_base, _ = _expand(probs, None)
        self.N_exp = self.c2m.shape[0]
        self._dec_cache = {}

    def decoder(self, cfg: dict):
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in self._dec_cache:
            d = rmc.CalibratedRelayBP(self.calib, **{k: (tuple(v) if isinstance(v, list) else v)
                                                     for k, v in cfg.items()})
            d.setup(self.circ)
            self._dec_cache[key] = d
        return self._dec_cache[key]

    def syn_tru_of_mechs(self, mech_lists):
        B = len(mech_lists)
        syn = np.zeros((B, self.det.shape[1]), dtype=bool)
        tru = np.zeros((B, self.obs.shape[1]), dtype=bool)
        for i, ms in enumerate(mech_lists):
            syn[i] = np.bitwise_xor.reduce(self.det[list(ms)], axis=0)
            tru[i] = np.bitwise_xor.reduce(self.obs[list(ms)], axis=0)
        return syn, tru

    def sample(self, rng, w, B):
        idx = rng.integers(0, self.N_exp, size=(B, w))
        if w > 1:
            while True:
                s = np.sort(idx, axis=1)
                bad = (s[:, 1:] == s[:, :-1]).any(axis=1)
                if not bad.any():
                    break
                idx[bad] = rng.integers(0, self.N_exp, size=(int(bad.sum()), w))
        syn = np.bitwise_xor.reduce(self.det[self.c2m[idx]], axis=1)
        tru = np.bitwise_xor.reduce(self.obs[self.c2m[idx]], axis=1)
        return idx, syn, tru

    def support(self, expanded_cfg):
        mechs, counts = np.unique(self.c2m[list(expanded_cfg)], return_counts=True)
        return frozenset(int(m) for m in mechs[counts % 2 == 1])

    def fails(self, dec, supports):
        syn, tru = self.syn_tru_of_mechs([sorted(s) for s in supports])
        return np.any(dec.decode_batch(syn) != tru, axis=1)


# --------------------------------------------------------------------- library/state
def load_json(p, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def lib_load():
    return load_json(OUT / "library.json", {"entries": []})


def lib_save(lib):
    OUT.mkdir(parents=True, exist_ok=True)
    ws = [e["w"] for e in lib["entries"]]
    lib["w_min"] = min(ws) if ws else None
    lib["n"] = len(ws)
    (OUT / "library.json").write_text(json.dumps(lib, indent=1), encoding="utf-8")


def lib_add(lib, supports, generator, iteration):
    seen = {frozenset(e["mechs"]) for e in lib["entries"]}
    added = 0
    for s in supports:
        fs = frozenset(int(m) for m in s)
        if fs and fs not in seen:
            lib["entries"].append({"mechs": sorted(fs), "w": len(fs),
                                   "generator": generator, "iteration": iteration})
            seen.add(fs)
            added += 1
    return added


# ------------------------------------------------------------------------ phase G
def strip_one(ctx, dec, cfg_set, rng, decode_budget):
    """Greedy failure-preserving descent of one expanded config; returns (final, spent)."""
    cur = list(cfg_set)
    spent = 0
    while len(cur) > 2 and spent < decode_budget:
        cand = np.array([[c for c in cur if c != drop] for drop in cur], dtype=np.int64)
        syn = np.bitwise_xor.reduce(ctx.det[ctx.c2m[cand]], axis=1)
        tru = np.bitwise_xor.reduce(ctx.obs[ctx.c2m[cand]], axis=1)
        bad = np.any(dec.decode_batch(syn) != tru, axis=1)
        spent += len(cur)
        if not bad.any():
            break
        cur = [c for c in cur if c != cur[int(rng.choice(np.nonzero(bad)[0]))]]
    return set(cur), spent


def phase_generate(ctx, lib, iteration, deadline, smoke, incumbent_cfg, report):
    rng = np.random.default_rng([101, iteration])
    gen = ctx.decoder(GEN_CFG)
    per = 10 if smoke else 40
    strip_cap = 3_000 if smoke else 20_000

    # probe harvest weight: start w=16, adjust so failures are affordable
    w_h = 16
    _, syn, tru = ctx.sample(rng, w_h, 1_000)
    f = float(np.any(gen.decode_batch(syn) != tru, axis=1).mean())
    while f < 0.004 and w_h < 40 and time.time() < deadline:
        w_h += 4
        _, syn, tru = ctx.sample(rng, w_h, 1_000)
        f = float(np.any(gen.decode_batch(syn) != tru, axis=1).mean())
    print(f"[G] harvest weight w={w_h} (f_gen={f:.3g})", flush=True)

    def harvest(dec, w, want, cap_shots):
        got, shots, out = 0, 0, []
        while got < want and shots < cap_shots and time.time() < deadline:
            idx, syn, tru = ctx.sample(rng, w, 2_000)
            bad = np.any(dec.decode_batch(syn) != tru, axis=1)
            out += [set(map(int, r)) for r in idx[bad][: want - got]]
            got = len(out)
            shots += 2_000
        return out, shots

    raw, shots = harvest(gen, w_h, per, 200_000)
    print(f"[G] harvested {len(raw)} at w={w_h} in {shots} shots", flush=True)
    minimal = []
    spent = 0
    for c in raw:
        m, s = strip_one(ctx, gen, c, rng, max(500, (strip_cap - spent)))
        minimal.append(m)
        spent += s
    if minimal:
        sizes = sorted(len(ctx.support(m)) for m in minimal)
        print(f"[G] stripped: support weights {sizes[0]}..{sizes[-1]} ({spent} decodes)", flush=True)

    # escalation: plateau escapes until stabilization (descent-only, code-agnostic)
    pool = sorted(minimal, key=len)
    w_hat = min((len(ctx.support(m)) for m in pool), default=None)
    stable = 0
    esc_round = 0
    while stable < STABLE_ROUNDS and time.time() < deadline and pool:
        esc_round += 1
        found_lighter = False
        for base in sorted(pool, key=len)[:6]:
            for n_add in (1, 1, 2):
                c = set(base)
                for _ in range(n_add):
                    c.add(int(rng.integers(ctx.N_exp)))
                syn, tru = ctx.syn_tru_of_mechs([sorted(ctx.support(c))])
                if not np.any(gen.decode_batch(syn) != tru, axis=1)[0]:
                    continue
                m, _ = strip_one(ctx, gen, c, rng, 2_000)
                wm = len(ctx.support(m))
                pool.append(m)
                if w_hat is None or wm < w_hat:
                    w_hat = wm
                    found_lighter = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        stable = 0 if found_lighter else stable + 1
        print(f"[G] escalation round {esc_round}: w_hat={w_hat} "
              f"({'lighter found' if found_lighter else f'stable {stable}/{STABLE_ROUNDS}'})",
              flush=True)

    supports = [ctx.support(m) for m in pool]
    added = lib_add(lib, supports, "ghw_nc1", iteration)

    # incumbent-failure top-up (iteration >= 2)
    if iteration >= 2 and time.time() < deadline:
        inc = ctx.decoder(incumbent_cfg)
        raw_i, shots_i = harvest(inc, w_h, per // 2, 60_000)
        sup_i = []
        for c in raw_i:
            m, _ = strip_one(ctx, inc, c, rng, 1_500)
            sup_i.append(ctx.support(m))
        a2 = lib_add(lib, sup_i, "incumbent", iteration)
        print(f"[G] incumbent top-up: {len(raw_i)} harvested, {a2} added", flush=True)

    # census requirement: >=10 at w <= w_hat+1 (w_hat = library's own minimum)
    ws = [e["w"] for e in lib["entries"]]
    w_hat_lib = min(ws)
    census = sum(1 for w in ws if w <= w_hat_lib + 1)
    while census < CENSUS_MIN and time.time() < deadline:
        # sibling generation: plateau-replay at the floor
        floor_cfgs = [set(e["mechs"]) for e in lib["entries"] if e["w"] <= w_hat_lib + 1]
        base = floor_cfgs[int(rng.integers(len(floor_cfgs)))]
        c = set(base); c.add(int(rng.integers(ctx.N_exp)))  # expanded-vs-mech blur ok for escape
        syn, tru = ctx.syn_tru_of_mechs([sorted(c)])
        if np.any(gen.decode_batch(syn) != tru, axis=1)[0]:
            # strip in mechanism space directly via expanded identity mapping
            mset = sorted(c)
            cur = mset
            for _ in range(4):
                if len(cur) <= 2:
                    break
                cand = [[x for x in cur if x != d] for d in cur]
                syn2, tru2 = ctx.syn_tru_of_mechs(cand)
                bad = np.any(gen.decode_batch(syn2) != tru2, axis=1)
                if not bad.any():
                    break
                cur = cand[int(rng.choice(np.nonzero(bad)[0]))]
            lib_add(lib, [frozenset(cur)], "ghw_nc1", iteration)
        ws = [e["w"] for e in lib["entries"]]
        w_hat_lib = min(ws)
        census = sum(1 for w in ws if w <= w_hat_lib + 1)
    lib_save(lib)
    hist = {}
    for w in sorted(ws):
        hist[str(w)] = hist.get(str(w), 0) + 1
    report["generate"] = dict(harvest_weight=w_h, f_gen=f, added=added,
                              w_hat=w_hat_lib, census=census,
                              census_ok=bool(census >= CENSUS_MIN), w_hist=hist)
    print(f"[G] library: n={len(ws)}, w_hat={w_hat_lib}, census(<=w_hat+1)={census} "
          f"{'OK' if census >= CENSUS_MIN else 'NOT MET (deadline)'}", flush=True)


# ------------------------------------------------------------------------ phase B
def neighborhood(cfg):
    out = {}
    def put(name, **kw):
        c = dict(cfg, **kw)
        out[name] = c
    put("nb_pre_x2", pre_iter=int(cfg["pre_iter"] * 2))
    put("nb_pre_half", pre_iter=max(20, int(cfg["pre_iter"] // 2)))
    put("nb_g0_x2", gamma0=cfg["gamma0"] * 2)
    put("nb_g0_half", gamma0=cfg["gamma0"] / 2)
    put("nb_sets_x3", num_sets=int(cfg["num_sets"] * 3))
    put("nb_sets_third", num_sets=max(5, int(cfg["num_sets"] // 3)))
    put("nb_iter_x2", set_max_iter=int(cfg["set_max_iter"] * 2))
    lo, hi = cfg["gamma_dist_interval"]
    put("nb_int_wide", gamma_dist_interval=(lo * 1.5, hi * 1.5))
    put("nb_int_narrow", gamma_dist_interval=(lo * 0.67, hi * 0.67))
    put("nb_nconv_p4", stop_nconv=cfg["stop_nconv"] + 4)
    put("nb_nconv_m4", stop_nconv=max(1, cfg["stop_nconv"] - 4))
    return out


def phase_bench(ctx, lib, incumbent_cfg, deadline, smoke, report):
    cands = dict(NAMED_GRID)
    cands.update(neighborhood(incumbent_cfg))
    # dedup (by config content) and make sure the incumbent itself is present
    seen, uniq = {}, {}
    for name, cfg in [("incumbent", incumbent_cfg)] + list(cands.items()):
        key = json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                         sort_keys=True)
        if key not in seen:
            seen[key] = name
            uniq[name] = cfg
    if smoke:
        keep = ["incumbent", "baseline", "sets600_paper", "pre320", "nb_pre_x2", "nb_g0_half"]
        uniq = {k: v for k, v in uniq.items() if k in keep}
    supports = [frozenset(e["mechs"]) for e in lib["entries"]]
    ws = np.array([e["w"] for e in lib["entries"]])
    rng = np.random.default_rng(202)
    _, psyn, ptru = ctx.sample(rng, 12, 300 if smoke else 500)   # throughput probe batch
    rows = []
    for name, cfg in uniq.items():
        if time.time() > deadline and name != "incumbent":
            print(f"[B] deadline — skipping remaining candidates from {name}", flush=True)
            break
        dec = ctx.decoder(cfg)
        bad = ctx.fails(dec, supports)
        t0 = time.time()
        dec.decode_batch(psyn)
        rate = len(psyn) / max(time.time() - t0, 1e-9)
        w_ub = int(ws[bad].min()) if bad.any() else None
        rows.append(dict(name=name, fixes=int((~bad).sum()), fails=int(bad.sum()),
                         n=len(supports), rate=round(rate, 1),
                         w_onset_ub=w_ub, cfg=cfg))
        print(f"[B] {name:16s} fixes {rows[-1]['fixes']:4d}/{len(supports)}  "
              f"{rate:8,.0f} dec/s  onset_ub={'>' + str(int(ws.max())) if w_ub is None else w_ub}",
              flush=True)
    rows.sort(key=lambda r: (-r["fixes"], -r["rate"]))
    report["bench"] = dict(n_candidates=len(rows), library_n=len(supports),
                           ranking=[{k: v for k, v in r.items() if k != "cfg"} for r in rows])
    return rows


# ------------------------------------------------------------------------ phase V
def phase_verify(ctx, rows, incumbent_cfg, deadline, smoke, report):
    inc_key = json.dumps({k: list(v) if isinstance(v, tuple) else v
                          for k, v in incumbent_cfg.items()}, sort_keys=True)
    inc_row = next(r for r in rows if json.dumps(
        {k: list(v) if isinstance(v, tuple) else v for k, v in r["cfg"].items()},
        sort_keys=True) == inc_key)
    challengers = [r for r in rows if r is not inc_row and r["fixes"] > inc_row["fixes"]][:2]
    verdicts = []
    inc = ctx.decoder(incumbent_cfg)
    rng = np.random.default_rng(303)
    T = 800 if smoke else 5_000
    weights = [10] if smoke else [10, 14]
    for ch in challengers:
        cd = ctx.decoder(ch["cfg"])
        b01 = b10 = f_i = f_c = n = 0
        for w in weights:
            done = 0
            while done < T and time.time() < deadline:
                B = min(1_000, T - done)
                _, syn, tru = ctx.sample(rng, w, B)
                xi = np.any(inc.decode_batch(syn) != tru, axis=1)
                xc = np.any(cd.decode_batch(syn) != tru, axis=1)
                b01 += int((xi & ~xc).sum()); b10 += int((~xi & xc).sum())
                f_i += int(xi.sum()); f_c += int(xc.sum())
                done += B; n += B
        # regression gate: is b10 significantly above b01?
        p = binomtest(b10, b10 + b01, 0.5, alternative="greater").pvalue if (b10 + b01) else 1.0
        ok = p > 0.05
        verdicts.append(dict(name=ch["name"], shots=n, inc_fails=f_i, ch_fails=f_c,
                             b01=b01, b10=b10, p_regress=round(float(p), 4), passes=bool(ok)))
        print(f"[V] {ch['name']:16s} paired {n} shots: inc {f_i} vs ch {f_c}  "
              f"b01={b01} b10={b10} p={p:.3f} -> {'PASS' if ok else 'REJECT'}", flush=True)
    winner = None
    for v in verdicts:
        if v["passes"]:
            winner = v["name"]
            break
    report["verify"] = dict(challengers=verdicts, promoted=winner)
    return winner, {r["name"]: r["cfg"] for r in rows}


# ------------------------------------------------------------------------- driver
def iterate(smoke=False):
    ctx = Ctx()
    OUT.mkdir(parents=True, exist_ok=True)
    state = load_json(OUT / "state.json",
                      {"iteration": 0, "incumbent_name": "ghw",
                       "incumbent_cfg": {k: list(v) if isinstance(v, tuple) else v
                                         for k, v in GHW_CFG.items()}})
    itn = state["iteration"] + 1
    inc_cfg = {k: tuple(v) if isinstance(v, list) else v
               for k, v in state["incumbent_cfg"].items()}
    t0 = time.time()
    boxes = (3, 5, 8) if smoke else (18, 12, 28)     # minutes for G, B, V
    report = dict(iteration=itn, incumbent=state["incumbent_name"], smoke=smoke)
    lib = lib_load()
    print(f"=== iteration {itn} (incumbent {state['incumbent_name']}; "
          f"library n={len(lib['entries'])}) ===", flush=True)
    phase_generate(ctx, lib, itn, t0 + boxes[0] * 60, smoke, inc_cfg, report)
    rows = phase_bench(ctx, lib, inc_cfg, t0 + (boxes[0] + boxes[1]) * 60, smoke, report)
    winner, cfg_of = phase_verify(ctx, rows, inc_cfg,
                                  t0 + (boxes[0] + boxes[1] + boxes[2]) * 60, smoke, report)
    if winner:
        state["incumbent_name"] = winner
        state["incumbent_cfg"] = {k: list(v) if isinstance(v, tuple) else v
                                  for k, v in cfg_of[winner].items()}
        print(f"[R] PROMOTED: {winner} is the new incumbent", flush=True)
    else:
        print(f"[R] incumbent {state['incumbent_name']} retained", flush=True)
    state["iteration"] = itn
    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / f"iter_{itn}.json").write_text(json.dumps(report, indent=1, default=str),
                                          encoding="utf-8")
    (OUT / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"[R] wrote iter_{itn}.json ({report['elapsed_s']:,.0f}s)", flush=True)


def status():
    state = load_json(OUT / "state.json", None)
    lib = lib_load()
    if not state:
        print("no loop state yet")
        return
    ws = [e["w"] for e in lib["entries"]]
    print(f"iteration {state['iteration']}, incumbent {state['incumbent_name']}")
    if ws:
        print(f"library: {len(ws)} entries, w_min={min(ws)}, "
              f"census(<=w_min+1)={sum(1 for w in ws if w <= min(ws) + 1)}")
    for f in sorted(OUT.glob("iter_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        v = j.get("verify", {})
        print(f"  {f.stem}: promoted={v.get('promoted')}  "
              f"w_hat={j.get('generate', {}).get('w_hat')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iterate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.iterate or a.smoke:
        iterate(smoke=a.smoke)
    else:
        ap.error("--iterate, --smoke or --status")


if __name__ == "__main__":
    main()
