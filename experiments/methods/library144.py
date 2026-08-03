"""Failure library for the [[144,12,12]] bare gross memory (lpu_idle circuit).

Same design as the 72-code decoder-loop library: fast-generator harvest at the
lowest affordable weight, lockstep strip, plateau-escape descent until stabilization,
HARD census requirement of >=10 entries at w <= w_hat+1 where w_hat is the library's
own achieved minimum (descent-only, code-agnostic — the paper's circuit distance 10
/ onset w0=5 is a DIAGNOSTIC to print, never an input; sub-onset w=4 failures are
expected if the decoder is imperfect and would be the headline find).

Generator/incumbent: the campaign relay (num_sets=20, nconv=5 — the fast decoder on
this circuit); priors campaign-style from the p_ref circuit (the idle circuit is its
own device). The camp ladder's w=9 specimen seeds the pool if it verifies.

Boxed by LIB144_BOX_H hours (default 6); resumable — the library accumulates.
Output: runs/decoder_loop144/library.json (+ log)
"""
import json
import os
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from repo_paths import REPO_ROOT
from experiment_runner import load_config, build_circuit, make_decoder
from bb_code_sim import RelayBPDecoder
from importance_sampling import _parse_dem, _expand
from decoder_loop import lockstep_strip

OUT = REPO_ROOT / "runs" / "decoder_loop144"
OUT.mkdir(parents=True, exist_ok=True)
BOX_H = float(os.environ.get("LIB144_BOX_H", "6"))
CENSUS_MIN = 10
STABLE_ROUNDS = 3


class Ctx144:
    """decoder_loop.Ctx interface over the bare gross memory circuit."""

    def __init__(self):
        self.cfg = load_config(str(REPO_ROOT / "experiments" / "configs" /
                                   "gross_lpu_idle.yaml"))
        self.circ = build_circuit(self.cfg)
        probs, self.det, self.obs = _parse_dem(self.circ)
        self.c2m, self.q_base, _ = _expand(probs, None)
        self.N_exp = self.c2m.shape[0]
        self._dec_cache = {}

    def decoder(self, key_cfg):
        key = json.dumps(key_cfg, sort_keys=True, default=str)
        if key not in self._dec_cache:
            if key_cfg == "campaign":
                d = make_decoder(self.cfg)
            else:
                d = RelayBPDecoder(**{k: (tuple(v) if isinstance(v, list) else v)
                                      for k, v in key_cfg.items()})
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

    def lift(self, mech_support):
        return set(int(c) for c in np.searchsorted(self.c2m, sorted(mech_support)))

    def fails(self, dec, supports):
        syn, tru = self.syn_tru_of_mechs([sorted(s) for s in supports])
        return np.any(dec.decode_batch(syn) != tru, axis=1)


def main():
    deadline = time.time() + BOX_H * 3600
    ctx = Ctx144()
    gen = ctx.decoder("campaign")
    rng = np.random.default_rng(1441)
    print(f"144 library: N_exp={ctx.N_exp}, box {BOX_H}h", flush=True)

    lib_path = OUT / "library.json"
    lib = (json.loads(lib_path.read_text(encoding="utf-8")) if lib_path.exists()
           else {"entries": []})
    have = {frozenset(e["mechs"]) for e in lib["entries"]}

    def lib_add(supports, generator):
        n = 0
        for s in supports:
            key = frozenset(int(m) for m in s)
            if key and key not in have:
                lib["entries"].append(dict(mechs=sorted(key), w=len(key),
                                           generator=generator, iteration=1))
                have.add(key); n += 1
        return n

    def lib_save():
        ws = [e["w"] for e in lib["entries"]]
        lib["n"] = len(ws); lib["w_min"] = min(ws) if ws else None
        lib_path.write_text(json.dumps(lib), encoding="utf-8")

    # seed: the camp ladder's w=9 specimen, if it verifies under the generator
    camp = json.loads((REPO_ROOT / "runs" / "splitting_crosscheck" /
                       "lpu_idle_camp.json").read_text(encoding="utf-8"))
    spec = camp.get("min_config_mechs")
    if spec and ctx.fails(gen, [spec])[0]:
        n = lib_add([spec], "camp_ladder")
        print(f"[seed] camp w=9 specimen verified failing (+{n})", flush=True)

    # LIB144_CENSUS_ONLY=1: skip harvest/strip (box 1 showed stripping w~73 harvests
    # costs ~1.8k decodes per weight step — days to descend); grow the low-weight
    # census outward from the existing floor entries instead (escapes cost tens of
    # decodes per attempt at w~9).
    census_only = os.environ.get("LIB144_CENSUS_ONLY") == "1"

    # harvest weight: climb from 40 until f_gen affordable
    w_h, f = 40, 0.0
    if census_only:
        w_h, f = 0, 0.0
        raw, minimal, shots = [], [], 0
        print("[G] census-only mode: skipping harvest/strip", flush=True)
    if not census_only:
        while f < 0.004 and w_h < 140 and time.time() < deadline:
            _, syn, tru = ctx.sample(rng, w_h, 500)
            f = float(np.any(gen.decode_batch(syn) != tru, axis=1).mean())
            if f < 0.004:
                w_h += 10
        print(f"[G] harvest weight w={w_h} (f_gen={f:.3g})", flush=True)

        got, shots, raw = 0, 0, []
        while got < 25 and shots < 100_000 and time.time() < deadline:
            idx, syn, tru = ctx.sample(rng, w_h, 1_000)
            bad = np.any(gen.decode_batch(syn) != tru, axis=1)
            raw += [set(map(int, r)) for r in idx[bad][: 25 - got]]
            got = len(raw); shots += 1_000
            if shots % 5_000 == 0:
                print(f"  [harvest] {shots} shots, {got} configs", flush=True)
        print(f"[G] harvested {len(raw)} at w={w_h} in {shots} shots", flush=True)

        minimal = lockstep_strip(ctx, gen, raw, rng, deadline)
        if minimal:
            sizes = sorted(len(ctx.support(m)) for m in minimal)
            print(f"[G] stripped: support weights {sizes[0]}..{sizes[-1]}", flush=True)
        lib_add([ctx.support(m) for m in minimal], "campaign_harvest")
        lib_save()

    def escape_cycle(pool_cfgs, n_escapes, depth=None):
        bases = sorted(pool_cfgs, key=len)[:max(3, n_escapes // 3)]
        if not bases:
            return []
        cands = []
        for k in range(n_escapes):
            c = set(bases[k % len(bases)])
            n_add = (1 + (k % depth) if depth else (1 if k % 3 else 2))
            for _ in range(n_add):
                c.add(int(rng.integers(ctx.N_exp)))
            cands.append(c)
        syn = np.zeros((len(cands), ctx.det.shape[1]), dtype=bool)
        tru = np.zeros((len(cands), ctx.obs.shape[1]), dtype=bool)
        for i, c in enumerate(cands):
            arr = np.array(sorted(c))
            syn[i] = np.bitwise_xor.reduce(ctx.det[ctx.c2m[arr]], axis=0)
            tru[i] = np.bitwise_xor.reduce(ctx.obs[ctx.c2m[arr]], axis=0)
        bad = np.any(gen.decode_batch(syn) != tru, axis=1)
        survivors = [cands[i] for i in np.nonzero(bad)[0]]
        return lockstep_strip(ctx, gen, survivors, rng, deadline) if survivors else []

    pool = list(minimal) + [ctx.lift(e["mechs"]) for e in lib["entries"]]
    w_hat = min((len(ctx.support(m)) for m in pool), default=None)
    stable, rnd = 0, 0
    # LIB144_PUSH=1: plateau-BREAKING mode — ignore the stabilization stop and spend the
    # whole box on escapes whose perturbation GROWS with the stall length (add 1..6
    # faults, re-strip). The default stabilization rule exits in minutes once 3 rounds
    # find nothing lighter, which is the right call for a converged descent but the
    # wrong one when the question is whether a plateau can be crossed at all.
    push = os.environ.get("LIB144_PUSH") == "1"
    while (push or stable < STABLE_ROUNDS) and time.time() < deadline and pool:
        rnd += 1
        new = escape_cycle(pool, 15, depth=min(6, 1 + stable) if push else None)
        lib_add([ctx.support(m) for m in new], "campaign_escape")
        pool += new
        w_new = min((len(ctx.support(m)) for m in pool), default=None)
        lighter = w_new is not None and (w_hat is None or w_new < w_hat)
        w_hat = w_new if lighter else w_hat
        stable = 0 if lighter else stable + 1
        lib_save()
        print(f"[G] escalation {rnd}: w_hat={w_hat} "
              f"({'lighter' if lighter else f'stable {stable}/{STABLE_ROUNDS}'})", flush=True)

    ws = [e["w"] for e in lib["entries"]]
    w_lib = min(ws)
    census = sum(1 for w in ws if w <= w_lib + 1)
    while census < CENSUS_MIN and time.time() < deadline:
        floor = [ctx.lift(e["mechs"]) for e in lib["entries"] if e["w"] <= w_lib + 2]
        new = escape_cycle(floor, 12)
        lib_add([ctx.support(m) for m in new], "campaign_census")
        ws = [e["w"] for e in lib["entries"]]
        w_lib = min(ws); census = sum(1 for w in ws if w <= w_lib + 1)
        lib_save()
        print(f"[G] census cycle: w_hat={w_lib}, census={census}", flush=True)

    lib_save()
    hist = {}
    for w in sorted(ws):
        hist[w] = hist.get(w, 0) + 1
    print(f"RESULT: library n={len(ws)}, w_hat={w_lib}, census(<=w_hat+1)={census} "
          f"{'OK' if census >= CENSUS_MIN else 'NOT MET (box)'}", flush=True)
    print(f"  weight hist: {hist}", flush=True)
    print(f"  diagnostic: paper circuit distance 10 -> w0=5; entries below 5 are "
          f"SUB-ONSET decoder failures", flush=True)


if __name__ == "__main__":
    main()
