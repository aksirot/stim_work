"""Replica-exchange splitting v2 — ergodicity-hardened (2026-08-01).

The v1 ladder (`splitting.replica_exchange_estimate`) passed its internal gates yet
sat orders of magnitude below reweighted IS on the [[72,4,8]] full-symmetric model
(direct-MC arbitration: 14 observed vs 0.3 predicted at p=2e-3 —
runs/splitting_crosscheck/VERDICT_72_full_ghw.md). Root cause: ergodicity across
failure basins was assumed, never engineered — local single-column moves cannot cross
the non-failing sea between basins, so the walker population collapses into the seeded
basin and the rung-ratio chain prices that basin only (biased LOW, compounding
down-ladder). v2 attacks each cause:

* seed DIVERSITY: caller passes a pool of independent failing supports (e.g. the
  decoder-loop failure library) — walkers start spread over many basins;
* COSET-JUMP moves: XOR a zero-syndrome element (x ^ correction(x), built from the
  seed pool at setup, plus any known logicals) onto a replica — a symmetric Metropolis
  proposal (deterministic first-column toggle = exact involution) that teleports
  between basins without leaving the syndrome class;
* INDEPENDENT LADDERS: n_ladders full replicas run interleaved — their proposals share
  decode batches, so wall-cost ~ one big ladder — and the spread of their independent
  log-P estimates is the honesty metric internal gates cannot fake;
* the caller must still arbitrate against IS/direct MC in the measurable regime; v2
  reports per-ladder curves to make that comparison first-class.
"""
from __future__ import annotations

import itertools
import time
from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import stim
from scipy.special import logsumexp

from importance_sampling import _parse_dem, _expand
from splitting import (SplittingResult, _config_syndrome_truth, _config_fails,
                       _mech_to_cols, _direct_mc_failure_prob, min_weight_logical_seeds)


def _zero_syndrome_pool(pool, det_mat, obs_mat, col_to_mech, decoder,
                        extra_logicals=None, max_weight=48):
    """Zero-syndrome mechanism supports for coset-jump moves.

    For each failing config x in the pool: decode its syndrome to a correction y
    (mechanism space); the mechanism-parity of x XOR y is syndrome-free. Jumping
    along it lands in the same syndrome class — failing status is re-verified by
    decode at proposal time.
    """
    elems, seen = [], set()
    for L in (extra_logicals or []):
        key = frozenset(int(m) for m in L)
        if key and key not in seen and len(key) <= max_weight:
            seen.add(key); elems.append(sorted(key))
    if pool and hasattr(decoder, "decode_corrections_batch"):
        M = det_mat.shape[1]
        synd = np.zeros((len(pool), M), dtype=bool)
        for k, cfg in enumerate(pool):
            synd[k], _ = _config_syndrome_truth(det_mat, obs_mat, col_to_mech, cfg)
        corr = np.asarray(decoder.decode_corrections_batch(synd))
        for k, cfg in enumerate(pool):
            x_par = {}
            for c in cfg:
                m = int(col_to_mech[c]); x_par[m] = x_par.get(m, 0) ^ 1
            y = set(int(m) for m in np.nonzero(corr[k])[0])
            L = {m for m, p in x_par.items() if p} ^ y
            key = frozenset(L)
            if key and key not in seen and len(key) <= max_weight:
                seen.add(key); elems.append(sorted(key))
    return elems


def replica_exchange_v2(
    circuit: stim.Circuit,
    decoder,
    *,
    p_ref: float,
    p_high: float,
    p_low: float,
    n_levels: int,
    n_ladders: int = 3,
    n_walkers: int = 8,
    local_steps: int = 3,
    n_sweeps: int = 50,
    burn_in: int = 15,
    thin: int = 1,
    anchor_shots: int = 4000,
    q_base: Optional[float] = None,
    distance: Optional[int] = None,
    seed: Optional[int] = None,
    seed_supports=None,
    extra_logicals=None,
    seed_p_grid: Optional[Sequence[float]] = None,
    gap_weights: Optional[Sequence[int]] = None,
    jump_every: int = 1,
    pool_cache=None,
    cache_key: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[SplittingResult, dict]:
    """Ergodicity-hardened replica-exchange splitting (see module docstring).

    seed_supports: iterable of mechanism-index supports (e.g. decoder-loop library
    entries that fail under `decoder`) — the diversity backbone of the pool.
    extra_logicals: known zero-syndrome mechanism supports (onset-hunt logicals).
    Returns (SplittingResult for the ladder-MEAN curve, diagnostics incl. per-ladder
    log10 curves and their spread — quote nothing whose spread you would not bar).
    """
    rng = np.random.default_rng(seed)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, q_base)
    decoder.setup(circuit)
    N_exp = col_to_mech.shape[0]
    L = n_levels
    m2c = _mech_to_cols(col_to_mech)

    p_ladder = np.geomspace(p_high, p_low, L + 1)
    q_ladder = np.clip(q_base * (p_ladder / p_ref), 1e-300, 1.0 - 1e-15)
    logodds = np.log(q_ladder) - np.log1p(-q_ladder)
    log_r_next = np.log(q_ladder[1:]) - np.log(q_ladder[:-1])
    log_r_1m = np.log1p(-q_ladder[1:]) - np.log1p(-q_ladder[:-1])

    P_high, P_high_se, mc_seeds = _direct_mc_failure_prob(
        det_mat, obs_mat, col_to_mech, decoder, float(q_ladder[0]), anchor_shots, rng)
    if P_high <= 0.0:
        raise ValueError(f"direct MC at p_high={p_high:g} saw no failures; "
                         f"raise p_high/anchor_shots")

    # --- pool cache: seed verification + jump-pool construction cost minutes of
    #     slow low-weight decodes per run; identical (decoder, DEM, supports) reuse it ---
    import json as _json, pathlib as _pl
    _ckey = dict(cache_key or {}, N_exp=int(N_exp),
                 n_supports=len(list(seed_supports)) if seed_supports else 0)
    pool: List[FrozenSet[int]] = []
    jumps: Optional[List[List[int]]] = None
    if pool_cache is not None:
        _cp = _pl.Path(pool_cache)
        if _cp.exists():
            _c = _json.loads(_cp.read_text(encoding="utf-8"))
            if _c.get("key") == _ckey:
                pool = [frozenset(int(x) for x in cfg) for cfg in _c["pool"]]
                jumps = [[int(m) for m in j] for j in _c["jumps"]]
                if verbose:
                    print(f"  [v2] pool cache HIT ({_cp.name}): {len(pool)} seeds, "
                          f"{len(jumps)} jumps", flush=True)
            elif verbose:
                print(f"  [v2] pool cache STALE (key mismatch) — rebuilding", flush=True)

    cached = jumps is not None
    if not cached:
        # --- seed pool: caller supports (diversity backbone) + the usual sources ---
        if seed_supports:
            supports = [frozenset(int(m) for m in s) for s in seed_supports]
            lifted = min_weight_logical_seeds(circuit, col_to_mech, det_mat, obs_mat,
                                              decoder, supports=supports,
                                              seed=int(rng.integers(2**31)))
            pool += lifted
            if verbose:
                print(f"  [v2] seed pool: {len(lifted)}/{len(supports)} caller supports "
                      f"verified failing", flush=True)
        # BP-OSD min-weight search ONLY as a fallback when the caller brings no seeds —
        # on big DEMs it is the 95-minutes-silent trap; the library pool replaces it.
        if not pool:
            pool += min_weight_logical_seeds(circuit, col_to_mech, det_mat, obs_mat,
                                             decoder, distance=distance,
                                             seed=int(rng.integers(2**31)))
        if gap_weights and pool:
            base = [set(s) for s in sorted(pool, key=len)[:30]]
            n_inf = 0
            for wt in gap_weights:
                got = 0
                for _ in range(40):
                    if got >= 20:
                        break
                    s = set(base[int(rng.integers(len(base)))])
                    while len(s) < wt:
                        s.add(int(rng.integers(N_exp)))
                    if _config_fails(det_mat, obs_mat, col_to_mech, s, decoder):
                        pool.append(frozenset(s)); got += 1; n_inf += 1
            if verbose:
                print(f"  [v2] inflated seeds to gap weights {list(gap_weights)} "
                      f"(+{n_inf})", flush=True)
        for p_s in (seed_p_grid or []):
            q_s = float(np.clip(q_base * (p_s / p_ref), 1e-300, 1.0 - 1e-15))
            _, _, s = _direct_mc_failure_prob(det_mat, obs_mat, col_to_mech, decoder,
                                              q_s, anchor_shots, rng)
            pool += s
        if not pool and not mc_seeds:
            raise ValueError("no failing seed configs found")

        jumps = _zero_syndrome_pool(pool + mc_seeds, det_mat, obs_mat, col_to_mech,
                                    decoder, extra_logicals=extra_logicals)
        if verbose:
            jl = [len(j) for j in jumps]
            print(f"  [v2] jump pool: {len(jumps)} zero-syndrome elements "
                  f"(weights {min(jl) if jl else 0}..{max(jl) if jl else 0})", flush=True)
        if pool_cache is not None:
            _pl.Path(pool_cache).write_text(_json.dumps(dict(
                key=_ckey, pool=[sorted(int(x) for x in cfg) for cfg in pool],
                jumps=jumps)), encoding="utf-8")
            if verbose:
                print(f"  [v2] pool cache written ({len(pool)} seeds, {len(jumps)} "
                      f"jumps)", flush=True)

    pool += mc_seeds        # fresh anchor typicals from THIS run (never cached)
    if not pool:
        raise ValueError("no failing seed configs found")

    # --- walkers: independent ladders; per-rung seeds drawn without replacement
    #     from the weight-matched pool neighborhood where possible ---
    pool_sorted = sorted(pool, key=len)
    npool = len(pool_sorted)

    def seed_for_rung(i, used):
        center = (1.0 - i / L) * (npool - 1)
        lo = max(0, int(center) - 4); hi = min(npool - 1, int(center) + 4)
        cands = [k for k in range(lo, hi + 1) if k not in used] or list(range(lo, hi + 1))
        k = int(cands[int(rng.integers(len(cands)))])
        used.add(k)
        return set(pool_sorted[k])

    replicas = []
    for _lad in range(n_ladders):
        used: set = set()
        replicas.append([[seed_for_rung(i, used) for i in range(L + 1)]
                         for _ in range(n_walkers)])

    swap_attempts = np.zeros((n_ladders, L)); swap_accepts = np.zeros((n_ladders, L))
    jump_attempts = jump_pre = jump_accepts = 0
    walker_terms = [[[[] for _ in range(L)] for _ in range(n_walkers)]
                    for _ in range(n_ladders)]
    wsum = np.zeros(L + 1); wcount = 0
    min_cfg = min(pool, key=len); w_min_seen = len(min_cfg)

    M = det_mat.shape[1]; K = obs_mat.shape[1]
    _t0 = time.time()
    if verbose:
        print(f"  [v2] running {n_sweeps} sweeps x {n_ladders} ladders x {n_walkers} "
              f"walkers x {L+1} rungs (local {local_steps}/sweep + jumps)", flush=True)

    def decode_cands(cands):
        synd = np.zeros((len(cands), M), dtype=bool)
        truth = np.zeros((len(cands), K), dtype=bool)
        for k, (_lad, _w, _i, cand) in enumerate(cands):
            synd[k], truth[k] = _config_syndrome_truth(det_mat, obs_mat, col_to_mech, cand)
        preds = decoder.decode_batch(synd)
        return np.any(preds != truth, axis=1)

    for sweep in range(n_sweeps):
        if verbose and (sweep % 5 == 0 or sweep == n_sweeps - 1):
            _el = time.time() - _t0
            _eta = _el / max(sweep, 1) * (n_sweeps - sweep)
            _d = ""
            if swap_attempts.sum() > 0:
                _sa = swap_accepts / np.maximum(swap_attempts, 1)
                _d = f"  swap {_sa.min():.2f}-{_sa.max():.2f}"
                if jump_attempts:
                    _d += f"  jump-acc {jump_accepts/max(jump_attempts,1):.3f}"
                if wcount > 0:
                    _mw = wsum / max(wcount * n_walkers * n_ladders, 1)
                    _d += f"  mean-w {_mw[0]:.0f}->{_mw[-1]:.0f}"
            print(f"  [v2] sweep {sweep}/{n_sweeps}  ({_el:.0f}s, ETA {_eta:.0f}s){_d}",
                  flush=True)

        # (a) local moves — one decode batch per sub-step across ALL ladders
        for _sub in range(local_steps):
            cands = []
            for lad in range(n_ladders):
                for w in range(n_walkers):
                    for i in range(L + 1):
                        cur = replicas[lad][w][i]; nx = len(cur); lo = float(logodds[i])
                        if nx == 0 or rng.random() < 0.5:
                            c = int(rng.integers(N_exp))
                            while c in cur:
                                c = int(rng.integers(N_exp))
                            log_acc = lo + np.log(N_exp - nx) - np.log(nx + 1)
                            adding = True
                        else:
                            c = next(itertools.islice(cur, int(rng.integers(nx)), None))
                            log_acc = -lo + np.log(nx) - np.log(N_exp - nx + 1)
                            adding = False
                        if np.log(rng.random()) < min(0.0, log_acc):
                            cand = set(cur)
                            cand.add(c) if adding else cand.discard(c)
                            cands.append((lad, w, i, cand))
            if cands:
                fails = decode_cands(cands)
                for k, (lad, w, i, cand) in enumerate(cands):
                    if fails[k]:
                        replicas[lad][w][i] = cand
                        if len(cand) < w_min_seen:
                            w_min_seen = len(cand); min_cfg = frozenset(cand)

        # (a') coset jumps — deterministic first-column toggle => exact involution
        #      (symmetric proposal); weight pre-accept, then failing re-verify
        if jumps and (sweep % jump_every == 0):
            cands = []
            for lad in range(n_ladders):
                for w in range(n_walkers):
                    for i in range(L + 1):
                        jump_attempts += 1
                        Lm = jumps[int(rng.integers(len(jumps)))]
                        cur = replicas[lad][w][i]
                        cand = set(cur)
                        for m in Lm:
                            c0 = m2c[m][0]
                            if c0 in cand:
                                cand.discard(c0)
                            else:
                                cand.add(c0)
                        dlog = (len(cand) - len(cur)) * float(logodds[i])
                        if np.log(rng.random()) < min(0.0, dlog):
                            jump_pre += 1
                            cands.append((lad, w, i, cand))
            if cands:
                fails = decode_cands(cands)
                for k, (lad, w, i, cand) in enumerate(cands):
                    if fails[k]:
                        jump_accepts += 1
                        replicas[lad][w][i] = cand
                        if len(cand) < w_min_seen:
                            w_min_seen = len(cand); min_cfg = frozenset(cand)

        # (b) swaps per ladder — no decode
        for lad in range(n_ladders):
            for parity in (0, 1):
                for i in range(parity, L, 2):
                    d_lo = float(logodds[i] - logodds[i + 1])
                    for w in range(n_walkers):
                        xi, xj = replicas[lad][w][i], replicas[lad][w][i + 1]
                        swap_attempts[lad][i] += 1
                        if np.log(rng.random()) < min(0.0, (len(xj) - len(xi)) * d_lo):
                            replicas[lad][w][i], replicas[lad][w][i + 1] = xj, xi
                            swap_accepts[lad][i] += 1

        # (c) collect
        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            wcount += 1
            for lad in range(n_ladders):
                for w in range(n_walkers):
                    for i in range(L + 1):
                        wsum[i] += len(replicas[lad][w][i])
                    for i in range(L):
                        wt = len(replicas[lad][w][i])
                        walker_terms[lad][w][i].append(
                            wt * log_r_next[i] + (N_exp - wt) * log_r_1m[i])

    # --- per-ladder estimates; headline = ladder mean; spread = the honesty bar ---
    log_P_ladders = []
    for lad in range(n_ladders):
        comb = np.zeros(L)
        for i in range(L):
            terms = np.concatenate([np.asarray(walker_terms[lad][w][i])
                                    for w in range(n_walkers) if walker_terms[lad][w][i]])
            comb[i] = float(logsumexp(terms) - np.log(terms.size)) if terms.size else -np.inf
        log_P_ladders.append(np.log(max(P_high, 1e-300)) +
                             np.concatenate([[0.0], np.cumsum(comb)]))
    log_P_arr = np.vstack(log_P_ladders)
    log_P_mean = log_P_arr.mean(axis=0)
    ladder_spread = log_P_arr.std(axis=0, ddof=1) if n_ladders > 1 else np.zeros(L + 1)
    P_logical = np.exp(log_P_mean)
    rel = np.sqrt((P_high_se / P_high) ** 2 +
                  (ladder_spread / max(np.sqrt(n_ladders), 1)) ** 2)

    sa = swap_accepts / np.maximum(swap_attempts, 1)
    diag = {"swap_accept": sa.min(axis=0).tolist(),          # worst ladder per pair
            "swap_accept_per_ladder": sa.tolist(),
            "mean_weight": (wsum / max(wcount * n_walkers * n_ladders, 1)).tolist(),
            "P_high": float(P_high), "n_pool": len(pool), "n_jumps": len(jumps),
            "jump_attempt": int(jump_attempts), "jump_pre_accept": int(jump_pre),
            "jump_accept": int(jump_accepts), "n_collect": int(wcount),
            "log10_P_ladders": (log_P_arr / np.log(10)).tolist(),
            "ladder_spread_log10": (ladder_spread / np.log(10)).tolist(),
            "w_min_seen": int(w_min_seen),
            "min_config_mechs": sorted({int(col_to_mech[c]) for c in min_cfg})}
    if verbose:
        print(f"  [v2] jump acceptance: {jump_accepts}/{jump_attempts} "
              f"(pre-accept {jump_pre})", flush=True)
        print(f"  [v2] ladder spread (log10) at p_low: "
              f"{diag['ladder_spread_log10'][-1]:.2f}", flush=True)
    return SplittingResult(
        q_ladder=q_ladder, p_ladder=p_ladder, P_logical=P_logical,
        P_logical_se=P_logical * rel, log_ratios=np.zeros(L), log_ratios_se=np.zeros(L),
        P_high=P_high, P_high_se=P_high_se, per_level_chains=[],
        n_seeds_used=n_ladders * n_walkers, q_base=q_base, p_ref=p_ref, n_expanded=N_exp,
    ), diag
