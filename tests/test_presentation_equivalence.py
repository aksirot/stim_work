"""G3a: the LPU paper's gross-code presentation IS Bravyi's gross code (campaign top risk).

`gross_code_lpu_tdg.py` builds circuits with A = 1+y+x³y⁻¹, B = 1+x+x⁻¹y⁻³ while the memory
campaign runs BB_144_12_12 with A = x³+y+y², B = y³+x+x². These tests pin the exact relation:

    A_tdg = y⁻¹ · A_bravyi          B_tdg = x⁻¹ · B_bravyi

and, because the group algebra F₂[Z_l × Z_m] is ABELIAN, a monomial multiple of A (resp. B) is a
pure relabeling of the L (resp. R) data-qubit block:  [mA | m'B] = [A | B] · blockdiag(P_m, P_m').
So the two presentations have IDENTICAL stabilizer groups after the explicit data permutation

    L block: (i, j) -> column of y⁻¹-translation      R block: (i, j) -> column of x⁻¹-translation

verified below for both H_X and H_Z row spaces simultaneously (same permutation!) — full
stabilizer-group equality, not just matching (n, k). Passing = LPU↔memory Λ ratios are
same-code by proof; the W1b quarantine lifts.
"""
import numpy as np
import pytest

from bb_code_sim import (BB_144_12_12, BB_144_TDG, BBCodeParams, build_parity_checks,
                         find_logical_ops, _gf2_rank, _gf2_rref)
import gross_code_lpu_tdg as tdg


def monomial_perm(l: int, m: int, s: int, t: int) -> np.ndarray:
    """Column permutation of the (i,j) -> i*m+j block index under translation by x^s y^t."""
    n = l * m
    P = np.zeros((n, n), dtype=np.uint8)
    for i in range(l):
        for j in range(m):
            P[i * m + j, ((i + s) % l) * m + ((j + t) % m)] = 1
    return P


def rowspace_sig(M: np.ndarray):
    """Canonical row-space signature (rref with zero rows dropped)."""
    R, piv = _gf2_rref(M)
    return R[: len(piv)].tobytes()


def block_perm(l: int, m: int, sL, sR) -> np.ndarray:
    from scipy.linalg import block_diag
    return block_diag(monomial_perm(l, m, *sL), monomial_perm(l, m, *sR)).astype(np.uint8)


def test_tdg_constants_match_registration():
    # the registered params ARE the LPU builder's constants (no drift between the two files)
    assert (BB_144_TDG.l, BB_144_TDG.m) == (tdg.L_, tdg.M_)
    assert sorted(BB_144_TDG.a_exps) == sorted(tdg.A_EXPS)
    assert sorted(BB_144_TDG.b_exps) == sorted(tdg.B_EXPS)


def test_polynomials_are_monomial_multiples():
    l, m = 12, 6
    shift = lambda exps, s, t: sorted(((a + s) % l, (b + t) % m) for a, b in exps)
    assert shift(BB_144_12_12.a_exps, 0, -1) == sorted(BB_144_TDG.a_exps)   # A_tdg = y^-1 A
    assert shift(BB_144_12_12.b_exps, -1, 0) == sorted(BB_144_TDG.b_exps)   # B_tdg = x^-1 B


def test_stabilizer_groups_identical_under_explicit_permutation():
    l, m = 12, 6
    HX_b, HZ_b = build_parity_checks(BB_144_12_12)
    HX_t, HZ_t = build_parity_checks(BB_144_TDG)
    # the SAME data permutation must map both sectors at once: L by y^-1, R by x^-1
    P = block_perm(l, m, (0, -1), (-1, 0))
    assert rowspace_sig((HX_b @ P) % 2) == rowspace_sig(HX_t)
    assert rowspace_sig((HZ_b @ P) % 2) == rowspace_sig(HZ_t)
    # sanity: identity permutation does NOT work (the presentations really differ as matrices)
    assert rowspace_sig(HX_b) != rowspace_sig(HX_t)


def test_tdg_code_parameters():
    HX, HZ = build_parity_checks(BB_144_TDG)
    n = HX.shape[1]
    k = n - _gf2_rank(HX) - _gf2_rank(HZ)
    assert (n, k) == (144, 12)
    Zs, Xs = find_logical_ops(HX, HZ)
    assert len(Zs) == len(Xs) == 12
    # commutation: X_i anticommutes with Z_i only (canonical pairs)
    G = (np.array(Xs, dtype=np.uint8) @ np.array(Zs, dtype=np.uint8).T) % 2
    assert np.array_equal(G, np.eye(12, dtype=np.uint8))
