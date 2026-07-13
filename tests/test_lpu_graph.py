"""L1b tier-1 gate: the hand-transcribed gross LPU fixture satisfies every App.-A.3 rule.

The fixture in gross_code_lpu_tdg.py was transcribed verbatim from the paper's TeX; lpu_graph
encodes the CONSTRUCTION RULES. Certifying the fixture against the rules proves (a) the rules
are correctly encoded and (b) the transcription contains no drift — the foundation for deriving
layouts of OTHER codes (mini-LPU, two-gross)."""
import pytest

import gross_code_lpu_tdg as tdg
from bb_code_sim import BB_144_TDG
from lpu_graph import (LPULayout, LPUDerivationError, certify_layout, derive_bridges,
                       derive_edges, derive_identified, build_HX_HZ, fundamental_cycles,
                       support)


@pytest.fixture(scope="module")
def gross_layout() -> LPULayout:
    return LPULayout(
        params=BB_144_TDG, p=tdg.P, q=tdg.Q, r=tdg.R_POLY, s=tdg.S_POLY, mu=(1, 1),
        V_l=list(tdg.V_L_RAW), V_r=list(tdg.V_R_RAW),
        identified_l=tdg.IDENTIFIED_L_SIDE, identified_r=tdg.IDENTIFIED_R_SIDE,
        E_l_detail=[(a, b, zc) for a, b, zc in tdg.E_L_DETAIL],
        E_r_detail=[(a, b, zc) for a, b, zc in tdg.E_R_DETAIL],
        bridge_top=list(tdg.BRIDGE_TOP), bridge_bottom=list(tdg.BRIDGE_BOTTOM),
        U_l=[list(cy) for cy in tdg.U_L], U_r=[list(cy) for cy in tdg.U_R],
        U_b=[list(cy) for cy in tdg.U_B])


def test_gross_fixture_certifies(gross_layout):
    counts = certify_layout(gross_layout)
    assert counts == {"n_vertices": 23, "n_edges": 47, "n_cycles": 19}


def test_identified_vertex_is_derived_not_stated(gross_layout):
    id_l, id_r = derive_identified(BB_144_TDG, tdg.P, tdg.Q, tdg.R_POLY, tdg.S_POLY, (1, 1))
    assert (id_l, id_r) == (tdg.IDENTIFIED_L_SIDE, tdg.IDENTIFIED_R_SIDE)


def test_edges_are_derived_in_fixture_order(gross_layout):
    _, H_Z = build_HX_HZ(BB_144_TDG)
    assert derive_edges(BB_144_TDG, list(tdg.V_L_RAW), H_Z) == \
        [(a, b, zc) for a, b, zc in tdg.E_L_DETAIL]
    assert derive_edges(BB_144_TDG, list(tdg.V_R_RAW), H_Z) == \
        [(a, b, zc) for a, b, zc in tdg.E_R_DETAIL]


def test_fixture_bridge_paths_are_findable(gross_layout):
    """The fixture's Hamiltonian tours must appear in the deterministic path enumeration —
    proving derive_bridges CAN reproduce them (selection policy pinned separately)."""
    paths_l = derive_bridges(tdg.V_L_RAW, gross_layout.E_l_detail, tdg.IDENTIFIED_L_SIDE)
    paths_r = derive_bridges(tdg.V_R_RAW, gross_layout.E_r_detail, tdg.IDENTIFIED_R_SIDE)
    assert list(tdg.BRIDGE_TOP) in paths_l
    assert list(tdg.BRIDGE_BOTTOM) in paths_r


def test_cycle_space_dimensions(gross_layout):
    """G_l/G_r have first Betti number 7; the paper keeps 5/3 after redundancy elimination.
    A full fundamental basis (used for derived layouts of new codes) must have rank 7."""
    assert len(fundamental_cycles(tdg.V_L_RAW, gross_layout.E_l_detail)) == 7
    assert len(fundamental_cycles(tdg.V_R_RAW, gross_layout.E_r_detail)) == 7
    assert len(gross_layout.U_l) == 5 and len(gross_layout.U_r) == 3


def test_certifier_rejects_corruption(gross_layout):
    import dataclasses
    bad = dataclasses.replace(gross_layout, bridge_top=list(reversed(gross_layout.bridge_top)))
    with pytest.raises(LPUDerivationError):
        certify_layout(bad)
    bad2 = dataclasses.replace(gross_layout, identified_l=('L', 0, 0), identified_r=('L', 0, 0))
    with pytest.raises(LPUDerivationError):
        certify_layout(bad2)


def test_support_matches_builder_vertices():
    assert sorted(support(BB_144_TDG, tdg.P, tdg.Q)) == sorted(tdg.V_L_RAW)
    assert sorted(support(BB_144_TDG, tdg.R_POLY, tdg.S_POLY)) == sorted(tdg.V_R_RAW)


# ------------------------------ stage 2: generic derivation ------------------------------
def test_gross_basis_report_clean():
    from lpu_graph import basis_report
    assert basis_report(BB_144_TDG, tdg.P, tdg.Q, tdg.R_POLY, tdg.S_POLY, (1, 1)) == []


def test_generic_derivation_reproduces_gross_structure():
    """derive_lpu_layout from the paper basis alone: same vertices/edges/id as the fixture;
    its own (valid) tour; full 25-cycle basis instead of the paper's reduced 19."""
    from lpu_graph import derive_lpu_layout
    lay = derive_lpu_layout(BB_144_TDG, tdg.P, tdg.Q, tdg.R_POLY, tdg.S_POLY, (1, 1))
    assert (lay.n_vertices, lay.n_edges) == (23, 47)
    assert (lay.identified_l, lay.identified_r) == (tdg.IDENTIFIED_L_SIDE, tdg.IDENTIFIED_R_SIDE)
    assert len(lay.cycles) == 7 + 7 + 11        # full fundamental bases + ladder cycles


def test_mini_lpu_impossible_at_3_3():
    """The [[18,4,4]] Kunlun code admits NO tour-de-gross LPU — proven two ways.

    (a) Pigeonhole lemma: every weight<=4 X-logical touches >= 6 of the 9 Z-checks, so two
        logicals can never have disjoint Z-check neighborhoods (property 4): 6+6 > 9.
        Higher-weight logicals only touch MORE checks, so no weight bound rescues it.
    (b) The exhaustive basis search fails loudly with property violations for all
        45x44x9 combinations."""
    import numpy as np
    from bb_code_sim import BB_18_4_4
    from lpu_graph import (LPUDerivationError, build_HX_HZ, enumerate_x_logicals,
                           find_mini_basis, qubit_index)
    _, H_Z = build_HX_HZ(BB_18_4_4)
    from lpu_graph import support as supp
    sizes = []
    for p, q in enumerate_x_logicals(BB_18_4_4, max_weight=4):
        cols = [qubit_index(BB_18_4_4, v) for v in supp(BB_18_4_4, p, q)]
        sizes.append(int((H_Z[:, cols].sum(axis=1) > 0).sum()))
    assert min(sizes) == 6 and 2 * min(sizes) > H_Z.shape[0]      # the pigeonhole
    with pytest.raises(LPUDerivationError, match="no valid LPU basis"):
        find_mini_basis(BB_18_4_4, max_weight=4)
