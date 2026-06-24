"""Generate BB6_meas_comparison.ipynb (source only; cells are not executed here).

Focused comparison of the **single-sector** fail-fast analysis under two noise models:
  * symmetric  (p_meas = p_phys)            -> bb6_fig10_curve/
  * 5x-meas    (p_meas = 5 x p_phys)        -> bb6_meas5x_curve/

Emits the **four** requested comparisons, all reusing the parameterized bb6_report helpers:
  1. LER(p) curve overlay            (fig_compare_ler)
  2. Table 2 min-weight properties   (table2_compare)         -- exact MITM both sides
  3. Failure spectrum f(w) overlay   (fig_compare_spectrum)
  4. Weight distribution pi_q(w)     (fig_compare_weight)

Parallel to make_fulldem_notebook.py. All heavy computation was done by
`bb6_fig10_sweep.py --p-meas-factor 5 --outdir .../bb6_meas5x_curve --no-split` plus the exact
single-sector MITM (distance_mitm.json); this notebook just loads + re-fits + overlays (fast).
"""
import json, pathlib

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# Asymmetric noise — single-sector BB(6): **5x measurement error** vs symmetric

Companion to `BB6_failfast_report.ipynb`. That report used a **symmetric** depolarizing model
(`p_meas = p_phys`). Here we ask: *what changes if measurement is noisier than the gates?* — we set
**`p_meas = 5 x p_phys`** and re-run the **single-sector** fail-fast pipeline, then compare to the
symmetric baseline across the **four** quantities below.

| # | comparison | helper | what it shows |
|---|---|---|---|
| 1 | **LER(p)** curve | `fig_compare_ler` | end-to-end logical error rate vs physical error rate |
| 2 | **Table 2** min-weight props | `table2_compare` | D, w0, N, \|L(D)\|, \|F\|, f0 — **exact MITM both sides** |
| 3 | **failure spectrum** f(w) | `fig_compare_spectrum` | weight-resolved failure profile + exact onset ★ |
| 4 | **weight distribution** pi_q(w) | `fig_compare_weight` | median weight of the *failing* configs vs p |

**Did this need a rewrite? No.** `ErrorModel` already carries independent `p_phys`/`p_meas`, and the
fail-fast reweighting `LER(p)=Σ_w C(N,w) q^w (1-q)^{N-w} f(w)` is **rate-agnostic**: each mechanism is
expanded into `round(p_j / q_base)` copies at a common `q_base=min(probs)`, and `q=q_base·(p/p_ref)`
scales *all* mechanisms by the same factor. So a fixed-ratio asymmetric model (here 5:1), swept by an
overall strength `p`, is a valid single-parameter family the existing machinery handles — the only new
code is a one-line `--p-meas-factor` seam in the sweep.

**What we expected vs what we found.** Naively the *combinatorial* structure (D, w0, the compressed
mechanism set) is rate-independent and should be unchanged. That held for **D=6, w0=3**, but the
single-sector *merged* representation turned out to be mildly rate-sensitive: the compressed counts
shift (Ñ 2233→2232, \|L(D)\|_comp 1524→912). The *rate-weighted* quantities move as expected and
dominate the LER: with 5x measurement noise there are more measurement fault mechanisms, so the
**expanded** N grows (46,224→80,064) and **LER(10⁻⁴) is ~4x higher**. The exact MITM (`distance_mitm.json`)
pins the onset f0 on both sides — the BP-OSD search undercounts \|L(D)\|.""")

md(r"""### Setup — imports, then load + fit both runs
First cell only imports; the second loads + fits **both** curve dirs via `bb6_report.compute` (~20-30 s
total, two fits + bootstrap bands).""")

code('''import sys, pathlib
print("kernel python:", sys.executable)   # should be the 'qec' env (numpy/scipy/matplotlib)
_here = pathlib.Path.cwd()
_cands = [_here, *_here.parents] + [c / "notebooks" / "bb_code" for c in [_here, *_here.parents]]
_nbdir = next((c for c in _cands if (c / "bb6_report.py").exists()), None)
assert _nbdir is not None, "Could not locate bb6_report.py — run this notebook from inside the repo."
sys.path.insert(0, str(_nbdir)); sys.path.insert(0, str(_nbdir.parent.parent / "src"))
import numpy as np, matplotlib.pyplot as plt
import bb6_report
SYM_DIR   = _nbdir / "bb6_fig10_curve"    # symmetric  (p_meas = p_phys)
MEAS5_DIR = _nbdir / "bb6_meas5x_curve"   # 5x-meas    (p_meas = 5 x p_phys)
print("imports OK — run the next cell to load + fit both runs (~20-30s)")''')

code('''import time
print("loading + fitting both runs ...", flush=True); _t = time.time()
R_sym = bb6_report.compute(SYM_DIR)      # symmetric baseline
R_5x  = bb6_report.compute(MEAS5_DIR)    # 5x measurement error
print(f"loaded in {time.time()-_t:.0f}s — "
      f"symmetric: {R_sym['meta'].get('noise_label','symmetric')} (method={R_sym['meta']['method']}); "
      f"5x-meas: {R_5x['meta'].get('noise_label')} (method={R_5x['meta']['method']})")''')

md(r"""## 1 + 2 — Table 2 (exact MITM) and the LER(p) curve

The Table-2 side-by-side uses the **exact** detector-pivot MITM on *both* runs (`distance_mitm.json`).
The fault distance and onset match (**D=6, w0=3**); the compressed counts shift slightly and the
rate-weighted counts (`n_expanded`, `|F|`, `|L(D)|_exp`) grow with the heavier measurement noise.""")

code('''from IPython.display import Markdown, display
display(Markdown(bb6_report.table2_compare(R_sym, R_5x)))
i = np.argmin(abs(R_sym["p_grid"] - 1e-4))
j = np.argmin(abs(R_5x["p_grid"]  - 1e-4))
ls, l5 = R_sym["LER"]["f3"][i], R_5x["LER"]["f3"][j]
print(f"\\nLER(1e-4):  symmetric f3 = {ls:.3e}   |   5x-meas f3 = {l5:.3e}   (ratio {l5/ls:.2f}x higher)")''')

code('''fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_compare_ler(R_sym, R_5x, ax)
plt.show()''')

md(r"""## 3 — Failure spectrum f(w)

The weight-resolved failure fraction. Both onsets sit at **(w0=3, f0)** (the ★, pinned by the exact
MITM). Above the onset the 5x-meas spectrum rises *more slowly* in w (it takes more fault weight to
reach the same failure fraction), but there are *more* low-weight mechanisms — the net effect on the
LER is set by the reweighting, which the LER curve above already integrates.""")

code('''fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_compare_spectrum(R_sym, R_5x, ax)
plt.show()''')

md(r"""## 4 — Weight distribution pi_q(w)

Median fault weight of the *failing* configurations vs p (reweighted ansatz). At the onset both
collapse to **w0=3**; as p rises the typical failing weight climbs, and it climbs **faster** for
5x-meas — at p=10⁻² the median failing-config weight is markedly higher, i.e. failures are dominated by
heavier (measurement-rich) fault patterns.""")

code('''fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_compare_weight(R_sym, R_5x, ax)
plt.show()''')

md(r"""## Takeaways

* **No rewrite needed** — the rate-agnostic reweighting + a one-line `--p-meas-factor` seam handle a
  fixed-ratio asymmetric noise model. (A genuine 2-D decoupling of `p_meas` from `p_phys` *would* need
  multi-rate reweighting — out of scope here.)
* **D=6, w0=3 are robust** to the noise profile; the single-sector *merged* counts are mildly
  rate-sensitive (Ñ, \|L(D)\|_comp shift), and the rate-weighted quantities (N, \|F|, f0, LER) move as
  expected.
* **5x measurement noise raises LER(10⁻⁴) by ~4x** — driven by the larger expanded mechanism count N
  (more measurement faults → C(N,3) grows), which outweighs the slightly *lower* onset fraction f0.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "BB6_meas_comparison.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
