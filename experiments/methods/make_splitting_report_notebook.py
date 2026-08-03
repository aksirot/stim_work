"""Generate splitting_arbitration_72.ipynb — did splitting survive arbitration?

Thin notebook over src/splitting_report.py. Data: runs/splitting_crosscheck/
{72_full_paper, 72_full_ghw, 72_full_v2_ghwdeep}.json plus the campaign spectra.
"""
import json

from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# Splitting arbitration on the [[72,4,8]] code

**The question.** Technique III (splitting) is supposed to reach rates where direct
Monte Carlo cannot. But an estimator that cannot be checked is an estimator that cannot
be trusted, so we ran every variant over a window where MC *is* affordable (p = 2e-3,
where the [[72,4,8]] full-symmetric circuit fails often enough to measure directly) and
asked whether each reproduces the measured value.

**What was compared.**

| estimator | what it is |
|---|---|
| **paper splitting** | arXiv:2511.15177 Alg. 2/3 as published — Eq. 18 fine ladder (steps $2^{-1/\sqrt{w}}$), multi-seeded warm starts from *typical* failing configurations at $p_0$, adaptive $\sigma+\Delta$ chain-growth controller |
| **tempering v1 / v2** | replica exchange — the paper's *future-work* suggestion, our implementation: coarse geometric ladder, walkers holding one replica per rate, swaps between adjacent rates. v2 adds library-diverse seeding, coset-jump moves and three independent ladders |
| **reweighted IS** | Technique I: measured $f(w)$ bins (topped up to 3e6 shots at low $w$ on rodan) reweighted to $p$ |
| **direct MC** | ground truth at p = 2e-3 |

All runs decode with a campaign-grade Relay-BP decoder; the paper run and tempering v2
use `ghw_deep`, the decoder-loop's verified recommendation.""")

code(r'''from splitting_report import SplitReport
S = SplitReport()
S.load()''')

md(r"""## 1. The arbitration

Left: all three estimators against the MC point (star). Right: the mechanism — the mean
fault weight of the configurations each method actually samples at every rung.""")
code(r'''S.fig_arbitration()''')
code(r'''S.arbitration_table()''')

md(r"""**Reading it.** The paper's algorithm lands within its own error bar of the MC
measurement; both tempering variants sit orders of magnitude below it and get *worse*
as p decreases. The right panel says why: the paper's fine ladder and typical-config
warm starts keep the sampled weight distribution where the probability mass is
(mean weight ~24-29 all the way down), whereas tempering slides onto minimal-weight
cores (mean weight -> ~5-8). The rung ratios then price that narrow basin instead of
the full failing set — biased low, compounding at every rung.""")

md(r"""## 2. The controller, and why its "converged" is not "correct"

The paper's adaptive rule grows each level's chain until $\sigma + \Delta \le
\epsilon/\sqrt{t}$, where $\sigma$ is the relative spread of the ratio terms and
$\Delta$ the full-vs-first-half mixing discrepancy.""")
code(r'''S.controller_table()''')

md(r"""## 3. Cost, and what the residual bias is

This run is deliberately laptop-scaled. The point estimate sits low, but the
across-instance spread — the paper's prescribed error bar — covers the truth, so
under-sampling announces itself rather than hiding.""")
code(r'''S.cost_table()''')

md(r"""## 4. Verdict

* **The published method survives arbitration** at 72-code scale (z = 1.3), at 1/500th
  of the paper's per-level sampling.
* **Our tempering shortcut does not**, in either implementation — and neither the
  internal gates (swap acceptance, mean-weight monotonicity) nor cross-ladder agreement
  detected the failure. Only the external MC check did.
* **Practical consequence for this campaign**: IS + top-up remains the cheaper route
  wherever f(w) is measurable, and it is now double-validated (direct MC here, fixed
  weight sampling on rodan). Splitting earns its keep only where IS cannot go — the
  idle-class circuits — and at a cost that must be budgeted, not assumed.
* **Never quote a splitting curve without an overlap point.** That is the whole lesson
  of this notebook.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "splitting_arbitration_72.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
