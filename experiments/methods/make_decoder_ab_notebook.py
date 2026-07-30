"""Generate emc_decoder_ab.ipynb — IS-spectra A/B of the two decoder generations.

Compares the baseline-decoder campaign cache (runs/error_model_comparison_18_4_4/)
against the system-level campaign (runs/error_model_comparison_18_4_4_sys_baseline18_ghw72/:
18-code = baseline decoder, 72-code = ghw). All plotting/table code lives in
src/emc_report.py (fig_ab_72, ab_ratio_table); the cells here are one-liners.
"""
import json

from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

SYS = "error_model_comparison_18_4_4_sys_baseline18_ghw72"

md(r"""# Decoder A/B — IS failure spectra, baseline vs ghw generation

Side-by-side importance-sampled failure spectra of the two campaign generations on the
**[[72,4,8]]** code:

* **baseline** (`runs/error_model_comparison_18_4_4/`): the tuned-cheap Relay-BP
  (`num_sets=20, pre_iter=80, gamma0=0.125`), with `--boost72` budgets **plus the 3e6-shot
  onset top-ups** — its low-weight bins are measured to the ~1e-6 level.
* **ghw** (`runs/""" + SYS + r"""/`): the SYSTEM-level run — 18-code keeps the baseline
  decoder (spectra bit-identical, so no 18-code panels here), 72-code runs full
  glo_heavy_wide (`pre_iter=320, gamma0=0.0625, 200x120 sets, gamma in (-0.5,1.0)`) at
  **1x adaptive budgets** (10k-shot caps, no top-up yet).

**The one reading rule:** the generations' budgets differ by up to 300x in the low-weight
bins. A gray baseline dot with only a crimson × above it means the ghw run's budget could
not resolve that bin (the × is a 1/(2T) *bound*), **not** that ghw is clean there. Ratios
are therefore quoted only where BOTH generations measured failures — that is where the
ghw improvement is real and quantified. The sub-onset comparison stays bound-limited until
the ghw top-up (onset_topup_72.py with the sys EMC envs) deepens those bins.""")

code(r'''from emc_report import Report, fig_ab_72, ab_ratio_table

B = Report()                      # baseline generation (deep budgets + topups)
S = Report("''' + SYS + r'''")    # system-level generation (72-code = ghw, 1x budgets)
for R in (B, S):
    R.load_spectra(); R.load_spectra_72(); R.load_ablations(); R.load_ablations_72(); R.load_asym()''')

md(r"""## 1. Isolated + full models""")
code(r'''fig_ab_72(B, S, family="models")''')
code(r'''ab_ratio_table(B, S, family="models")''')

md(r"""## 2. Leave-one-out mixes""")
code(r'''fig_ab_72(B, S, family="ablated")''')
code(r'''ab_ratio_table(B, S, family="ablated")''')

md(r"""## 3. Asymmetric ×5 mixes (meas, meas-idle ×5)""")
code(r'''fig_ab_72(B, S, family="asym")''')
code(r'''ab_ratio_table(B, S, family="asym")''')

md(r"""## Takeaways

* Wherever both generations measure the same bin, **ghw is equal or better** — the mid/onset
  region improvements (up to ~3× on gate idle) match the local paired benches that selected it.
* **Do not read Λ off this generation yet**: its 72-code low-weight bins are zero-at-10k
  bounds, which the reweighting counts as zero — Λ computed from them is a truncation-inflated
  upper bound (the ×5 box's honest interval spans back down to the baseline's value).
* Next data step: the ghw onset top-up into the sys cache; these panels tighten automatically
  on re-run once it lands.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "emc_decoder_ab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
