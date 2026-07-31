"""Generate device_calib_transition.ipynb — priors-vs-structure at the convention switch.

Documents WHY the per-model-calibrated sub-model data was retired (2026-07-31) and what
changed: the same five [[72,4,8]] sub-models decoded by their retired specialist
decoders vs the one device decoder. All code lives in src/emc_report.py (DevTransition);
cells are one-liners and re-execute as the device campaign + top-up deepen the bins.
"""
import json

from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# Device-calibration transition — priors vs structure on the [[72,4,8]] channels

**The convention change (2026-07-31, commits 3916611b/da2ae43d):** the legacy campaign
calibrated each error-model task's decoder priors on that task's OWN noise model — so
"meas only" was decoded by a decoder that believed only measurement errors exist. That
specialist is not the device's decoder, and the per-channel rows never answered the
error-budget question. Under `EMC_CALIB=device` every model and ablation of a code is
decoded by ONE decoder calibrated on the full-symmetric circuit at p* = 5e-4 (the ×5
asym rays keep their own full-asym device decoder). The legacy sub-model results were
retired to `runs/_retired_permodel_calib/` — kept only so this notebook can measure the
priors effect by differencing.

**Reading:** same circuits, same sampling — every device-vs-specialist difference below
is purely the decoder's priors. |z| > 2 marks real effects. Bins deepen as the rodan
device campaign + specimen top-up merge in; re-execute to tighten.""")

code(r'''from emc_report import DevTransition
D = DevTransition()
D.load()''')

md(r"""## 1. Channel spectra — specialist (retired) vs device decoder

First finding (generation-depth bins): the priors effect is channel-dependent, and it
moves the error budget's ordering — **gate idle** decodes 2–3× BETTER under the device
decoder (cross-channel context helps; the legacy data overstated its budget share),
while **meas idle** decodes 2–6× WORSE (it genuinely benefited from specialist priors;
the legacy data understated it). CZ and prep show no priors effect.""")
code(r'''D.fig_channels()''')
code(r'''D.ratio_table()''')

md(r"""## 2. Full-symmetric anchors (convention-invariant)

The full-symmetric model calibrates on itself under BOTH conventions, so the headline
decoder results carry over unchanged: the measured-vs-measured spectra (3e6-shot top-up
bins, w = 2–10 both decoders) and the reweighted LER at p*.""")
code(r'''D.anchor_table()''')

md(r"""## 3. What re-executes as data lands

* **Device campaign** (rodan `emc_device_s*`): ablations + asym under the device
  decoder — the error-budget sections of the main report regenerate from that dir.
* **Specimen top-up** (`run_sys_topup.sh` on the device dir): deepens the w = 2–10
  channel bins to the 3e6 cap AND records failing mechanism configurations
  (`failure_configs`, ≤200/weight). Under the device convention a sub-model failing
  config IS a full-model failing config — specimens feed the decoder-loop library
  directly (`add_w3_harvest_to_library.py` pattern).
* This notebook's §1 ratios tighten automatically on re-execution; the main report
  copy against the device dir is the final deliverable once the campaign completes.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "device_calib_transition.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
