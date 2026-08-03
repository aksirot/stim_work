# Rendered reports (viewable without running anything)

The `runs/` task cache is gitignored, so the notebooks in the parent directory cannot
be re-executed from a clone — but their results are readable here.

| report | render on GitHub | self-contained file |
|---|---|---|
| error-model comparison, **device calibration** (18/72, one decoder per code) | [Markdown](error_model_comparison_18_4_4_device.md) | [HTML](error_model_comparison_18_4_4_device.html) |
| device-calibration **transition** (priors vs structure) | [Markdown](device_calib_transition.md) | [HTML](device_calib_transition.html) |

* **Markdown** renders inline on github.com, images and all — click those first.
* **HTML** is a single self-contained page (figures embedded); GitHub shows its
  *source*, not the page, so download it or open it through a raw-HTML previewer.
* The `.ipynb` originals carry the same outputs, but the device report is ~1.4 MB and
  GitHub's notebook viewer gives up above ~1 MB — another reason to use the Markdown.

Regenerate after re-executing a notebook:

```
jupyter nbconvert --to html     --output-dir notebooks/methods/rendered notebooks/methods/<nb>.ipynb
jupyter nbconvert --to markdown --output-dir notebooks/methods/rendered notebooks/methods/<nb>.ipynb
```
