# rodan quickstart — Wave 6i inter-module campaign

Copy-paste, zero to launched. Self-contained: every command you need, in order, including the
git fetch/checkout. Deeper rationale lives in `cluster_runbook.md` section **Wave 6i**.

## What rodan is (and is not)

Four constraints shape every command below. All four have already broken something.

| | consequence |
|---|---|
| **No scheduler** — no `sbatch`, no `srun` | Nothing tracks allocations. You launch detached podman containers and police your own footprint. Any runbook line using `srun`/`sbatch` is from the SLURM era and will not work. |
| **No host python** | `python`/`pytest` on the login node do not exist. Everything python runs *inside* the image. Host tooling is git, rsync, podman, shell. |
| **Shared, 96 cores** | Someone else may be working. We cap ourselves at **half the box**. |
| **Rootless podman** | `podman ps` shows **only your own** containers — see the trap in step 3. |

Plus one that is specific to this campaign: **the shipped image predates the code.** The image is
from 2026-07-24; `build_joint_x1x1_circuit` and its `close_cycles` fix landed 2026-07-27. The
Containerfile *bakes* `src/`, so every command that runs this campaign's code must bind-mount
`src/` over the baked copy. Miss it and jobs die on an import error that reads like a circuit bug.

---

## 1. Get the code

**Do not just `git pull`.** The Wave-1 setup left this clone on `bb144-split-better`, a branch that
no longer exists on either remote, so a bare pull fails or silently does nothing — and you then
launch stale code.

```bash
cd stim_work
git branch --show-current                        # expect a surprise (deleted branch)
git fetch origin --prune
git checkout main || git checkout -b main origin/main
git pull --ff-only origin main                   # ff-only: no surprise merge, unattended
```

Verify you actually got it:

```bash
git rev-parse --short HEAD                       # record this as your pinned SHA
ls container/box_load.sh                         # must exist
grep -m1 'CPUS:-' container/run_intermodule.sh   # expect CPUS:-24
```

If `box_load.sh` is missing, the pull did not take.

## 2. Verify the suite — inside the container

Mount `tests/` as well as `src/`, or you validate the image's **baked** tests (which predate the
builder) instead of the checkout you just pulled.

```bash
podman run --rm \
  -v "$PWD/src:/opt/stim_work/src:Z" \
  -v "$PWD/experiments:/opt/stim_work/experiments:Z" \
  -v "$PWD/tests:/opt/stim_work/tests:Z" \
  -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work \
  localhost/stim-work-qec:latest python -m pytest -q
```

## 3. Check the box has room

```bash
bash container/box_load.sh              # report + a recommended CPUS
bash container/box_load.sh --want 48    # does the half-box footprint fit?
```

> **The trap:** podman is rootless here, so `podman ps` shows only *your* containers. A colleague
> can be using 60 cores through their own podman and your `podman ps` comes back **empty**. Never
> conclude "the box is free" from podman. `box_load.sh` reads the 1-minute load average and `ps`,
> which see every user.

If it reports less headroom than the policy cap, launch with the smaller `CPUS` it suggests
(`CPUS=16 bash container/run_intermodule.sh`).

## 4. Smoke it first — do not skip

Validates the memory footprint and per-shot rate before committing days of shared-box time. The
DEM build alone is several minutes.

```bash
podman run --rm -e OMP_NUM_THREADS=4 -e RAYON_NUM_THREADS=4 \
  -v "$PWD/src:/opt/stim_work/src:Z" \
  -v "$PWD/experiments:/opt/stim_work/experiments:Z" \
  -v "$PWD/runs:/opt/stim_work/runs:Z" \
  -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work \
  localhost/stim-work-qec:latest \
  python -m experiment_runner --config experiments/configs/gross_intermodule_r1.yaml \
    --smoke --cpus 4
```

**PASS** = `runs/framework/bb144/intermodule_r1_smoke/result.npz` exists.

Watch `podman stats` while it runs. Measured peak is **9.0 GB per job** (~18 GB for the pair), so
the manifest's 64G has ~7x headroom — but nothing here enforces a cap, so an OOM would take down
whatever else shares the box, not just your job.

## 5. Dry-run, then launch

```bash
bash container/run_intermodule.sh --dry-run     # expect exactly two podman commands
bash container/run_intermodule.sh
```

The launcher preflights the builder import inside the container before starting anything, so a
stale image fails in seconds rather than after the DEM build.

**Footprint:** `CPUS=24` x 2 legs = **48 of 96 cores**, half the box, campaign-only.
**Expected wall:** ~35 h per leg, both in parallel ≈ **1.5 days**.

## 6. Watch

```bash
podman ps
podman logs -f im_r1
podman stats
```

## 7. Resume after a kill

`run_is_sweep` checkpoints after **every weight**, so a killed container loses at most one bin.

```bash
podman rm im_r1
bash container/run_intermodule.sh --only r1
```

> **Never edit a config while its job is live.** The guard at `experiment_runner.py:642` aborts the
> resume on any `weights_plan`/seed change. The frozen blocks are r1 `[1,1674]`, r10 `[1,1742]`.

## 8. Pull results home

Analysis needs python, so it happens on the **local box**, not here.

```bash
# from the LOCAL box
rsync -av rodan:stim_work/runs/framework/ runs/cluster/framework/
```

Safe at any time — checkpoints are atomic, so you can pull mid-run.

**Close criteria:** `f(w)` for `inter_module` within Poisson error of the `y1` baseline at every
weight, and the IS spectra land inside the frozen weight blocks (no mass piled at `w_hi`).

> No separate large-T `f(w)` job is needed — the production sweep subsumes it. The frozen block
> starts at w=1 with stride 6 and sub-onset bins run to the 3000-shot cap, giving f(1) resolution
> ~3.3e-4 at the *production* geometry, finer than the standalone probe.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `git pull` fails / changes nothing | clone is on the deleted `bb144-split-better` | step 1, explicit `checkout main` |
| `ImportError` / `AttributeError` on `build_joint_x1x1_circuit` | `src/` not mounted; image predates the builder | add the `-v .../src:...` mount (the launcher already has it) |
| launcher exits `[preflight] ... FAILED` | same as above, caught early | check the `src/` mount, or rebuild the image |
| `python: command not found` | no host python | run it inside `podman run ... localhost/stim-work-qec:latest python ...` |
| tests pass but the run fails | you tested the image's **baked** `tests/` | mount `tests/` too (step 2) |
| `podman ps` empty but box is slow | rootless podman hides other users | `bash container/box_load.sh` |
| mount denied / permission errors | missing SELinux relabel | keep the `:Z` suffix on every `-v` |
| `podman --cpus` errors | rootless cgroup delegation unavailable | thread caps are env vars (`OMP_/RAYON_NUM_THREADS`), never `--cpus` |
| resume aborts on mismatch | a config changed under a live run | restore the config, or start a new outdir |
