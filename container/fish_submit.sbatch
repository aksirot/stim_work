#!/bin/bash
# REFERENCE sbatch for the "fish"-style setup: a SHARED podman image maintained by
# others, with our checkout bind-mounted in rather than baked. Adapt, don't adopt —
# PODMAN_ROOT/PODMAN_RUNROOT and the partition/account flags are site-specific.
#
# Established on fish 2026-08-03:
#   * the shared image HAS stim, relay_bp, numpy, scipy  (test.py package checks pass)
#   * it does NOT have our project installed              (project checks failed)
#     -> PYTHONPATH=<repo>/src fixes that with no rebuild and no fork of their image
#   * SLURM binaries are NOT visible inside the container -> always pass --no-slurm
#     to test.py; a containerised job cannot query the scheduler
#
# Submit from the directory ABOVE the checkout (so $PWD holds stim_work/ and this file):
#     sbatch --array=0-3 stim_work/container/fish_submit.sbatch
#
#SBATCH --job-name=qec-fish
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16G

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm

REPO_DIR="${REPO_DIR:-stim_work}"          # checkout name under $PWD
IMAGE="${QEC_IMAGE:?set QEC_IMAGE to the shared image}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# What to run INSIDE the container, relative to the mounted parent directory.
#   $REPO_DIR/test.py --no-slurm                     environment preflight
#   $REPO_DIR/container/smoke_decode.py              [[18,4,4]] decode (~1 min)
#   $REPO_DIR/container/smoke_splitting.py           real splitting path (~10 min)
#   $REPO_DIR/experiments/methods/version_report.py  checkout/feature verification
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$REPO_DIR/container/smoke_decode.py}"

echo "[fish] host=$(hostname) task=${IDX} cpus=${CPUS} script=${PYTHON_SCRIPT}"

# THREAD CAPS matter: relay_bp's Rayon pool defaults to every core on the NODE, not the
# cores SLURM gave this task. Without these, concurrent tasks oversubscribe and decode
# throughput collapses (measured ~10-30x on a 24-core box).
exec podman ${PODMAN_ROOT:+--root "$PODMAN_ROOT"} ${PODMAN_RUNROOT:+--runroot "$PODMAN_RUNROOT"} \
  run --rm \
  -v "$PWD:/work" -w /work \
  -e PYTHONPATH="/work/${REPO_DIR}/src" \
  -e OMP_NUM_THREADS="${CPUS}" -e OPENBLAS_NUM_THREADS="${CPUS}" \
  -e MKL_NUM_THREADS="${CPUS}" -e RAYON_NUM_THREADS="${CPUS}" \
  -e IDX="${IDX}" -e SLURM_ARRAY_TASK_ID="${IDX}" \
  ${SMOKE_DECODER:+-e SMOKE_DECODER="$SMOKE_DECODER"} \
  "${IMAGE}" \
  python /work/${PYTHON_SCRIPT}
