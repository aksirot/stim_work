#!/usr/bin/env bash
# Host-side launcher for the stim-work-qec container on a NO-SLURM box (e.g. the 96-core node).
# Runs any command inside the image with runs/ bind-mounted so outputs persist on the host. Both the
# base error-model campaign and the onset top-up resolve runs/ via repo_paths to /opt/stim_work/runs,
# so this one bind-mount covers both, and the top-up sees the spectra the base campaign wrote.
#
# Typical sequence on the 96-core box:
#   # 1) base campaign FIRST — writes the cached 72-code spectra the top-up merges into:
#   bash container/run_local.sh python experiments/methods/run_error_model_comparison.py
#
#   # 2) confirm all 17 spectra are present before topping up:
#   bash container/run_local.sh python experiments/methods/onset_topup_72.py --list
#
#   # 3) onset top-up — long, so detach it (half the cores by default):
#   DETACH=1 bash container/run_local.sh bash experiments/methods/run_onset_topup_local.sh
#
#   # quick sanity check of the container itself:
#   bash container/run_local.sh python test.py --no-slurm
#
# Env knobs (all optional):
#   IMAGE     localhost/stim-work-qec:latest   image to run
#   FRACTION  0.5      core fraction the onset driver self-caps to (NPROC x THREADS)
#   CPUS      (unset)  hard cap the whole container to N cpus via --cpus (needs rootless cgroup
#                      delegation; if podman errors about cgroups, just leave it unset)
#   RUNS_DIR  $PWD/runs   host dir bind-mounted to /opt/stim_work/runs
#   MOUNT_OPT :z       SELinux relabel suffix; set MOUNT_OPT= (empty) if the node isn't SELinux-enforcing
#   DETACH    0        1 = detached, kept as a named container so logs/exit survive
#   NAME      qecjob   container name when detached
#   THREADS NPROC ONSET_SHOTS_MAX ONSET_TARGET ONSET_CHUNK  forwarded to the onset driver if set
set -euo pipefail

IMAGE="${IMAGE:-localhost/stim-work-qec:latest}"
FRACTION="${FRACTION:-0.5}"
RUNS_DIR="${RUNS_DIR:-$PWD/runs}"
MOUNT_OPT="${MOUNT_OPT-:z}"          # ${VAR-default}: keep an explicitly-empty MOUNT_OPT
NAME="${NAME:-qecjob}"
DETACH="${DETACH:-0}"

if [ "$#" -eq 0 ]; then
  echo "usage: [DETACH=1] [FRACTION=0.5] bash container/run_local.sh <command...>" >&2
  echo "  e.g. bash container/run_local.sh python test.py --no-slurm" >&2
  exit 2
fi

mkdir -p "$RUNS_DIR"

run=( podman run )
if [ "$DETACH" = "1" ]; then
  run+=( -d --name "$NAME" )         # no --rm: keep it so `podman logs`/exit code survive
else
  run+=( --rm )
fi
[ -n "${CPUS:-}" ] && run+=( --cpus "${CPUS}" )
run+=( -v "${RUNS_DIR}:/opt/stim_work/runs${MOUNT_OPT}" -e "FRACTION=${FRACTION}" )
for v in THREADS NPROC ONSET_SHOTS_MAX ONSET_TARGET ONSET_CHUNK \
         RAYON_NUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS; do
  [ -n "${!v:-}" ] && run+=( -e "${v}=${!v}" )
done
run+=( "$IMAGE" "$@" )

if [ "$DETACH" = "1" ]; then
  "${run[@]}"
  echo "[run_local] detached as '${NAME}' — results under ${RUNS_DIR}"
  echo "[run_local]   watch:   podman logs -f ${NAME}"
  echo "[run_local]   cleanup: podman rm ${NAME}   (after it exits)"
else
  exec "${run[@]}"
fi
