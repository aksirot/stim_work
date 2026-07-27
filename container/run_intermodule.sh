#!/usr/bin/env bash
# Launch the Wave-6i gross-to-gross inter-module campaigns (X1(A)xX1(B)) on a NO-SCHEDULER,
# SHARED box (rodan: 96 cores, no sbatch/srun). Sibling of container/run_lpu_boost.sh.
#
#   bash container/run_intermodule.sh --dry-run   # print the podman commands, launch nothing
#   bash container/run_intermodule.sh             # launch both legs, detached
#   bash container/run_intermodule.sh --only r1   # substring filter on the job name
#
# WHY NOT run_local.sh: that launcher bind-mounts only runs/, so it would run against the image's
# BAKED experiments/ and never see these configs. This one mounts experiments/ too.
#
# SHARED-BOX BUDGET: CPUS=8 per job x 2 jobs = 16 of 96 cores (17%). If the Wave-5b boost pass is
# also running (3 jobs x 8 = 24), the combined footprint is 40 of 96 (42%). CHECK WHO ELSE IS ON
# THE BOX before launching; stagger the waves if the machine is busy.
#
# MEMORY: this is the largest circuit in the campaign -- ~418k DEM mechanisms, 10903 detectors at
# the production geometry (C=10, d_init=12, idle ON). The manifest's 64G is an UNTESTED ESTIMATE
# and there is NO SCHEDULER HERE TO ENFORCE A CAP: an OOM takes down whatever else shares the box,
# not just this job. Run the smoke first (see docs/cluster_runbook.md "Wave 6i" step 2) and watch
# RSS with `podman stats` before committing to the full run.
#
# Thread capping is done with env vars, NOT podman --cpus: rootless cgroup delegation is often
# unavailable and --cpus then errors out.
#
# RESUMABLE: run_is_sweep checkpoints after every weight, so a killed container loses at most one
# bin. To resume, `podman rm <name>` and re-run this script — it picks up from spectrum.json.
# Do NOT edit a config while its job is live: the guard at experiment_runner.py:642 aborts the
# resume on any weights_plan/seed change. The frozen blocks are r1 [1,1674] / r10 [1,1742].
set -euo pipefail
cd "$(dirname "$0")/.."             # repo root
REPO="$PWD"

DRY=0
ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --only)    [[ $# -ge 2 ]] || { echo "--only needs a substring"; exit 1; }
               ONLY+=("$2"); shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:Z}"         # SELinux relabel; rodan NEEDS this (denies the mount without it)
CPUS="${CPUS:-8}"                   # threads per job — see the shared-box budget above
mkdir -p runs/framework

matches() {
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local n="$1" s
  for s in "${ONLY[@]}"; do [[ "$n" == *"$s"* ]] && return 0; done
  return 1
}

launch() {  # name  config-path
  local name="$1" cfg="$2"
  matches "$name" || return 0
  local cmd=( podman run -d --name "$name"
    -e "OMP_NUM_THREADS=${CPUS}" -e "OPENBLAS_NUM_THREADS=${CPUS}"
    -e "MKL_NUM_THREADS=${CPUS}" -e "RAYON_NUM_THREADS=${CPUS}"
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
    -w /opt/stim_work "${IMAGE}"
    python -m experiment_runner --config "${cfg}" --cpus "${CPUS}" )
  echo "[launch] ${name}  (threads=${CPUS})  cfg=${cfg}"
  if [[ $DRY -eq 1 ]]; then printf '    %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

launch im_r1  experiments/configs/gross_intermodule_r1.yaml
launch im_r10 experiments/configs/gross_intermodule_r10.yaml

echo
echo "[run_intermodule] watch:  podman ps ; podman logs -f im_r1 ; podman stats"
echo "[run_intermodule] resume: podman rm <name> && bash container/run_intermodule.sh --only <name>"
