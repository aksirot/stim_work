#!/usr/bin/env bash
# Launch the three Wave-5b LPU boost campaigns on a NO-SCHEDULER, SHARED box (rodan: 96 cores,
# no sbatch/srun). Replaces experiments/slurm/submit_lpu_boost.sh, which is unusable there.
#
#   bash container/run_lpu_boost.sh --dry-run   # print the podman commands, launch nothing
#   bash container/run_lpu_boost.sh             # launch all three, detached
#   bash container/run_lpu_boost.sh --only y1   # substring filter on the job name
#
# WHY NOT run_local.sh: that launcher bind-mounts only runs/, so it would run against the image's
# BAKED experiments/ and never see these configs. This one mounts experiments/ too.
#
# SHARED-BOX BUDGET: CPUS=8 per job x 3 jobs = 24 of 96 cores (25%). Do not raise it without
# checking who else is on the box. Expected wall at 8 cores/job, all three in parallel:
#   idle ~4.6 d   automorphism ~3.5 d   Y1 ~4.9 d      (floors 1.0e-5 / 1.5e-4 / 3.0e-5)
#
# Thread capping is done with env vars, NOT podman --cpus: rootless cgroup delegation is often
# unavailable and --cpus then errors out.
#
# RESUMABLE: run_is_sweep checkpoints after every weight, so a killed container loses at most one
# bin. To resume, `podman rm <name>` and re-run this script — it picks up from spectrum.json.
# Do NOT edit a config while its job is live: the guard at experiment_runner.py:642 aborts the
# resume on any weights_plan/seed change.
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
MOUNT_OPT="${MOUNT_OPT-:z}"         # SELinux relabel. LOWERCASE z = SHARED label: these
                                    # launchers start MULTIPLE containers over the SAME
                                    # src//experiments//runs mounts. Uppercase :Z applies a
                                    # PRIVATE unshared label, so the second container
                                    # RELABELS the volume and REVOKES the first one's
                                    # access mid-run -- it dies with ModuleNotFoundError on
                                    # code that existed seconds earlier. Set MOUNT_OPT= to
                                    # disable relabelling on a non-SELinux node.
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

launch b_idle experiments/configs/gross_lpu_idle_boost.yaml
launch b_auto experiments/configs/gross_automorphism_boost.yaml
launch b_y1   experiments/configs/gross_lpu_y1_boost.yaml

echo
echo "[run_lpu_boost] watch:  podman ps ; podman logs -f b_idle"
echo "[run_lpu_boost] resume: podman rm <name> && bash container/run_lpu_boost.sh --only <name>"
