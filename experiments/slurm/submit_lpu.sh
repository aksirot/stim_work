#!/bin/bash
# Submit the three Wave-5 gross-code [[144,12,12]] LPU campaigns on a SLURM + podman cluster.
# Each config runs as its OWN independent job INSIDE the stim-work-qec image, with its own
# cpus/time/mem. Targets the configs BY PATH (no manifest-index coupling), so it's a one-shot for
# reproducing the local Wave-5 runs (idle / shift-automorphism / Y1 joint-Pauli). The three jobs
# are independent and run in parallel — unlike the rodan base->top-up chain.
#
#   bash experiments/slurm/submit_lpu.sh              # submit all three
#   bash experiments/slurm/submit_lpu.sh --dry-run    # print the sbatch commands, submit nothing
#
# Prereqs on the cluster (same as the onset/benchmark container jobs):
#   * image loaded:  gunzip -c qec_image.tar.gz | podman load     (-> localhost/stim-work-qec:latest)
#   * if rootless podman has no subuid/subgid: ~/.config/containers/storage.conf with
#     ignore_chown_errors=true + mount_program=/usr/bin/fuse-overlayfs  (see cluster-podman notes).
#   * runs/ + experiments/ on a shared filesystem the compute nodes can see.
#
# Tunables (env):  QEC_IMAGE (localhost/stim-work-qec:latest),  MOUNT_OPT (:Z SELinux relabel; set
#   MOUNT_OPT= empty if the cluster isn't SELinux-enforcing and rejects it).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
REPO="$PWD"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:Z}"         # ${VAR-default}: keep an explicitly-empty MOUNT_OPT
mkdir -p runs/slurm runs/framework

submit() {  # name  config-path  cpus  time  mem
  local name="$1" cfg="$2" cpus="$3" time="$4" mem="$5"
  # One-line podman command (backslash-newlines are line continuations, joined into one string).
  local cmd="podman run --rm --cpus ${cpus} \
-e OMP_NUM_THREADS=${cpus} -e OPENBLAS_NUM_THREADS=${cpus} -e MKL_NUM_THREADS=${cpus} -e RAYON_NUM_THREADS=${cpus} \
-v ${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT} \
-v ${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT} \
-w /opt/stim_work ${IMAGE} \
python -m experiment_runner --config ${cfg} --cpus ${cpus}"
  echo "[submit] ${name}  (cpus=${cpus} time=${time} mem=${mem})  cfg=${cfg}"
  if [[ $DRY -eq 1 ]]; then
    echo "    sbatch --job-name=${name} --cpus-per-task=${cpus} --time=${time} --mem=${mem} --wrap=\"${cmd}\""
  else
    sbatch --job-name="${name}" \
           --output="runs/slurm/%x_%j.out" --error="runs/slurm/%x_%j.out" \
           --cpus-per-task="${cpus}" --time="${time}" --mem="${mem}" \
           --wrap="${cmd}"
  fi
}

# Resources per the Wave-5 manifest budgets (edit here if a fresh, uncached run hits TIMEOUT):
#         name                config                                        cpus  time      mem
submit gross_lpu_idle     experiments/configs/gross_lpu_idle.yaml     16 24:00:00 16G
submit gross_automorphism experiments/configs/gross_automorphism.yaml 16 24:00:00 16G
submit gross_lpu_y1       experiments/configs/gross_lpu_y1.yaml       32 48:00:00 32G

echo
echo "[submit_lpu] submitted (or dry-ran) 3 jobs — watch: squeue -u \$USER ; logs: runs/slurm/"
echo "[submit_lpu] each is resumable: re-run this to continue any that hit TIMEOUT (checkpoints persist in runs/)."
