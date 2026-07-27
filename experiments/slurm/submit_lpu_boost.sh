#!/bin/bash
# Submit the Wave-5 BOOST pass for the three gross-code [[144,12,12]] LPU campaigns on rodan
# (SLURM + podman). Deepens the shot budget of the completed local runs so the failure spectrum's
# zero-failure floor drops from the ~1e-3 rule-of-three limit toward 1e-6/1.5e-5/3e-6.
#
#   bash experiments/slurm/submit_lpu_boost.sh              # submit all three
#   bash experiments/slurm/submit_lpu_boost.sh --dry-run    # print the sbatch commands only
#   bash experiments/slurm/submit_lpu_boost.sh --only y1    # substring filter on the job name
#
# These are NEW configs at seed 43 writing to their OWN outdirs (*_boost), pooled with the parent
# spectra downstream via lambda_analysis.pool_spectra. They do NOT touch the completed runs.
#
# CAPS ARE PER CAMPAIGN AND ARE NOT INTERCHANGEABLE. run_is_sweep checkpoints per WEIGHT with no
# intra-bin save, so a bin that cannot finish inside one walltime NEVER lands — every requeue
# restarts it and the job silently burns the allocation on the same weight forever. Each cap is
# sized so the worst bin is ~21-25 h at 48 cores, i.e. ~1/4 of the 96 h wall:
#
#     campaign        onset w0   s/shot @48c   cap        stride  worst bin  floor 3/T  run @48c
#     idle                  39        0.025    3,000,000   1 -> 3    ~21 h     1.0e-6     ~4.0 d
#     automorphism          69        0.373      200,000   4 -> 8    ~21 h     1.5e-5     ~3.5 d
#     Y1                    25        0.089    1,000,000        6    ~25 h     3.0e-6     ~3.0 d
#
# The automorphism's cap is 15x smaller because its onset sits at w=69, where RelayBP costs
# ~2.24 s/shot, vs w=25 / 0.53 s/shot for Y1 — decode cost climbs steeply with fault weight, and
# the bins that hit the cap are always the sub-onset ones. See the config headers for the details.
#
# RESUMABILITY: idle is expected to need ONE requeue (~4.0 d against a 96 h wall); the other two
# should land in a single submission. Re-run the SAME line to resume — per-weight checkpoints mean
# at most one bin is lost. Do NOT edit a config while its job is live: the guard at
# experiment_runner.py:642 aborts the resume on any weights_plan/seed change.
#
# Prereqs on the cluster (see the cluster-podman notes):
#   * image loaded:  gunzip -c qec_image.tar.gz | podman load   (-> localhost/stim-work-qec:latest)
#   * rootless podman without subuid/subgid: ~/.config/containers/storage.conf with
#     ignore_chown_errors=true + mount_program=/usr/bin/fuse-overlayfs
#   * runs/ + experiments/ on a shared filesystem the compute nodes can see
#
# Tunables (env): QEC_IMAGE, MOUNT_OPT (set MOUNT_OPT= empty if the cluster rejects :Z).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
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
MOUNT_OPT="${MOUNT_OPT-:Z}"         # ${VAR-default}: keep an explicitly-empty MOUNT_OPT
mkdir -p runs/slurm runs/framework

matches() {  # does $1 contain any of the --only substrings? (no --only = match everything)
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local n="$1" s
  for s in "${ONLY[@]}"; do [[ "$n" == *"$s"* ]] && return 0; done
  return 1
}

submit() {  # name  config-path  cpus  time  mem
  local name="$1" cfg="$2" cpus="$3" time="$4" mem="$5"
  matches "$name" || return 0
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

#      name                     config                                              cpus time       mem
submit gross_lpu_idle_boost     experiments/configs/gross_lpu_idle_boost.yaml     48 96:00:00 32G
submit gross_automorphism_boost experiments/configs/gross_automorphism_boost.yaml 48 96:00:00 48G
submit gross_lpu_y1_boost       experiments/configs/gross_lpu_y1_boost.yaml       48 96:00:00 64G

echo
echo "[submit_lpu_boost] submitted (or dry-ran) — watch: squeue -u \$USER ; logs: runs/slurm/"
echo "[submit_lpu_boost] idle + automorphism are expected to need ONE requeue: re-run the same line."
