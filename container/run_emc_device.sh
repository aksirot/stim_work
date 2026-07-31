#!/usr/bin/env bash
# Device-calibration campaign (EMC_CALIB=device) on rodan -- reruns the sub-model /
# ablation / asym-ablation tasks with ONE decoder per code, calibrated on that code's
# FULL SYMMETRIC circuit (the device). Replaces the retired per-model-calibrated
# sub-model data (runs/_retired_permodel_calib/README.md, commit 3916611b): a decoder
# whose priors only know one channel exists is not the device's decoder, so the legacy
# rows never answered the error-budget question.
#
#   bash container/run_emc_device.sh --dry-run   # print the podman command only
#   bash container/run_emc_device.sh --smoke     # foreground --list inside the container
#   bash container/run_emc_device.sh             # seed + launch detached (CPUS=30 default)
#
# SEEDING: full-symmetric and asym-full results are IDENTICAL under both conventions
# (they calibrate on themselves; their task configs are deliberately unmarked). This
# script seeds the fresh device dir from the sys dir's kept files so only the ~51
# changed tasks recompute. RE-SEED tech1_72__full_symmetric.json after the sys top-up
# finishes merging (it keeps improving in the sys dir).
#
# DECODERS: same system-level pair as run_emc_ghw.sh (18=baseline, 72=ghw); same
# refusal of ghw on the 18-code. Box rules in docs/RODAN_QUICKSTART.md.
set -euo pipefail
cd "$(dirname "$0")/.."             # repo root
REPO="$PWD"

DRY=0
SMOKE=0
EXTRA="${EMC_EXTRA:-}"
NAME="emc_device"
VAR18="${EMC_DECODER_18:-baseline}"
VAR72="${EMC_DECODER_72:-ghw}"
DEV_RESULTS="${EMC_RESULTS:-error_model_comparison_18_4_4_device_${VAR18}18_${VAR72}72}"
SEED_FROM="${EMC_SEED_FROM:-error_model_comparison_18_4_4_sys_${VAR18}18_${VAR72}72}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --smoke)   SMOKE=1; shift ;;
    --extra)   [[ $# -ge 2 ]] || { echo "--extra needs a quoted flag string"; exit 1; }
               EXTRA="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

if [[ "$VAR18" == "ghw" && "${EMC_FORCE_18:-0}" != "1" ]]; then
  echo "[refuse] EMC_DECODER_18=ghw: the wide gamma interval breaks the 18-code" >&2
  exit 1
fi

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:z}"
CPUS="${CPUS:-30}"
mkdir -p "runs/${DEV_RESULTS}"

# SEED convention-invariant results from the sys campaign dir (skip files already
# present -- a later re-seed of the topped-up full-symmetric spectrum is done by hand).
if [[ -d "runs/${SEED_FROM}" ]]; then
  N_SEED=0
  for f in runs/${SEED_FROM}/*.json; do
    b=$(basename "$f")
    case "$b" in
      tech1__full_symmetric.json|tech2__*.json|tech3__full_symmetric.json|\
      mc__full_symmetric.json|tech1_72__full_symmetric.json|tech2_72__*.json|\
      mc72__full_symmetric.json|asym__full_*.json|setup__*.json|schedule__*.json)
        if [[ ! -e "runs/${DEV_RESULTS}/$b" ]]; then
          cp "$f" "runs/${DEV_RESULTS}/$b"; N_SEED=$((N_SEED+1))
        fi ;;
    esac
  done
  echo "[seed] copied ${N_SEED} convention-invariant results from ${SEED_FROM}"
else
  echo "[seed] WARNING: runs/${SEED_FROM} not found -- device run recomputes EVERYTHING"
fi

# PREFLIGHT: EMC_CALIB=device + decoder variants must resolve inside the container.
if [[ $DRY -eq 0 ]]; then
  echo -n "[preflight] device calibration + decoders resolve in the container... "
  set +e
  PF_OUT=$(podman run --rm \
       -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
       -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
       -e "EMC_CALIB=device" \
       -e "EMC_DECODER_18=${VAR18}" -e "EMC_DECODER_72=${VAR72}" -e "EMC_RESULTS=${DEV_RESULTS}" \
       -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
       python -c "import sys; sys.path.insert(0, 'experiments/methods')
import run_error_model_comparison as r
assert r.CALIB_MODE == 'device', r.CALIB_MODE
assert (r.DECODER_18, r.DECODER_72) == ('${VAR18}', '${VAR72}'), (r.DECODER_18, r.DECODER_72)
assert r.RESULTS.name == '${DEV_RESULTS}', r.RESULTS
print('device calib ok: 18=' + r.DECODER_18 + ' 72=' + r.DECODER_72)" 2>&1)
  PF_RC=$?
  set -e
  if [[ $PF_RC -eq 0 ]]; then
    echo "OK"
  else
    echo "FAILED (exit $PF_RC)"
    printf '  %s\n' "$PF_OUT" >&2
    echo "  Common causes: repo too old (git pull --ff-only; needs commit 3916611b+)," >&2
    echo "  image missing, SELinux relabel refused (retry with MOUNT_OPT=)." >&2
    exit 1
  fi
fi

if [[ $SMOKE -eq 1 ]]; then
  podman run --rm -t \
    -e "EMC_CALIB=device" \
    -e "EMC_DECODER_18=${VAR18}" -e "EMC_DECODER_72=${VAR72}" -e "EMC_RESULTS=${DEV_RESULTS}" \
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}" \
    -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
    python -u experiments/methods/run_error_model_comparison.py --list ${EXTRA}
  echo "[smoke] seeded tasks read cached; sub-models/ablations read missing. Safe to launch."
  exit 0
fi

if command -v loginctl >/dev/null 2>&1; then
  set +e
  LINGER=$(loginctl show-user "$USER" --property=Linger 2>/dev/null | cut -d= -f2)
  KILL=$(grep -sE '^ *KillUserProcesses' /etc/systemd/logind.conf 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' ')
  set -e
  if [[ "$LINGER" != "yes" && "${KILL:-no}" == "yes" ]]; then
    echo "[warn] Linger=$LINGER, KillUserProcesses=$KILL -- container dies at logout;"
    echo "       run: loginctl enable-linger $USER   (then relaunch)"
  fi
fi

set +e
EXISTING=$(podman ps -a --filter "name=^${NAME}$" --format '{{.Status}}' 2>/dev/null | head -1)
set -e
if [[ -n "$EXISTING" ]]; then
  if [[ "$EXISTING" == Up* ]]; then
    echo "[skip] ${NAME} is ALREADY RUNNING (${EXISTING}) -- leaving it alone"
    exit 0
  fi
  echo "[reuse] removing stale ${NAME} (${EXISTING}); tasks resume from the per-task cache"
  podman rm "$NAME" >/dev/null 2>&1 || true
fi

CMD=( podman run -d --name "$NAME"
  -e "OMP_NUM_THREADS=${CPUS}" -e "OPENBLAS_NUM_THREADS=${CPUS}"
  -e "MKL_NUM_THREADS=${CPUS}" -e "RAYON_NUM_THREADS=${CPUS}"
  -e "EMC_CALIB=device"
  -e "EMC_DECODER_18=${VAR18}" -e "EMC_DECODER_72=${VAR72}" -e "EMC_RESULTS=${DEV_RESULTS}"
  -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}"
  -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
  -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
  -e PYTHONDONTWRITEBYTECODE=1
  -w /opt/stim_work "${IMAGE}"
  python -u experiments/methods/run_error_model_comparison.py ${EXTRA} )
echo "[launch] ${NAME}  (threads=${CPUS})  EMC_CALIB=device  decoders 18=${VAR18} 72=${VAR72}"
if [[ $DRY -eq 1 ]]; then
  printf '    %q ' "${CMD[@]}"; echo
else
  "${CMD[@]}"
  echo "[run_emc_device] watch:  podman logs -f ${NAME}"
  echo "[run_emc_device] pull:   rsync -av cluster:workspace/stim_work_emc/runs/${DEV_RESULTS}/ runs/${DEV_RESULTS}/"
fi
exit 0
