#!/usr/bin/env bash
# Deepen the SYSTEM-LEVEL campaign's [[72,4,8]] onset bins (w=2..10) with the ghw
# decoder, MERGING into the sys cache in place -- onset_topup_72.py with the EMC envs.
# This is the step that turns the sys generation's Lambda from a truncation-inflated
# upper bound into a measurement (its 1x bins saw 0-in-10k where the baseline's
# topped-up bins measured 1e-6-class f(w)).
#
#   bash container/run_sys_topup.sh --dry-run
#   bash container/run_sys_topup.sh --list        # print the index -> spectrum map
#   bash container/run_sys_topup.sh               # detached, default = the 6 models
#   bash container/run_sys_topup.sh --indices "11 12 13 14 15 16"   # the x5-asym family
#
# MERGE SEMANTICS: onset_topup_72.py pools (trials += / failures +=) into the sys dir's
# tech1_72/asym spectra, per-weight-atomic and idempotent (a completed weight is
# recorded in the JSON's onset_topup block and skipped on rerun). Kill + relaunch any
# time. It asserts the cached decoder calibrated_at and the expansion identity before
# touching anything.
#
# COST (ghw decoder, 24 threads; a CAPPED bin = ONSET_SHOTS_MAX shots): full symmetric
# ~250 dec/s => ~3.3 h per capped 3e6 bin; CZ ~110/s => ~7.5 h; meas/prep/meas-idle
# ~60/s => ~14 h; gate idle ~25/s => ~33 h. With ghw's lower f(w), MORE bins run to cap
# than in the baseline top-up. All six models at the 3e6 default is likely ~1.5-2 WEEKS
# sequential -- either accept (checkpointed, stop anytime, cheapest-first order), or cap
# down: ONSET_SHOTS_MAX=1e6 bounds zero bins at 3e-6 for ~1/3 the time; 3e5 => 1e-5.
# Order below is cheapest-first so an early stop still leaves complete cheap models.
#
# INDEX MAP (onset_topup_72.py --list): 0 full_symmetric, 1 CZ_only, 2 meas_only,
# 3 prep_only, 4 gate_idle, 5 meas_idle, 6-10 tech1_72_abl, 11-16 asym_72.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

SYS_RESULTS="${EMC_RESULTS:-error_model_comparison_18_4_4_sys_baseline18_ghw72}"
BASELINE_NAME="error_model_comparison_18_4_4"

# SAFETY: never point a ghw top-up at the baseline cache -- pooled ghw samples would
# silently corrupt the baseline estimator. No override; this is never correct.
if [[ "$SYS_RESULTS" == "$BASELINE_NAME" ]]; then
  echo "[refuse] EMC_RESULTS points at the BASELINE cache ($BASELINE_NAME);" >&2
  echo "         a ghw top-up must merge into the sys dir only." >&2
  exit 1
fi

DRY=0
LIST=0
INDICES="${TOPUP_INDICES:-0 1 5 2 3 4}"   # models, cheapest-first (gate idle last)
NAME="sys_topup72"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY=1; shift ;;
    --list)     LIST=1; shift ;;
    --indices)  [[ $# -ge 2 ]] || { echo "--indices needs a quoted list"; exit 1; }
                INDICES="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:z}"
CPUS="${CPUS:-24}"

if [[ $LIST -eq 1 ]]; then
  podman run --rm \
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
    -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
    python experiments/methods/onset_topup_72.py --list
  exit 0
fi

LOOP="for i in ${INDICES}; do python -u experiments/methods/onset_topup_72.py --index \$i || exit 1; done"

# PREFLIGHT: the split must resolve to ghw on BOTH slots (onset_topup decodes via
# DEC()'s default = the 18-slot config, so EMC_DECODER must set both), and the results
# dir must be the sys cache with its tasks present.
if [[ $DRY -eq 0 ]]; then
  echo -n "[preflight] ghw decoder + sys cache resolve in the container... "
  set +e
  PF_OUT=$(podman run --rm \
       -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
       -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
       -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}" \
       -e EMC_DECODER=ghw -e "EMC_RESULTS=${SYS_RESULTS}" \
       -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
       python -c "import sys; sys.path.insert(0, 'experiments/methods')
import run_error_model_comparison as r
assert (r.DECODER_18, r.DECODER_72) == ('ghw', 'ghw'), (r.DECODER_18, r.DECODER_72)
assert r.DEC_CFG_18['pre_iter'] == 320 and r.DEC_CFG_18['gamma_dist_interval'] == (-0.5, 1.0)
assert r.RESULTS.name == '${SYS_RESULTS}', r.RESULTS
assert (r.RESULTS / 'tech1_72__full_symmetric.json').exists(), 'sys cache tasks missing'
print('ok: ghw both slots, sys cache present')" 2>&1)
  PF_RC=$?
  set -e
  if [[ $PF_RC -eq 0 ]]; then
    echo "OK"
  else
    echo "FAILED (exit $PF_RC)"
    printf '  %s\n' "$PF_OUT" >&2
    exit 1
  fi
fi

set +e
EXISTING=$(podman ps -a --filter "name=^${NAME}$" --format '{{.Status}}' 2>/dev/null | head -1)
set -e
if [[ -n "$EXISTING" ]]; then
  if [[ "$EXISTING" == Up* ]]; then
    echo "[skip] ${NAME} is ALREADY RUNNING (${EXISTING})"
    exit 0
  fi
  echo "[reuse] removing stale ${NAME}; completed weights are skipped on resume"
  podman rm "$NAME" >/dev/null 2>&1 || true
fi

CMD=( podman run -d --name "$NAME"
  -e "OMP_NUM_THREADS=${CPUS}" -e "OPENBLAS_NUM_THREADS=${CPUS}"
  -e "MKL_NUM_THREADS=${CPUS}" -e "RAYON_NUM_THREADS=${CPUS}"
  -e EMC_DECODER=ghw -e "EMC_RESULTS=${SYS_RESULTS}"
  -e "ONSET_SHOTS_MAX=${ONSET_SHOTS_MAX:-3000000}"
  -e "ONSET_TARGET=${ONSET_TARGET:-20}"
  -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}"
  -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
  -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
  -e PYTHONDONTWRITEBYTECODE=1
  -w /opt/stim_work "${IMAGE}"
  bash -c "$LOOP" )
echo "[launch] ${NAME}  (threads=${CPUS})  indices: ${INDICES}  cap=${ONSET_SHOTS_MAX:-3000000}"
if [[ $DRY -eq 1 ]]; then
  printf '    %q ' "${CMD[@]}"; echo
else
  "${CMD[@]}"
  echo "[run_sys_topup] watch:  podman logs -f ${NAME}"
  echo "[run_sys_topup] pull:   rsync the sys results dir home as before (bins merge in place)"
fi
exit 0
