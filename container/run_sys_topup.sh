#!/usr/bin/env bash
# Deepen the SYSTEM-LEVEL campaign's [[72,4,8]] onset bins (w=2..10) with the ghw
# decoder, MERGING into the sys cache in place -- onset_topup_72.py with the EMC envs.
# This is the step that turns the sys generation's Lambda from a truncation-inflated
# upper bound into a measurement (its 1x bins saw 0-in-10k where the baseline's
# topped-up bins measured 1e-6-class f(w)).
#
#   bash container/run_sys_topup.sh --dry-run
#   bash container/run_sys_topup.sh --list        # print the index -> spectrum map
#   bash container/run_sys_topup.sh               # 3 parallel shards x 24 threads = 72 cores
#   bash container/run_sys_topup.sh --indices "11 12 13 14 15 16"     # one container
#   bash container/run_sys_topup.sh --shards "11 12 13|14 15 16"      # custom parallel split
#
# PARALLELISM: one detached container per shard (disjoint index groups; each index owns
# its spectrum JSON, so shards never write the same file). Default = 3 shards balanced
# by cost x 24 threads = 72 of 96 cores; scale with CPUS= and/or --shards.
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
# TOPUP_DECODER: decoder variant for the top-up (must match the campaign that filled
# EMC_RESULTS — the topup script itself asserts calibrated_at + expansion identity).
TU_DEC="${TOPUP_DECODER:-ghw}"

# SAFETY: never point a ghw top-up at the baseline cache -- pooled ghw samples would
# silently corrupt the baseline estimator. No override; this is never correct.
if [[ "$SYS_RESULTS" == "$BASELINE_NAME" ]]; then
  echo "[refuse] EMC_RESULTS points at the BASELINE cache ($BASELINE_NAME);" >&2
  echo "         a ghw top-up must merge into the sys dir only." >&2
  exit 1
fi

DRY=0
LIST=0
# PARALLEL SHARDS: pipe-separated index groups, one detached container per group,
# CPUS threads each (default 3 x 24 = 72 cores). Safe because each index is its own
# spectrum JSON -- disjoint groups never touch the same file. The default split
# balances the per-model cost estimates (gate idle ~100h | meas+prep ~84h |
# full+CZ+meas_idle ~75h at the 3e6 cap), so the shards finish together-ish.
SHARDS="${TOPUP_SHARDS:-4|2 3|0 1 5}"
NAME="sys_topup72"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY=1; shift ;;
    --list)     LIST=1; shift ;;
    --indices)  [[ $# -ge 2 ]] || { echo "--indices needs a quoted list"; exit 1; }
                SHARDS="$2"; shift 2 ;;      # single shard, one container
    --shards)   [[ $# -ge 2 ]] || { echo "--shards needs a quoted pipe-separated list"; exit 1; }
                SHARDS="$2"; shift 2 ;;
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

# (shard loops are built per shard at launch time, below)

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
       -e "EMC_DECODER=${TU_DEC}" -e "EMC_RESULTS=${SYS_RESULTS}" \
       -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
       python -c "import sys; sys.path.insert(0, 'experiments/methods')
import run_error_model_comparison as r
assert (r.DECODER_18, r.DECODER_72) == ('${TU_DEC}', '${TU_DEC}'), (r.DECODER_18, r.DECODER_72)
if '${TU_DEC}' == 'ghw':
    assert r.DEC_CFG_18['pre_iter'] == 320 and r.DEC_CFG_18['gamma_dist_interval'] == (-0.5, 1.0)
if '${TU_DEC}' == 'ghw_deep':
    assert r.DEC_CFG_18['pre_iter'] == 640 and r.DEC_CFG_18['num_sets'] == 1200
assert r.RESULTS.name == '${SYS_RESULTS}', r.RESULTS
assert (r.RESULTS / 'tech1_72__full_symmetric.json').exists(), 'sys cache tasks missing'
print('ok: ${TU_DEC} both slots, cache present')" 2>&1)
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

# SANITY: shard index groups must be disjoint (two containers pooling into the same
# spectrum file would race and double-merge).
ALL_IDX=$(echo "${SHARDS}" | tr '|' ' ')
DUP=$(echo ${ALL_IDX} | tr ' ' '\n' | sort | uniq -d | head -1)
if [[ -n "$DUP" ]]; then
  echo "[refuse] index ${DUP} appears in more than one shard - shards must be disjoint" >&2
  exit 1
fi

FAILED=()
n=0
IFS='|' read -ra GROUPS_ARR <<< "$SHARDS"
for GROUP in "${GROUPS_ARR[@]}"; do
  GROUP=$(echo $GROUP)                       # trim
  [[ -z "$GROUP" ]] && continue
  n=$((n + 1))
  SNAME="${NAME}_s${n}"
  LOOP="for i in ${GROUP}; do python -u experiments/methods/onset_topup_72.py --index \$i || exit 1; done"

  set +e
  EXISTING=$(podman ps -a --filter "name=^${SNAME}$" --format '{{.Status}}' 2>/dev/null | head -1)
  set -e
  if [[ -n "$EXISTING" ]]; then
    if [[ "$EXISTING" == Up* ]]; then
      echo "[skip]   ${SNAME} is ALREADY RUNNING (${EXISTING}) - leaving it alone"
      continue
    fi
    echo "[reuse]  removing stale ${SNAME}; completed weights are skipped on resume"
    podman rm "$SNAME" >/dev/null 2>&1 || true
  fi

  CMD=( podman run -d --name "$SNAME"
    -e "OMP_NUM_THREADS=${CPUS}" -e "OPENBLAS_NUM_THREADS=${CPUS}"
    -e "MKL_NUM_THREADS=${CPUS}" -e "RAYON_NUM_THREADS=${CPUS}"
    -e "EMC_DECODER=${TU_DEC}" -e "EMC_RESULTS=${SYS_RESULTS}"
    -e "ONSET_SHOTS_MAX=${ONSET_SHOTS_MAX:-3000000}"
    -e "ONSET_TARGET=${ONSET_TARGET:-20}"
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}"
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
    -e PYTHONDONTWRITEBYTECODE=1
    -w /opt/stim_work "${IMAGE}"
    bash -c "$LOOP" )
  echo "[launch] ${SNAME}  (threads=${CPUS})  indices: ${GROUP}  cap=${ONSET_SHOTS_MAX:-3000000}"
  if [[ $DRY -eq 1 ]]; then
    printf '    %q ' "${CMD[@]}"; echo
  else
    set +e
    "${CMD[@]}"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[error]  ${SNAME} failed to start (exit ${rc})" >&2
      FAILED+=("$SNAME")
    fi
  fi
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[run_sys_topup] ${#FAILED[@]} shard(s) FAILED to start: ${FAILED[*]}" >&2
fi
echo "[run_sys_topup] ${n} shard(s) x ${CPUS} threads = $((n * CPUS)) cores"
echo "[run_sys_topup] watch:  podman ps ; podman logs -f ${NAME}_s1"
echo "[run_sys_topup] pull:   rsync the sys results dir home as before (bins merge in place)"
[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
