#!/usr/bin/env bash
# Rerun the ENTIRE error-model-comparison campaign (run_error_model_comparison.py, all
# tasks: 18-code + [[72,4,8]], tech I/II/III, direct MC, ablations, asymmetric point) with
# the GHW decoder on rodan -- NO-SCHEDULER, SHARED box, rootless podman.
# Sibling of container/run_ghw_topup.sh / run_intermodule.sh; box rules in
# docs/RODAN_QUICKSTART.md (box_load.sh before launching, half-box policy, rootless
# `podman ps` shows only YOUR containers).
#
#   bash container/run_emc_ghw.sh --dry-run    # print the podman command, launch nothing
#   bash container/run_emc_ghw.sh --smoke      # foreground --list inside the container
#   bash container/run_emc_ghw.sh              # launch detached (CPUS=10 default)
#   bash container/run_emc_ghw.sh --extra "--only tech1_72 asym"   # forward runner flags
#
# DECODER (2026-07-28 FINAL, SYSTEM-LEVEL Λ): per-code best validated configs --
#   18-code: baseline DEC_CFG (its own best: zero symmetric sub-onset failures; the ghw
#            knobs buy it nothing, and ghw's wide gamma interval actively BREAKS it --
#            meas_idle single-fault f(1)=1.4e-2 + ~40% inflation at w=3).
#   72-code: ghw (full glo_heavy_wide, stop_nconv=5) -- 71/71 harvested sub-onset fixes,
#            44:8 paired mid-weight win over baseline.
# Λ from this run is SYSTEM-level (code + its decoder), not shared-decoder -- the report
# must say so. Fast-stopping variants (nc1/nc2) REJECTED outright: one-sided losses in
# the w>=12 ambiguity band. Paired evidence: runs/subonset_tune_72/. Do not switch
# variants mid-campaign; the EMC_DECODER_* envs exist for separate A/B dirs only.
#
# ISOLATION: EMC_DECODER_18/_72 + a split-named EMC_RESULTS dir (default
# error_model_comparison_18_4_4_sys_baseline18_ghw72). The runner HARD-REFUSES any
# non-baseline decoder without a dedicated EMC_RESULTS, so the baseline cache
# (runs/error_model_comparison_18_4_4/) cannot be resampled away. Pull results home to
# runs/cluster/, do NOT merge into the baseline dir.
#
# BUDGETS: 1x budgets, NO --boost72 (default EXTRA is empty). The boost's purpose --
# tighter 72-code onset bins -- is superseded by the run_ghw_topup.sh top-up pass
# (same nc5 decoder, deeper caps), which runs AFTER this completes. COST: nc5-GHW is
# ~2-4x slower than the old baseline -> expect ~12-15 h at CPUS=30 without the boost.
# Resumable: per-task cache, kill + relaunch continues where it stopped.
# 3-DAY SCHEDULE (sequential, 30 cores each): this script (~12-15 h), then the top-up
# launches in run_ghw_topup.sh's header (~45-50 h).
#
# MOUNTS src/ + experiments/ + runs/: the shipped image predates the methods scripts and
# the EMC_DECODER/EMC_RESULTS switches. pip install -e . resolves to /opt/stim_work/src,
# so bind-mounts suffice; no rebuild. Thread caps via env vars, not podman --cpus
# (rootless cgroup delegation unavailable).
set -euo pipefail
cd "$(dirname "$0")/.."             # repo root
REPO="$PWD"

DRY=0
SMOKE=0
EXTRA="${EMC_EXTRA:-}"
NAME="emc_ghw"
VAR18="${EMC_DECODER_18:-baseline}"
VAR72="${EMC_DECODER_72:-ghw}"
GHW_RESULTS="${EMC_RESULTS:-error_model_comparison_18_4_4_sys_${VAR18}18_${VAR72}72}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --smoke)   SMOKE=1; shift ;;
    --extra)   [[ $# -ge 2 ]] || { echo "--extra needs a quoted flag string"; exit 1; }
               EXTRA="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:z}"         # lowercase z (SHARED label) -- see run_intermodule.sh
CPUS="${CPUS:-30}"                  # sequential plan: full 30-core budget per container
mkdir -p "runs/${GHW_RESULTS}"

# PREFLIGHT: the decoder-variant switch must resolve inside the container. Seconds;
# catches the baked-src failure mode. Never discard the diagnostic.
if [[ $DRY -eq 0 ]]; then
  echo -n "[preflight] per-code decoders (18=${VAR18}, 72=${VAR72}) resolve in the container... "
  set +e
  PF_OUT=$(podman run --rm \
       -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
       -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
       -e "EMC_DECODER_18=${VAR18}" -e "EMC_DECODER_72=${VAR72}" -e "EMC_RESULTS=${GHW_RESULTS}" \
       -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
       python -c "import sys; sys.path.insert(0, 'experiments/methods')
import run_error_model_comparison as r
assert (r.DECODER_18, r.DECODER_72) == ('${VAR18}', '${VAR72}'), (r.DECODER_18, r.DECODER_72)
# 18 must NOT carry the wide gamma interval (it breaks a meas_idle single there)
if '${VAR18}' == 'baseline': assert r.DEC_CFG_18['gamma_dist_interval'] == (-0.24, 0.66), r.DEC_CFG_18
if '${VAR72}' == 'ghw':
    assert r.DEC_CFG_72['gamma_dist_interval'] == (-0.5, 1.0), r.DEC_CFG_72
    assert r.DEC_CFG_72['stop_nconv'] == 5 and r.DEC_CFG_72['pre_iter'] == 320, r.DEC_CFG_72
assert r.RESULTS.name == '${GHW_RESULTS}', r.RESULTS
print('decoders ok: 18=' + r.DECODER_18 + ' 72=' + r.DECODER_72)" 2>&1)
  PF_RC=$?
  set -e
  if [[ $PF_RC -eq 0 ]]; then
    echo "OK"
  else
    echo "FAILED (exit $PF_RC)"
    echo "  ---- podman/python said ----" >&2
    printf '  %s\n' "$PF_OUT" >&2
    echo "  ----------------------------" >&2
    echo "  Common causes: repo too old (git pull --ff-only), image missing, SELinux" >&2
    echo "  relabel refused (retry with MOUNT_OPT=). Not launching." >&2
    exit 1
  fi
fi

# SMOKE: --list inside the container with the variant envs -- proves plumbing + shows the
# task roster (all should read "missing" on a fresh dir). Foreground, seconds.
if [[ $SMOKE -eq 1 ]]; then
  podman run --rm -t \
    -e "EMC_DECODER_18=${VAR18}" -e "EMC_DECODER_72=${VAR72}" -e "EMC_RESULTS=${GHW_RESULTS}" \
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}" \
    -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
    python -u experiments/methods/run_error_model_comparison.py --list ${EXTRA}
  echo "[smoke] roster printed above (fresh dir => everything missing). Safe to launch."
  exit 0
fi

# Overnight survival (rootless containers die at logout without lingering).
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

# Stale-name handling: dead container keeps its name; a RUNNING one is left alone.
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
  -e "EMC_DECODER_18=${VAR18}" -e "EMC_DECODER_72=${VAR72}" -e "EMC_RESULTS=${GHW_RESULTS}"
  -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}"
  -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
  -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
  -e PYTHONDONTWRITEBYTECODE=1
  -w /opt/stim_work "${IMAGE}"
  python -u experiments/methods/run_error_model_comparison.py ${EXTRA} )
echo "[launch] ${NAME}  (threads=${CPUS})  decoders 18=${VAR18} 72=${VAR72}  runner flags: ${EXTRA}"
if [[ $DRY -eq 1 ]]; then
  printf '    %q ' "${CMD[@]}"; echo
else
  "${CMD[@]}"
  echo "[run_emc_ghw] watch:  podman logs -f ${NAME} ; podman stats --no-stream"
  echo "[run_emc_ghw] pull:   rsync -av cluster:stim_work/runs/${GHW_RESULTS}/ runs/cluster/${GHW_RESULTS}/"
  echo "[run_emc_ghw] resume: re-run this script (per-task cache; running container untouched)"
fi
exit 0
