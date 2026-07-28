#!/usr/bin/env bash
# Launch the GHW-decoder [[72,4,8]] spectrum run (weights 3..10, 3e6-shot cap) on rodan —
# NO-SCHEDULER, SHARED box, rootless podman. Sibling of container/run_intermodule.sh; see
# that script and docs/RODAN_QUICKSTART.md for the box rules (half-box cap, box_load.sh
# before launching, rootless `podman ps` only shows YOUR containers).
#
#   bash container/run_ghw_topup.sh --dry-run              # print commands, launch nothing
#   bash container/run_ghw_topup.sh --smoke                # tiny-budget validation, foreground
#   bash container/run_ghw_topup.sh                        # launch default list, detached
#   bash container/run_ghw_topup.sh --indices "1 3 4 5"    # explicit SPECTRA indices
#
# WHAT IT RUNS: experiments/methods/ghw_spectrum_72.py --index N, sequentially over the
# index list, in ONE detached container (decode is Rust-parallel across CPUS threads, so
# one container at CPUS threads is the efficient footprint; more containers would just
# contend). SPECTRA order (from onset_topup_72.py --list):
#   0 full_symmetric  1 CZ_only  2 meas_only  3 prep_only  4 gate_idle  5 meas_idle
#   (6-10 ablated, 11-16 x5 asym — valid too if the budget allows)
# Default list = the five cheap-to-affordable symmetric models, CHEAPEST FIRST under the
# nc1 decoder: gate idle, meas idle, meas, prep, then full symmetric (the long pole).
# CZ_only (index 1) is deliberately NOT in the default: its w=3 rate (~360 dec/s) makes a
# capped 1e8 bin ~3 days alone — add it explicitly if the budget allows.
#
# ROLE (2026-07-28 FINAL): the LATER follow-up to the run_emc_ghw.sh campaign — same
# nc5-GHW decoder (ONE decoder for every weight bin; the fast nc1/nc2 variants were
# rejected, see run_emc_ghw.sh header), deeper onset bins than the campaign's adaptive
# sweep. Default cap 3e6 shots/bin = the baseline top-up protocol (zero-bin bound 1e-6);
# raise per launch via GHW_SHOTS_MAX once budget allows — a later deepening run must use
# a fresh GHW_OUT_DIR + distinct GHW_SEED_BASE so bins pool without draw reuse.
#
# COST under nc5-GHW (local 24-thread rates; a CAPPED 3e6 bin, w=2 and w=3 the usual
# suspects): full symmetric ~3.3 h/bin (~250 dec/s), meas/prep/meas-idle ~11-14 h/bin
# (~60-75/s), gate idle ~33 h/bin (~25/s), CZ ~7.5 h/bin (~110/s). Early-stopped bins
# (>= 20 failures) are cheap. DEFAULT = ALL SIX MODELS, cheapest-first (full_sym, CZ,
# meas, prep, meas_idle, gate_idle) ~= 6.5 days sequential at 30 threads — the order
# front-loads the cheap spectra so stopping the container early (per-weight + intra-
# weight checkpoints make that lossless) still leaves complete models behind; gate idle
# alone is ~2.7 days of it. Trim per launch with --indices, or split across relaunches.
# Ablated (6-10) and x5-asym (11-16) spectra are valid indices too (baseline top-up
# covered all 17) — add them in a later pass.
#
# RESUMABLE: ghw_spectrum_72.py checkpoints INTRA-weight (per-chunk seeding, every 1M
# shots by default) into runs/ghw_topup_72/<name>.json and refuses to resume against
# changed budgets/decoder config. Kill + relaunch loses at most ~1M shots of decode.
#
# MOUNTS src/ + experiments/: the shipped image predates this script AND the methods/
# scripts it imports (run_error_model_comparison.py, onset_topup_72.py). Same reasoning
# as run_intermodule.sh — pip install -e . resolves to /opt/stim_work/src, so bind-mounts
# are enough, no rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."             # repo root
REPO="$PWD"

DRY=0
SMOKE=0
INDICES="${GHW_INDICES:-0 1 2 3 5 4}"  # all six models, cheapest-first, gate_idle last
                                       # (cost table above; SPECTRA order in the header)
NAME="ghw_topup72"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY=1; shift ;;
    --smoke)    SMOKE=1; shift ;;
    --indices)  [[ $# -ge 2 ]] || { echo "--indices needs a quoted list"; exit 1; }
                INDICES="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:z}"         # lowercase z (SHARED label) — see run_intermodule.sh
CPUS="${CPUS:-30}"                  # runs AFTER the campaign container: full 30-core budget
mkdir -p runs/ghw_topup_72

# The sequential index loop, as one shell command inside the container.
LOOP="for i in ${INDICES}; do python -u experiments/methods/ghw_spectrum_72.py --index \$i || exit 1; done"

# PREFLIGHT: verify the mounted methods scripts import inside the container (the baked
# image has neither ghw_spectrum_72.py nor its imports). Seconds, catches the exact
# baked-src failure mode. Capture stderr — never discard the diagnostic (learned the
# hard way in run_intermodule.sh).
if [[ $DRY -eq 0 ]]; then
  echo -n "[preflight] ghw_spectrum_72 imports inside the container... "
  set +e
  PF_OUT=$(podman run --rm \
       -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
       -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
       -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
       python -c "import sys; sys.path.insert(0, 'experiments/methods')
import ghw_spectrum_72 as g
assert g.GHW_CFG['pre_iter'] == 320 and g.GHW_CFG['gamma0'] == 0.0625
print('ok')" 2>&1)
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

# SMOKE: tiny budgets via env, foreground, first index only. ~2-5 min: DEM build + 8
# small bins. EXPECTED: mostly 0-failure bins at these budgets — plumbing test, not physics.
if [[ $SMOKE -eq 1 ]]; then
  SCPUS="${CPUS_SMOKE:-8}"
  FIRST=$(echo ${INDICES} | cut -d' ' -f1)
  echo "[smoke] index ${FIRST} at ${SCPUS} threads, GHW_SHOTS_MAX=2000 GHW_TARGET=1"
  set +e
  podman run --rm -t \
    -e "OMP_NUM_THREADS=${SCPUS}" -e "RAYON_NUM_THREADS=${SCPUS}" \
    -e GHW_SHOTS_MAX=2000 -e GHW_TARGET=1 -e GHW_CHUNK=1000 \
    -e GHW_OUT_DIR=runs/ghw_topup_72/_smoke \
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}" \
    -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
    python -u experiments/methods/ghw_spectrum_72.py --index "${FIRST}"
  RC=$?
  set -e
  if [[ $RC -eq 0 ]] && ls runs/ghw_topup_72/_smoke/*.json >/dev/null 2>&1; then
    echo "[smoke] PASS — runs/ghw_topup_72/_smoke/ has a checkpoint. Safe to launch."
    echo "[smoke] (remove runs/ghw_topup_72/_smoke/ before or after; production ignores it)"
  else
    echo "[smoke] FAIL — exit ${RC}; no _smoke checkpoint written" >&2
    exit 1
  fi
  exit 0
fi

# Overnight survival warning (rootless containers die at logout if lingering is off) —
# same logic as run_intermodule.sh, same set -e traps guarded.
if command -v loginctl >/dev/null 2>&1; then
  set +e
  LINGER=$(loginctl show-user "$USER" --property=Linger 2>/dev/null | cut -d= -f2)
  KILL=$(grep -sE '^ *KillUserProcesses' /etc/systemd/logind.conf 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' ')
  set -e
  if [[ "$LINGER" != "yes" && "${KILL:-no}" == "yes" ]]; then
    echo "[warn] Linger=$LINGER, KillUserProcesses=$KILL — container dies at logout;"
    echo "       run: loginctl enable-linger $USER   (then relaunch)"
  fi
fi

# Stale-name handling: a dead container keeps its name; a RUNNING one is left alone.
set +e
EXISTING=$(podman ps -a --filter "name=^${NAME}$" --format '{{.Status}}' 2>/dev/null | head -1)
set -e
if [[ -n "$EXISTING" ]]; then
  if [[ "$EXISTING" == Up* ]]; then
    echo "[skip] ${NAME} is ALREADY RUNNING (${EXISTING}) — leaving it alone"
    exit 0
  fi
  echo "[reuse] removing stale ${NAME} (${EXISTING}); weights resume from checkpoints"
  podman rm "$NAME" >/dev/null 2>&1 || true
fi

# GHW_* knobs forward into the container (host env at launch time -> worker env).
CMD=( podman run -d --name "$NAME"
  -e "OMP_NUM_THREADS=${CPUS}" -e "OPENBLAS_NUM_THREADS=${CPUS}"
  -e "MKL_NUM_THREADS=${CPUS}" -e "RAYON_NUM_THREADS=${CPUS}"
  -e "GHW_VARIANT=${GHW_VARIANT:-ghw}"
  -e "GHW_SHOTS_MAX=${GHW_SHOTS_MAX:-3e6}"
  -e "GHW_TARGET=${GHW_TARGET:-20}"
  -e "GHW_WEIGHTS=${GHW_WEIGHTS:-2..10}"
  -e "GHW_SAVE_EVERY=${GHW_SAVE_EVERY:-20}"
  -e "GHW_SEED_BASE=${GHW_SEED_BASE:-204}"
  -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}"
  -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
  -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
  -e PYTHONDONTWRITEBYTECODE=1
  -w /opt/stim_work "${IMAGE}"
  bash -c "$LOOP" )
echo "[launch] ${NAME}  (threads=${CPUS})  indices: ${INDICES}"
if [[ $DRY -eq 1 ]]; then
  printf '    %q ' "${CMD[@]}"; echo
else
  "${CMD[@]}"
  echo "[run_ghw_topup] watch:  podman logs -f ${NAME} ; podman stats --no-stream"
  echo "[run_ghw_topup] pull:   rsync -av cluster:stim_work/runs/ghw_topup_72/ runs/cluster/ghw_topup_72/"
  echo "[run_ghw_topup] resume: re-run this script (running container is left untouched)"
fi
exit 0
