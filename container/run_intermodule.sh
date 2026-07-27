#!/usr/bin/env bash
# Launch the Wave-6i gross-to-gross inter-module campaigns (X1(A)xX1(B)) on a NO-SCHEDULER,
# SHARED box (rodan: 96 cores, no sbatch/srun). Sibling of container/run_lpu_boost.sh.
#
#   bash container/run_intermodule.sh --dry-run   # print the podman commands, launch nothing
#   bash container/run_intermodule.sh             # launch both legs, detached
#   bash container/run_intermodule.sh --smoke     # validate first (foreground, ~15-30 min)
#   bash container/run_intermodule.sh --only r1   # substring filter on the job name
#
# WHY NOT run_local.sh: that launcher bind-mounts only runs/, so it would run against the image's
# BAKED experiments/ and never see these configs. This one mounts experiments/ too.
#
# ⚠️ WHY THIS ALSO MOUNTS src/ (run_lpu_boost.sh does NOT, and does not need to):
# the Containerfile BAKES the code (`COPY src/ ./src/`), and the shipped image is from 2026-07-24.
# build_joint_x1x1_circuit and its close_cycles fix both landed 2026-07-27, so the image has NO
# inter-module builder at all — without this mount every job dies immediately on an import/attribute
# error. `pip install -e .` resolves the package to /opt/stim_work/src, so bind-mounting the host
# src/ over it is enough; no image rebuild needed. Drop this mount only once a rebuilt image that
# contains the builder has actually been loaded on the target box.
#
# SHARED-BOX BUDGET: CPUS=24 per job x 2 jobs = 48 of 96 cores — HALF THE BOX, deliberately, set
# 2026-07-27. This is the only campaign running: the Wave-5b LPU boost pass is NOT being launched,
# so its 24 cores are not in play and the whole allocation goes here. Half is a self-imposed cap on
# a SHARED machine, not a machine limit — do not raise it to fill the box. Override for a quieter
# or busier machine with e.g. CPUS=16. CHECK WHO ELSE IS ON before launching.
#
# COST (measured 2026-07-27 at the real geometry, not estimated): decode rate saturates at
# ~1.84 s/shot from w=150 up (0.44 at w=5); onset ~w=100; 52% of the sweep is sub-onset bins
# capped at 3000 shots. ~35 h per leg at 24 threads => BOTH LEGS IN PARALLEL ~1.5 DAYS WALL.
#
# MEMORY: this is the largest circuit in the campaign -- ~418k DEM mechanisms, 10903 detectors at
# the production geometry (C=10, d_init=12, idle ON). Measured peak RSS 9.0 GB per job, so ~18 GB
# for the pair: the manifest's 64G figure has ~7x headroom and is NOT the risk it was assumed to
# be. Still watch `podman stats` on the first run — there is no scheduler here to enforce a cap,
# so an OOM would take down whatever else shares the box, not just this job.
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
SMOKE=0
ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --smoke)   SMOKE=1; shift ;;
    --only)    [[ $# -ge 2 ]] || { echo "--only needs a substring"; exit 1; }
               ONLY+=("$2"); shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
MOUNT_OPT="${MOUNT_OPT-:Z}"         # SELinux relabel; rodan NEEDS this (denies the mount without it)
CPUS="${CPUS:-24}"                  # threads per job — 2 x 24 = 48 = half the box; see header
mkdir -p runs/framework

# PREFLIGHT: the shipped image predates this builder (see the src/ mount note above). Verify the
# mounted code actually resolves INSIDE the container before committing to a multi-minute DEM
# build followed by an import error. Costs a couple of seconds and catches the exact failure mode
# that the baked-src image would produce.
if [[ $DRY -eq 0 ]]; then
  echo -n "[preflight] inter-module builder visible in the container... "
  # Capture stderr instead of discarding it: an earlier version sent it to /dev/null, so a
  # failure printed "FAILED" and threw away the podman/python error that says WHY — leaving
  # a silent no-launch with nothing to debug. Never suppress the diagnostic you will need.
  # set -e aborts on a failing command substitution in an assignment, BEFORE PF_RC is read,
  # so the whole diagnostic below would be skipped. Disable it across the capture.
  set +e
  PF_OUT=$(podman run --rm \
       -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
       -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
       python -c "from gross_code_lpu_tdg import build_joint_x1x1_circuit as f
import inspect; assert 'close_cycles' in inspect.signature(f).parameters
print('ok')" 2>&1)
  PF_RC=$?
  set -e
  if [[ $PF_RC -eq 0 ]]; then
    echo "OK"
  else
    echo "FAILED (exit $PF_RC)"
    echo "  The container could not import build_joint_x1x1_circuit (with close_cycles)." >&2
    echo "  ---- podman/python said ----" >&2
    printf '  %s\n' "$PF_OUT" >&2
    echo "  ----------------------------" >&2
    echo "  Common causes:" >&2
    echo "    * repo not on main / too old  -> git rev-parse --short HEAD; git pull --ff-only origin main" >&2
    echo "    * image missing               -> podman images | grep stim-work-qec" >&2
    echo "    * SELinux relabel refused     -> retry with MOUNT_OPT= (drops the :Z suffix)" >&2
    echo "  Not launching." >&2
    exit 1
  fi
fi

# SMOKE: same mounts as the real launch, foreground, small thread count, --smoke. Exists so the
# validation step is one short command instead of a dozen pasted flags — a wrapped paste of the
# long form silently mangles the backslash continuations and podman then sees a bare `-e`.
if [[ $SMOKE -eq 1 ]]; then
  SCPUS="${CPUS_SMOKE:-8}"
  echo "[smoke] gross_intermodule_r1 at ${SCPUS} threads — builds the REAL production"
  echo "        circuit (C=10, d_init=12, ~418k mechanisms) on tiny budgets. ~3 min at 8"
  echo "        threads (measured): the DEM build dominates, the w=2..10 sweep is cheap."
  echo "        EXPECTED: 0 failures in every bin and a \"ansatz fit failed ... all-zero"
  echo "        spectrum\" WARN — w=2..10 sits far below the onset (~w=100). Not an error."
  # `set -e` would abort before RC is read, so the failure diagnostic below would never print.
  set +e
  podman run --rm -t \
    -e "OMP_NUM_THREADS=${SCPUS}" -e "RAYON_NUM_THREADS=${SCPUS}" \
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}" \
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}" \
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}" \
    -e PYTHONDONTWRITEBYTECODE=1 -w /opt/stim_work "${IMAGE}" \
    python -u -m experiment_runner \
      --config experiments/configs/gross_intermodule_r1.yaml --smoke --cpus "${SCPUS}"
  RC=$?
  set -e
  OUT="runs/framework/bb144/inter_module_smoke/result.npz"
  if [[ $RC -eq 0 && -f "$OUT" ]]; then
    echo "[smoke] PASS — ${OUT} exists. Safe to launch."
  else
    echo "[smoke] FAIL — exit ${RC}; ${OUT} $( [[ -f "$OUT" ]] && echo exists || echo missing )" >&2
    exit 1
  fi
  exit 0
fi

# OVERNIGHT SURVIVAL: rootless podman containers are children of your login session. If
# systemd lingering is OFF and logind is configured to KillUserProcesses, they are killed the
# moment you disconnect — a multi-day unattended run would silently die at logout. Warn (do not
# block: the answer depends on site policy, and per-weight checkpointing makes a death recoverable).
if command -v loginctl >/dev/null 2>&1; then
  # set +e is ESSENTIAL: `grep` returns 1 when it matches nothing (the normal case —
  # KillUserProcesses is usually commented out), pipefail propagates that through the
  # pipeline, and set -e then kills the script HERE, silently, after the preflight has
  # already printed OK and before anything launches. That is precisely the bug this block
  # shipped with. Any command substitution in this script needs the same guard.
  set +e
  LINGER=$(loginctl show-user "$USER" --property=Linger 2>/dev/null | cut -d= -f2)
  KILL=$(grep -sE '^ *KillUserProcesses' /etc/systemd/logind.conf 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' ')
  set -e
  if [[ "$LINGER" != "yes" && "${KILL:-no}" == "yes" ]]; then
    echo "[warn] Linger=$LINGER and KillUserProcesses=$KILL — these containers will be KILLED"
    echo "       when you log out. Before an unattended run: loginctl enable-linger $USER"
    echo "       (may need admin). Or verify empirically: launch, log out, log back in, podman ps."
  elif [[ "$LINGER" != "yes" ]]; then
    echo "[note] Linger=${LINGER:-unknown}; KillUserProcesses=${KILL:-unset (default no)}."
    echo "       Containers SHOULD survive logout. Confirm once: log out, back in, podman ps."
  fi
fi

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
    -v "${REPO}/src:/opt/stim_work/src${MOUNT_OPT}"
    -v "${REPO}/experiments:/opt/stim_work/experiments${MOUNT_OPT}"
    -v "${REPO}/runs:/opt/stim_work/runs${MOUNT_OPT}"
    -e PYTHONDONTWRITEBYTECODE=1        # host __pycache__ may be Windows-written; don't mix
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
