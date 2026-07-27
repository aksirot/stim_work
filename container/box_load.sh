#!/usr/bin/env bash
# "Is there room for my jobs right now?" — for a NO-SCHEDULER shared box (rodan).
#
#   bash container/box_load.sh            # report + a recommended CPUS value
#   bash container/box_load.sh --want 48  # check whether a specific footprint fits
#
# With no scheduler there is nothing tracking allocations, so the ONLY truth is what is
# actually running. Read the 1-minute load average against nproc: that is the number of
# runnable threads, i.e. how much of the machine is genuinely in use right now.
#
# ⚠️ THE TRAP: podman is ROOTLESS here, so `podman ps` shows YOUR containers and nobody
# else's. A colleague can be using 60 cores through their own podman and your `podman ps`
# will be empty. Never conclude "the box is free" from podman alone — use loadavg and ps,
# which see every user's processes.
set -uo pipefail

WANT=0
[[ "${1:-}" == "--want" ]] && WANT="${2:-0}"

CORES=$(nproc)
read -r L1 L5 L15 _ < /proc/loadavg
# integer math only: bash has no floats
L1I=${L1%.*}
FREE=$(( CORES - L1I ))
(( FREE < 0 )) && FREE=0

echo "=== $(hostname) ==="
echo "cores            : ${CORES}"
echo "load 1/5/15 min  : ${L1} / ${L5} / ${L15}"
echo "≈ cores in use   : ${L1I}"
echo "≈ cores free     : ${FREE}"
echo
echo "memory (GB):"
if command -v free >/dev/null 2>&1; then
  free -g | awk 'NR<=2 {printf "  %s\n", $0}'
else
  echo "  (free unavailable — Linux only; fine on rodan)"
fi
echo
echo "top CPU consumers (ALL users — this is what podman ps cannot show you):"
if ps -eo user:12,pcpu,rss,etime,comm --sort=-pcpu >/dev/null 2>&1; then
  ps -eo user:12,pcpu,rss,etime,comm --sort=-pcpu | head -11 | awk '{printf "  %s\n", $0}'
else
  echo "  (ps -eo unavailable — Linux only; fine on rodan)"
fi
echo
echo "logged in:"
(who | awk '{print $1}' | sort -u | tr '\n' ' '; echo) | sed 's/^/  /'
echo
echo "YOUR containers (rootless — other users' are invisible here):"
if command -v podman >/dev/null 2>&1; then
  podman ps --format '  {{.Names}}  {{.Status}}  {{.Command}}' 2>/dev/null || echo "  (podman ps failed)"
  [[ -z "$(podman ps -q 2>/dev/null)" ]] && echo "  (none running)"
else
  echo "  (podman not found)"
fi

echo
echo "=== verdict ==="
HALF=$(( CORES / 2 ))
if (( WANT > 0 )); then
  if (( WANT <= FREE )); then
    echo "  requesting ${WANT} of ~${FREE} free — FITS"
  else
    echo "  requesting ${WANT} but only ~${FREE} free — WOULD OVERCOMMIT by $(( WANT - FREE ))"
    echo "  consider CPUS=$(( FREE / 2 )) per job for a 2-job campaign, or wait."
  fi
else
  BUDGET=$(( FREE < HALF ? FREE : HALF ))     # never exceed half the box, even if idle
  PER=$(( BUDGET / 2 ))
  (( PER < 1 )) && PER=1
  echo "  half-the-box policy cap : ${HALF}"
  echo "  usable now              : ${BUDGET}  (min of free and the policy cap)"
  echo "  => for the 2-leg Wave-6i campaign:  CPUS=${PER} bash container/run_intermodule.sh"
  if (( BUDGET < HALF )); then
    echo "  NOTE: the box is busier than the policy cap allows for; someone else is working."
  fi
fi
