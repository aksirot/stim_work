#!/bin/bash
# Fish single-node splitting launcher — PODMAN edition (2026-08-19).
# Replaces the apptainer path: fish's podman store works post-reset, runs the full
# stack, and dodges both host-env traps (no $HOME bind, no env inheritance -> no
# conda shadowing, no FIPS-aware-crypto trigger). Detached rodan-style containers.
#
#   IMG=localhost/<name:tag> bash container/fish_split_podman.sh gross 8
#   IMG=localhost/<name:tag> bash container/fish_split_podman.sh 72 6
#
# Budgets via env: GS_TINIT/GS_PHIGH (gross), FL_TINIT/FL_TCAP (72). One-time on
# each machine: `loginctl enable-linger $USER` so containers survive logout.
# Watch: tail the shard logs; stop all: podman rm -f $(podman ps -q --filter name=split_)
set -euo pipefail
TARGET="${1:?usage: fish_split_podman.sh gross-or-72 N_SHARDS}"
N="${2:?usage: fish_split_podman.sh gross-or-72 N_SHARDS}"
IMG="${IMG:?set IMG=localhost/<name:tag> (see podman images)}"
THREADS="${THREADS:-20}"

if [ "$TARGET" = gross ]; then
  DRIVER=experiments/methods/splitting_gross_idle.py
  LOGDIR=runs/splitting_gross
else
  DRIVER="experiments/methods/splitting_fast_ladder_72.py ladder"
  LOGDIR=runs/splitting_crosscheck/fast_ladder
fi
mkdir -p "$LOGDIR"

for i in $(seq 0 $((N - 1))); do
  if [ "$TARGET" = gross ]; then
    EX=(-e "GS_TAG=s$i" -e "GS_SEED=$((100 + i))" -e "GS_TINIT=${GS_TINIT:-1e5}" \
        -e "GS_PHIGH=${GS_PHIGH:-2e-3}")
  else
    EX=(-e "FL_TAG=s$i" -e "FL_SEED=$((100 + i))" -e "FL_L=2" -e "FL_M=3" \
        -e "FL_TINIT=${FL_TINIT:-1e5}" -e "FL_TCAP=${FL_TCAP:-4e5}")
  fi
  podman rm -f "split_${TARGET}_s$i" >/dev/null 2>&1 || true
  podman run -d --name "split_${TARGET}_s$i" \
    -v "$PWD:/work" -w /work \
    -e "PYTHONPATH=/work/src:/work/experiments/methods" \
    -e "RAYON_NUM_THREADS=$THREADS" "${EX[@]}" \
    "$IMG" bash -c "python -u $DRIVER > $LOGDIR/shard_$i.log 2>&1"
  echo "shard $i: split_${TARGET}_s$i"
done
echo "all $N shards launched; logs in $LOGDIR/shard_*.log; check survival after a"
echo "logout/login with: podman ps   (enable-linger once if they die at logout)"
