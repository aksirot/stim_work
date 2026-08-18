#!/bin/bash
# Fish single-node launcher: N nohup'd splitting shards inside apptainer.
# No SLURM, no podman. Run from the repo root ON THE NODE, inside nothing:
#
#   bash container/fish_split_launch.sh gross 8      # 8 gross-idle shards
#   bash container/fish_split_launch.sh 72 6         # 6 [[72,4,8]] shards
#
# Budgets via env (see the drivers): GS_TINIT / FL_TINIT etc. Defaults here are the
# paper-scale test (T_init=1e5 -- 25x the laptop budget that failed arbitration).
# Each shard gets a distinct seed + tag; combine afterwards (locally) with
# splitting_shard_combine.py. SIF points at the built image on this cluster.
set -euo pipefail
TARGET="${1:?usage: fish_split_launch.sh {gross|72} N_SHARDS}"
N="${2:?usage: fish_split_launch.sh {gross|72} N_SHARDS}"
SIF="${SIF:-/tmp/bbcode.sif}"
THREADS="${THREADS:-20}"

unset XDG_RUNTIME_DIR                 # detached shards must not depend on the login session
[ -f "$SIF" ] || { echo "stage the image first: cp -n <your .sif> $SIF"; exit 1; }

if [ "$TARGET" = gross ]; then
  DRIVER=experiments/methods/splitting_gross_idle.py
  LOGDIR=runs/splitting_gross
else
  DRIVER=experiments/methods/splitting_fast_ladder_72.py
  LOGDIR=runs/splitting_crosscheck/fast_ladder
fi
mkdir -p "$LOGDIR"

for i in $(seq 0 $((N - 1))); do
  if [ "$TARGET" = gross ]; then
    ENVS="GS_TAG=s$i GS_SEED=$((100 + i)) GS_TINIT=${GS_TINIT:-1e5}"
    ARGS=""
  else
    ENVS="FL_TAG=s$i FL_SEED=$((100 + i)) FL_L=2 FL_M=3 FL_TINIT=${FL_TINIT:-1e5} FL_TCAP=${FL_TCAP:-4e5}"
    ARGS="ladder"
  fi
  nohup env $ENVS RAYON_NUM_THREADS=$THREADS \
    apptainer exec --bind "$PWD:/work" --pwd /work \
      --env "PYTHONPATH=/work/src:/work/experiments/methods" \
      "$SIF" python -u "$DRIVER" $ARGS \
    > "$LOGDIR/shard_$i.log" 2>&1 &
  echo "shard $i: pid $!"
done
echo "all $N shards launched; logs in $LOGDIR/shard_*.log; safe to log out"
