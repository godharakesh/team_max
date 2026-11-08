#!/bin/bash
# CPU arena evaluation of one deep-RL checkpoint on beast (cores 30,31 only).
# usage: offline/drl/arena_eval.sh <ckpt.npz> <out tag> <workers> <opponents,comma,sep> <seeds...>
# env: SCALE (BOT_BUDGET_SCALE, default 1.0), CORES (taskset list, default 30,31)
CK=$1; TAG=$2; W=$3; OPPS=$4; shift 4
cd "$(dirname "$0")/../.." || exit 1
mkdir -p ../arena
ARENA_WORKERS=$W BOT_BUDGET_SCALE=${SCALE:-1.0} taskset -c ${CORES:-30,31} ~/miniconda3/bin/python offline/arena.py \
    "nn:$CK" "$OPPS" 1000 "$@" >> "../arena/$TAG.txt" 2>&1
