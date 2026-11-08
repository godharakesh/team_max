#!/usr/bin/env bash
# usage: offline/run_match.sh <opponent.py> <seed> [deals] [port1] [port2]
# Plays agent.py vs an opponent exactly like the tournament: each half in fresh servers.
set -u
OPP=$1; SEED=$2; DEALS=${3:-1000}; P1=${4:-5008}; P2=${5:-5009}
PY=${PY:-.venv/bin/python}
OUT=${OUT:-offline/results}; mkdir -p "$OUT"
TAG="$(basename "$OPP" .py)_s${SEED}_p${P1}"
wait_up() { for i in $(seq 1 100); do curl -s "http://localhost:$1/" >/dev/null && return 0; sleep 0.2; done; return 1; }
for H in 1 2; do
  BOT_DEBUG=1 PORT=$P1 $PY agent.py >"$OUT/$TAG.a1.log" 2>&1 & A=$!
  PORT=$P2 $PY "$OPP" >"$OUT/$TAG.a2.log" 2>&1 & B=$!
  wait_up $P1; wait_up $P2
  $PY judge_engine.py --agent1 $P1 --agent2 $P2 --half $H --seed "$SEED" --deals "$DEALS" --json "$OUT/$TAG.h$H.json" >"$OUT/$TAG.h$H.txt" 2>&1
  kill $A $B 2>/dev/null; wait $A $B 2>/dev/null
done
$PY judge_engine.py combine "$OUT/$TAG.h1.json" "$OUT/$TAG.h2.json" > "$OUT/$TAG.match.txt" 2>&1
echo "== $TAG"; cat "$OUT/$TAG.match.txt"
