#!/bin/bash
# usage: final_queue.sh <ckpt.npz> <tag> [workers] [scale]   (beast, cores 30,31)
CK=$1; TAG=$2; W=${3:-2}; export SCALE=${4:-1.0}
cd ~/poker-arena/gpu/drl/code
E=offline/drl/arena_eval.sh
$E $CK ${TAG}_bp $W bp 101 202 303 404 505 606 707 808 909 1010 1111 1212 1313 1414 1515 1616
$E $CK ${TAG}_mix $W mix 101 202 303 404 505 606 707 808
$E $CK ${TAG}_bal $W bp1,heur,rel 101 202 303 404
$E $CK ${TAG}_simple $W station,nit,tag,maniac,barrel 101 202 303 404
