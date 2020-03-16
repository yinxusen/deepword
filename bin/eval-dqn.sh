#!/bin/bash

set -e -x

FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PDIR="$FWDIR/.."
filename=$(basename "$0")
extension="${filename##*.}"
filename="${filename%.*}"

DATAHOME=$1
MODELHOME=$2

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

./bin/run.sh python/deeptextworld/dqn_train.py \
    --game-path $DATAHOME -m $MODELHOME \
    --mode eval-dqn --eval-episode 1 --eval-randomness 0 --eval-mode eval-train \
    --game-episode-terminal-t 100
