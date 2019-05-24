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

./bin/run.sh python/deeptextworld/main.py \
    --game-path $DATAHOME -m $MODELHOME \
    --mode human-check --eval-episode 1 --eval-randomness 0.05 --eval-mode all \
    --game-episode-terminal-t 100
