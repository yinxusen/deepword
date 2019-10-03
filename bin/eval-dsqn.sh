#!/bin/bash

set -e -x

FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PDIR="$FWDIR/.."
filename=$(basename "$0")
extension="${filename##*.}"
filename="${filename%.*}"

DATAHOME=$1
F_GAMES=$2
MODELHOME=$3

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

./bin/run.sh python/deeptextworld/main.py \
    --game-path $DATAHOME -m $MODELHOME --f-games $F_GAMES \
    --mode eval-dsqn --eval-episode 10 --eval-randomness 0 --eval-mode all \
    --game-episode-terminal-t 100
