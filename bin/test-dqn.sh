#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --time=60:00:00
#SBATCH --partition=isi

set -e -x


if [[ `hostname` =~ "hpc" ]]; then
    PDIR=""
else
    FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    PDIR="$FWDIR/.."
fi

MODELHOME="$PDIR/../experiments-drrn/agent-dqn-test"

VOCAB_FILE="$PDIR/resources/vocab.txt"
GAMEPATH=$1

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deeptextworld/main.py \
    -m $MODELHOME --mode train-dqn \
    --game-path $GAMEPATH \
    --vocab-file $VOCAB_FILE \
    --annealing-eps-t 30000 --annealing-gamma-t 1000 --observation-t 500 --replay-mem 1000 \
    --eval-episode 1 --embedding-size 64 \
    --save-gap-t 1000 --batch-size 32 --game-episode-terminal-t 100 \
    --model-creator CNNEncoderDQN
popd
