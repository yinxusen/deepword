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

MODELHOME="$PDIR/../experiments-drrn/agent-drrn-test"

VOCAB_FILE="$PDIR/resources/vocab.txt"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deeptextworld/main.py \
    -m $MODELHOME --mode train-drrn \
    --game-path /Users/xusenyin/git-store/textworld-competition-games/train-1 \
    --vocab-file $VOCAB_FILE \
    --annealing-eps-t 300 --annealing-gamma-t 10 --observation-t 50 --replay-mem 100 \
    --eval-episode 1 --embedding-size 64 \
    --save-gap-t 50 --batch-size 32 --game-episode-terminal-t 20 \
    --model-creator AttnEncoderDRRN
popd
