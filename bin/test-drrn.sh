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
    --game_dir /Users/xusenyin/git-store/textworld-competition-games/train-1 \
    --vocab_file $VOCAB_FILE \
    --annealing_eps_t 300 --annealing_gamma_t 10 --observation_t 50 --replay_mem 100 \
    --eval_episode 1 --embedding_size 64 \
    --save_gap_t 50 --batch_size 32 --game_episode_terminal_t 20 --model_creator CNNEncoderDRRN
popd
