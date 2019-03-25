#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --time=60:00:00
#SBATCH --partition=isi

set -e -x

export PYTHONPATH="$HOME/git-store/zplayer/python/:$PYTHONPATH"

if [[ `hostname` =~ "hpc" ]]; then
    PDIR=""
else
    FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    PDIR="$FWDIR/.."
fi

DATAHOME="$PDIR/../zplayer/resources/games/zork1.z5"
MODELHOME="$PDIR/../experiments/agent-zork-test"

ACTION_FILE="$PDIR/resources/commands-zork1-egg-take.txt"
VOCAB_FILE="$PDIR/resources/vocab.50K.en.trimed"
TGT_VOCAB_FILE="$PDIR/resources/vocab.mini-zork.txt"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

./bin/run.sh python/deepdnd/main.py \
    -d $DATAHOME -m $MODELHOME \
    --action_file $ACTION_FILE --vocab_file $VOCAB_FILE --tgt_vocab_file $TGT_VOCAB_FILE \
    --annealing_eps_t 100 --annealing_gamma_t 10 --observation_t 50 --replay_mem 100 \
    --eval_episode 5 --embedding_size 64 \
    --save_gap_t 10 --batch_size 32 --game_episode_terminal_t 20 --eval_episode 2 --model_creator CNNEncoderDQN --delay_target_network 1
