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

# DATAHOME="$PDIR/../textworld-competition/starting_kit/sample_games/tw-cooking-recipe1+take1-11Oeig8bSVdGSp78.ulx"
DATAHOME="$PDIR/../textworld-competition/starting_kit/sample_games/tw-cooking-recipe2+take2+cut+open-BnYEixa9iJKmFZxO.ulx"
MODELHOME="$PDIR/../experiments/agent-drrn-test"

ACTION_FILE=""
VOCAB_FILE="$PDIR/resources/vocab.txt"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deepdnd/main.py \
    -d $DATAHOME -m $MODELHOME --mode train-drrn \
    --game_dir /Users/xusenyin/git-store/textworld-competition/starting_kit/sample_games \
    --vocab_file $VOCAB_FILE \
    --annealing_eps_t 100000 --annealing_gamma_t 10 --observation_t 500 --replay_mem 100000 \
    --eval_episode 5 --embedding_size 64 \
    --save_gap_t 100 --batch_size 32 --game_episode_terminal_t 300 --model_creator CNNEncoderDRRN
popd
