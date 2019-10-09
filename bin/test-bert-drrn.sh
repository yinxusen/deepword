#!/bin/bash

#SBATCH --gres=gpu:p100:2
#SBATCH --time=60:00:00
#SBATCH --partition=isi

set -e -x

if [[ `hostname` =~ "hpc" ]]; then
    PDIR="$SLURM_SUBMIT_DIR"
    filename="$SLURM_JOB_NAME"
    extension="${filename##*.}"
    filename="${filename%.*}"
    export PYENV_ROOT="$HOME/local/lib/pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv 1>/dev/null 2>&1; then
      eval "$(pyenv init -)"
    fi
    eval "$(pyenv virtualenv-init -)"
    pyenv activate deepdnd-drrn
else
    FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    PDIR="$FWDIR/.."
    filename=$(basename "$0")
    extension="${filename##*.}"
    filename="${filename%.*}"
fi

MODELHOME="$PDIR/../experiments-drrn/agent-drrn-test"
BERT_CKPT_DIR="$HOME/local/opt/bert-models/bert-model"
VOCAB_FILE="$BERT_CKPT_DIR/vocab.txt"
GAME_DIR="$PDIR/../textworld-competition-games/train-1"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deeptextworld/main.py \
    -m $MODELHOME --mode train-drrn \
    --game-path $GAME_DIR \
    --vocab-file $VOCAB_FILE \
    --annealing-eps-t 300 --annealing-gamma-t 10 --observation-t 50 --replay-mem 100 \
    --eval-episode 1 \
    --save-gap-t 50 --batch-size 32 --game-episode-terminal-t 20 \
    --model-creator BertEncoderDRRN \
    --bert-ckpt-dir $BERT_CKPT_DIR --bert-num-hidden-layers 1
popd
