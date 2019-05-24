#!/bin/bash

#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=4
#SBATCH --time=100:00:00
#SBATCH --partition=isi
#SBATCH --mail-user=xusenyin@isi.edu
#SBATCH --mail-type=ALL

set -e -x

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOBNAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR


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

MODELHOME="$PDIR/../experiments-drrn/agent-drrn-${filename}"
BERT_CKPT_DIR="$HOME/local/opt/bert-models/bert-model"
VOCAB_FILE="$BERT_CKPT_DIR/vocab.txt"
GAME_DIR="$PDIR/../textworld-competition-games/train"
F_GAMES="$PDIR/../textworld-competition-games/train-tier1-recipe1.games.txt-diff"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deeptextworld/main.py \
    -m $MODELHOME --mode train-drrn \
    --game-path $GAME_DIR --f-games $F_GAMES --vocab-file $VOCAB_FILE \
    --annealing-eps-t 2000000 --annealing-gamma-t 2000 --observation-t 10000 --replay-mem 500000 \
    --eval-episode 2 --save-gap-t 10000 --batch-size 32 --game-episode-terminal-t 100 \
    --model-creator BertCNNEncoderDRRN --bert-ckpt-dir $BERT_CKPT_DIR \
    --start-t-ignore-model-t
popd
