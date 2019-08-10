#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=200:00:00
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
    pyenv activate deepdnd-drrn-cpu
else
    FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    PDIR="$FWDIR/.."
    filename=$(basename "$0")
    extension="${filename##*.}"
    filename="${filename%.*}"
fi

PDIR="."

MODELHOME="$PDIR/../experiments-drrn/agent-snn-test"

VOCAB_FILE="$PDIR/resources/vocab.txt"
GAMEPATH=${1:-"$PDIR/../textworld-competition-games/train"}

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deeptextworld/main.py \
    -m $MODELHOME --mode train-snn \
    --game-path $GAMEPATH \
    --vocab-file $VOCAB_FILE \
    --annealing-eps-t 30000 --annealing-gamma-t 1000 --observation-t 500 --replay-mem 1000 \
    --eval-episode 1 --embedding-size 64 \
    --save-gap-t 1000 --batch-size 32 --game-episode-terminal-t 100 \
    --model-creator CNNEncoderSNN
popd
