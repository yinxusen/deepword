#!/bin/bash

#SBATCH --gres=gpu:k80:2
#SBATCH --ntasks=4
#SBATCH --time=48:00:00
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

export PYTHONPATH="$HOME/git-store/zplayer/python/:$PYTHONPATH"

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

DATAHOME="$PDIR/../zplayer/resources/games/zork1.z5"
MODELHOME=$1

ACTION_FILE="$PDIR/resources/commands-zork1-egg-take.txt"
VOCAB_FILE="$PDIR/resources/vocab.50K.en.trimed"
TGT_VOCAB_FILE="$PDIR/resources/vocab.mini-zork.txt"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

./bin/run.sh python/deepdnd/main.py \
    -d $DATAHOME -m $MODELHOME \
    --mode human --eval_episode 1 \
    --game_episode_terminal_t 100
