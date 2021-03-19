#!/bin/bash

#SBATCH --gres=gpu:k80:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=300:00:00
#SBATCH --partition=isi

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

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

MODELHOME="$PDIR/../experiments-drrn/agent-${filename}"
PRE_CONF_FILE="$PDIR/model_config/drrn-zork-cnn.yaml"
GAME_PATH="$PDIR/../zplayer/resources/games/zork1.z5"
ACTION_FILE="$PDIR/resources/miscellany/commands-zork1-minimum.txt"


pushd "$PDIR"
./sbin/run.sh python/deepword/main.py \
    --config-file "$PRE_CONF_FILE" \
    --model-dir "$MODELHOME" \
    --action-file "$ACTION_FILE" \
    "train-dqn" \
    --game-path "$GAME_PATH"
popd
