#!/bin/bash

#SBATCH --gres=gpu:k80:2
#SBATCH --ntasks=4
#SBATCH --time=100:00:00
#SBATCH --partition=isi
#SBATCH --mail-user=xusenyin@isi.edu
#SBATCH --mail-type=ALL

set -e -x

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOBNAME=$SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"
echo "working directory=$SLURM_SUBMIT_DIR"


if [[ $(hostname) =~ "hpc" ]]; then
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
    # shellcheck source=/dev/null
    source "$HOME/local/etc/init_tensorflow.sh"
fi


if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit
fi


MODELHOME=$1
GAMEPATH=$2
FGAMES=$3


pushd "$PDIR"
./sbin/run.sh python/deepword/main.py \
    --model-dir "$MODELHOME" \
    "gen-data" \
    --game-path "$GAMEPATH" --f-games "$FGAMES" \
    --load-best \
    --epoch-size 50000 \
    --epoch-limit 1
popd
