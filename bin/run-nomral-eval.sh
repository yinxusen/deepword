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


if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    exit
fi

MODEL_DIR=$1

pushd "$PDIR"
./sbin/run.sh python/deepword/main.py \
    --model-dir $MODEL_DIR \
    --eval-episode 1 \
    --policy-to-action eps \
    --policy-eps 0 \
    --always-compute-policy \
    --agent-clazz BaseAgent \
    --append-objective-to-tj \
    eval-dqn \
    --eval-mode "eval" \
    --game-path ~/git-store/textworld-competition-games/tw_games-train/ \
    --f-games ~/git-store/textworld-competition-games/train-th-single-game.txt
popd
