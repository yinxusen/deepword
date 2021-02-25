#!/bin/bash

#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=10:00:00
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

MODELHOME="$PDIR/../experiments-drrn/agent-student-train-student-model-bert-commonsense"
GAMEPATH="$HOME/git-store/zplayer/resources/games/zork1.z5"
ACTIONFILE="$HOME/git-store/deep-dnd/resources/commands-zork1-minimum.txt"

pushd "$PDIR"
./sbin/run.sh python/deepword/main.py \
    --agent-clazz "ZorkAgent" \
    --eval-episode 1 \
    --model-dir "$MODELHOME" \
    --action-file "$ACTIONFILE" \
    --disable-collect-floor-plan \
    --game-episode-terminal-t 600 \
    --policy-to-action eps --policy-eps 0 \
    "eval-dqn" \
    --eval-mode "eval" --game-path "$GAMEPATH" --debug
popd
