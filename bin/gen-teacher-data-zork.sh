#!/bin/bash

#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=30:00:00
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


MODELHOME="$PDIR/../experiments-drrn/agent-zork-ZAllCNN-max-pool-pos-emb"
ACTION_FILE="$MODELHOME/deps/commands-zork1-minimum.txt"
GAME_PATH="$PDIR/../zplayer/resources/games/zork1.z5"


pushd "$PDIR"
./sbin/run.sh python/deepword/main.py \
    --model-dir "$MODELHOME" \
    --game-episode-terminal-t 600 \
    --disable-collect-floor-plan \
    --action-file "$ACTION_FILE" \
    --agent-clazz "DSQNZorkAgent" \
    --policy-to-action "LinUCB" \
    "gen-data" \
    --game-path "$GAME_PATH" --epoch-limit 2 --epoch-size 100000 --max-randomness 0.5
popd
