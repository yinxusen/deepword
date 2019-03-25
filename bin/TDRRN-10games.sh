#!/bin/bash

#SBATCH --gres=gpu:k80:2
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
VOCAB_FILE="$PDIR/resources/vocab.txt"
GAME_DIR="$PDIR/../textworld-competition-games/train"

if [[ -f $HOME/local/etc/init_tensorflow.sh ]]; then
    source $HOME/local/etc/init_tensorflow.sh
fi

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deeptextworld/main.py \
    -m $MODELHOME --mode train-drrn --game_dir $GAME_DIR --vocab_file $VOCAB_FILE \
    --annealing_eps_t 1000000 --annealing_gamma_t 2000 --observation_t 5000 --replay_mem 500000 \
    --embedding_size 64 --save_gap_t 5000 --batch_size 32 \
    --game_episode_terminal_t 200 --eval_episode 10 --model_creator CNNEncoderDRRN \
    --max_games_used 10
popd
