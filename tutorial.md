```text
██████╗ ███████╗███████╗██████╗ ██╗    ██╗ ██████╗ ██████╗ ██████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║    ██║██╔═══██╗██╔══██╗██╔══██╗
██║  ██║█████╗  █████╗  ██████╔╝██║ █╗ ██║██║   ██║██████╔╝██║  ██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║███╗██║██║   ██║██╔══██╗██║  ██║
██████╔╝███████╗███████╗██║     ╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝
╚═════╝ ╚══════╝╚══════╝╚═╝      ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝ 
```

# Tutorial

DeepWord is a project of building automatic agents to play text-based games.

## Overview

The architecture of the DeepWord

<img src="resources/deepword-archi.png" width="400">

- Agent: Interacting with the TextWorld, receiving text descriptions or reactions,
scores, and sending the next action to the game. The game could be anything
only if it sends over texts and scores and receives the next action.

- Core: Interacting with the Agent to receive training samples or requests of computing
policies, and forwarding these to the Model.

- Model: Mostly a Tensorflow model (v1.x).

These are three essential parts of DQN.


## Install requirements
Python3 is required for the package. I recommend to use 
[pyenv](https://github.com/pyenv/pyenv) and 
[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).
You can use your own Python environment such as Conda, or the original python virtualenv.
You may need to change `python` and `pip` to `python3` and `pip3` according to your Python environment.

You may want to change the
`tensorflow` to `tensorflow-gpu` in the `requirements.txt` if GPUs are available.

```bash
pip install -r requirements.txt
```

## Important paths

- `$HOME`: user home dir
- `$PDIR`: the root path of the package
- `$DATA_DIR`: the dir that contains data
- `MODEL_HOME`: the dir that contains model
  - `last_weights/`: the dir contains latest models
  - `best_weights/`: the dir contains best models (evaluated by the dev set)
  - `game_script.log`: training log
  - `hparams.json`: hyper-parameters
  - `actions-(steps).npz`: snapshot of actions for games
  - `floor_plan-(steps).npz`: snapshot of floor plan collected
  - `trajectories-(steps).npz`: snapshot of trajectories collected
  - `state_text-(steps).npz`: snapshot of state texts (a combination of observation and inventory)
  - `hs2tj-(steps).npz`: snapshot of mapping from states to trajectories
  - `memo-(steps).npz`: snapshot of the replay memory

## Add dependencies

To simplify the code, we assume the following packages are installed to
pre-defined paths.

`mkdir -p $HOME/local/opt/bert-models`

### Download BERT

```bash
cd $HOME/local/opt/bert-models
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
ln -s uncased_L-12_H-768_A-12 bert-model
```


### Download Albert (optional)

```bash
cd $HOME/local/opt/bert-models
wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz
tar -zxvf albert_base_v2.tar.gz
ln -s albert_base albert-model
```

## hyper-parameters and configuration

We have four methods to set hyperparameters, and they have different priorities
when there are conflicts among them.

- `hparams.json` in `model_home`
- `$PRE_CONF_FILE` (YAML file, examples shown in `$PDIR/mode_config`)
- pre-set values in `deepword.hparams`
- command line args

### full hyperparameters in `deepword.hparams`

Different models have different hyperparameters. For a full list,
see `deepword.hparrams.default_config`

### PRE_CONF_FILE

`$PRE_CONF_FILE` is a YAML file that contains some user-defined hyper-parameters,
usually used as templates for the same set of training.

Example model config file:

```yaml
---
model_creator: CnnDSQN  # use CnnDSQN model
num_tokens: 500  # trajectory max length
num_turns: 5  # action-master turns to construct trajectory
batch_size: 32  # batch size for training
save_gap_t: 10000  # training steps for model saving
embedding_size: 64  # word embedding size
learning_rate: 5e-5  # learning rate
num_conv_filters: 32  # number of CNN convolutional filters if using CNN
tokenizer_type: BERT  # tokenizer type, e.g. BERT, NLTK, Albert
max_snapshot_to_keep: 3  # number of snapshots to keep before deleting
eval_episode: 2  # number of episode for each game during evaluation
game_episode_terminal_t: 100  # number of steps for each game-playing
replay_mem: 100000  # the experience replay memory size
collect_floor_plan: True  # collect floor plan during playing
annealing_eps_t: 2000000  # number of steps to anneal eps from MAX_eps to MIN_eps
observation_t: 10000  # number of observation before training
init_eps: 1.0  # MAX_eps
start_t_ignore_model_t: False  # game-playing steps different with model training steps
use_step_wise_reward: True  # use step-related reward schema
agent_clazz: DSQNCompetitionAgent  # use DSQN Competition agent
core_clazz: DSQNCore  # use DSQN Core
policy_to_action: LinUCB  # use LinUCB method when choose action
```

### command line args (CMD args)

CMD args are pretty useful when you want to do some small tweaks for a PER_CONF_FILE.
E.g., you want to train a new model with a different learning rate, but all other 
hyperparameters are unchanged, Or when you want to do the evaluation, but with a
different `policy_to_action` method, e.g. changing from `LinUCB` to `sampling`.

You can see the full allowed CMD args in `deepword.main.hp_parser`.

### `hparams.json` in `$MODEL_HOME`

`hparams.json` is the hyperparameter config file that goes along with the model.
This file is saved during the training process automatically.
The next time training or evaluation will read hyperparameters in this file.

### Hyper-parameter priorities

This package has four methods to set hyper-parameters, and there are priorities.

1. set in hparams.py  (for programmers)

2. set in pre_config_file  (usually for training)

3. set in cmd args  (usually for inference)

4. set in `$MODEL_HOME/hparams.json`  (not recommended to use, only for hparams reading)

If there is no `$MODEL_HOME/hparams.json`, e.g., the first time training, then
3 > 2 > 1. (x > y means x overrides y)

If there is `$MODEL_HOME/hparams.json`, e.g., the second time training and during
inference,
the priority is 4 > 2 > 1. Some hyper-parameters in 3 will override 4, depend
on the logic and usage for that parameter. These hyperparameters are defined in
`deepword.hparams.load_hparams.allowed_to_change`.

### Which hyperparameter is actually working?

Which hyper-parameter is actually working? When you run the code, the hyper-parameters
will output to stderr, e.g.

```
hparams.json exists! Some hyper-parameter set from CMD and the pre-config file will be disabled. Make sure to clear model_dir first if you want to train a new agent from scratch!
------------hparams---------------
batch_size -> 10
beam_size -> 3
cls_val -> [CLS]
cls_val_id -> 101
embedding_size -> 64
eos -> [unused1]
eos_id -> 2
glove_trainable -> False
learner_clazz ->
learning_rate -> 5e-05
mask_val -> [MASK]
mask_val_id -> 103
max_snapshot_to_keep -> 30
max_src_len -> 500
max_tgt_len -> 50
model_creator -> BertPGN
model_dir -> /home/ubuntu/git-store/experiments/nq-pgn
padding_val -> [PAD]
padding_val_id -> 0
sep_val -> [SEP]
sep_val_id -> 102
sos -> [unused0]
sos_id -> 1
temperature -> 0.5
tokenizer_type -> Bert
unk_val -> [UNK]
unk_val_id -> 100
use_glove_emb -> False
use_greedy -> True
vocab_size -> 30522
-----------------------------------
```

## Basic Usage

### A simple Zork DQN training

Zork is not a TextWorld generated game, even though you can still run it with TextWorld.
Instead of the game file `Zork1.z5`, you still need an action file containing
all possible actions, e.g., `commands-zork1-130.txt`.

```bash
cd $PDIR
wget https://github.com/yinxusen/dqn-zork/blob/master/resources/games/zork1.z5 ./zork1.z5
wget https://github.com/yinxusen/dqn-zork/blob/master/resources/commands-zork1-minimum.txt ./commands-zork1-minimum.txt

MODEL_HOME="example-model"
PRE_CONF_FILE="model_config/dqn-zork-cnn.yaml"
ACTION_FILE="commands-zork1-minimum.txt"
GAME_PATH="zork1.z5"

./sbin/run.sh python/deepword/main.py \
    --config-file "$PRE_CONF_FILE" \
    --model-dir "$MODELHOME" \
    --action-file "$ACTION_FILE" \
    "train-dqn" \
    --game-path "$GAME_PATH"
```
According to the `$PRE_CONF_FILE`, this will run 2 million training steps.
You can decrease it by using CMD arg, e.g., `--annealing-eps-t 10000`.
After the training, you can test your model by evaluation:

```bash
./sbin/run.sh python/deepword/main.py \
    --model-dir "$MODEL_HOME" \
    eval-dqn \
    --game-path "$GAME_PATH"
popd
```

You can also switch to the DRRN model to train Zork. You need to create a new model
dir since the DRRN model is different from the DQN model.

```bash
MODEL_HOME="example-model-drrn"

./sbin/run.sh python/deepword/main.py \
    --config-file "$PRE_CONF_FILE" \
    --model-creator "CnnDRRN" \
    --model-dir "$MODELHOME" \
    --action-file "$ACTION_FILE" \
    "train-dqn" \
    --game-path "$GAME_PATH"
```

### Teacher-Student Training

Refer to the [Learning to Generalize for Sequential Decision Making](https://github.com/yinxusen/learning_to_generalize)
code repository.
