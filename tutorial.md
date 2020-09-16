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

This project is initialized from one of the ideas Jon thought about for 2018
summer projects at USC/ISI, the Natural Language Group.
The initial goal is to build a automatic Dungeons and Dragons player but is
evolved into the simplified scenario that playing single person text-based games
such as Zork.

Over the years, the codebase becomes larger and larger with more than 10k line
of code and ~300k line of changes, supporting four game-playing papers, from
the simplest Zork DQN, to Cooking DRRN, Commonsense reasoning, and discriminative
state representation learning. There are also a lot of code are about game
action generation, but we decide not to move on for the action generation direction
at this time because of
lacking motivation --- who will accept a paper about generating languages to be
understandable by a game?

Beside game playing, we find out that there could be some practical usages for
the game-playing code, and there have already been some of them appeared in top
NLP venues. Though text-based game playing is still a niche domain, I'm optimistic
about the future. 

I hope the code can be user for

- beginners to learn how to write reinforcement learning code
- researchers to run comparison experiments, or to reach next step of game-playing
- domain experts that convert their problems into games to solve


## Install requirements
You may want to change the
tensorflow to tensorflow-gpu in the `requirements.txt` if GPUs are available.

```bash
cat requirements.txt | sed 's/tensorflow==1.15.3/tensorflow-gpu==1.15.3/'
pip install -r requirements.txt
```

## Add dependencies

To simplify the code, we assume the following packages are installed to
pre-defined paths.

`mkdir -p $HOME/local/opt/bert-models`

### Download BERT

`cd $HOME/local/opt/bert-models`

`wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip`

`unzip uncased_L-12_H-768_A-12.zip`

`ln -s uncased_L-12_H-768_A-12 bert-model`


### Download Albert (optional)

`cd $HOME/local/opt/bert-models`

`wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz`

`tar -zxvf albert_base_v2.tar.gz`

`ln -s albert_base albert-model`


## Important paths

- `$HOME`: user home dir

- `$PDIR`: the root path of the package

- `$DATA_DIR`: the dir that contains data

- `MODEL_HOME`: the dir that contains model

## Prepare datasets

We use three datasets -
SQuAD (Du split [\[Du et al., 2017\]](https://arxiv.org/abs/1705.00106)),
NewsQA, and Natural Questions (NQ).

### SQuAD - Du split
SQuAD can be found on Github: https://github.com/xinyadu/nqg, the data in
`nqg/data/processed` can be used directly. These data are pre-tokenized by NLTK
so if you use different tokenizer, e.g. BPE, then you may want to do another
pre-processing.

### NewsQA

NewsQA data reply on the CNN/Daily Mail (CNN/DM) dataset first, so download
the CNN/DM from the link: https://cs.nyu.edu/~kcho/DMQA/. You can only download
the CNN stories.

Then download the NewsQA dataset:
https://www.microsoft.com/en-us/research/project/newsqa-dataset/, login to
Microsoft account is required.
This will give you all the human generated questions and answers.

After having these two datasets, use the introduction in the Github to get a
combined dataset: https://github.com/Maluuba/newsqa. I recommend to use their
pre-built docker image to do the processing.

Finally you will get a combined dataset: `combined-newsqa-data-v1.json`.

Then use the `$PDIR/python/tools/newsqa_data_transformation.py`
in this package to do pre-process:

`Usage: newsqa_data_transformation.py FN_COMBINED_NEWSQA MAX_SRC_LEN
 FN_LA FN_SA FN_HIGHLIGHTS`

 The `MAX_SRC_LEN` is used for BPE tokenization.

E.g.

```bash
cd $PDIR
`./sbin/run.sh python/questgen/tools/newsqa_data_transformation.py combined-newsqa-data-v1.json 490 newsqa_la_size_490.txt newsqa_sa_size_490.txt newsqa_la_size_490.highlights.txt`
```

### NQ

Download the simplified train set and dev set from this url:
https://ai.google.com/research/NaturalQuestions/download

then pre-process the data with `nq_data_transformation.py`:

`Usage: nq_data_transformation.py FN_NQ FN_SRC FN_TGT IS_TESTING`

E.g.
```
cd $PDIR
./sbin/run.sh python/questgen/tools/nq_data_transformation.py v1.0-simplified-simplified-nq-train.jsonl src-total.txt tgt-total.txt False
```

We then split the data into train/dev set:

```bash
awk '{if(rand(seed)<0.9) {print > "src-train.txt"} else {print > "src-dev.txt"}}' src-total.txt
awk '{if(rand(seed)<0.9) {print > "tgt-train.txt"} else {print > "tgt-dev.txt"}}' tgt-total.txt
```


Notice that for `awk`, the random seed is fixed, so we can run the splits of
source and target separately.

Use the same method to extract test set from the original dev set:

E.g.
```bash
cd $PDIR
./sbin/run.sh python/questgen/tools/nq_data_transformation.py v1.0-simplified-simplified-nq-dev-all.jsonl src-test.txt tgt-test.txt True
```

Notice that there will be overlapped contexts between train/dev and the test
set, so we need a further filtering to remove overlapped contexts from
test sets with `filter_same.py`:

`Usage: filter_same.py FN_SRC_TOTAL FN_SRC_TEST`

E.g.
```bash
cd $PDIR
./sbin/run.sh python/questgen/tools/filter_same.py src-total.txt src-test.txt > test-mix-removals.txt
```

This will produce all line numbers in FN_SRC_TEST to be removed.

Then use `line_remover.py` to remove them:

`Usage: line_remover.py FN FN_REMOVAL NEW_FN`

E.g.
```bash
cd $PDIR
./sbin/run.sh python/questgen/tools/line_remover.py src-test.txt test-mix-removals.txt src-test.removal.txt
```


## Start training

run
```bash
cd $PDIR
nohup ./bin/run-train-nq-pgn.sh &> log.txt &
```

Three important paths in the script:

```bash
MODEL_HOME="$HOME/git-store/experiments/nq-pgn"
PRE_CONF_FILE="$PDIR/model_config/nq_pgn.yaml"
DATA_DIR="$HOME/data/NQ/processed"
```

models and hyper-parameters will be saved in `$MODEL_HOME`;

`$DATA_DIR` should be a dir contains `src-train.txt`, `src-dev.txt`, `src-test.txt`,
and `tgt-train.txt`, `tgt-dev.txt`. Soft links are allowed.

`$PRE_CONF_FILE` is a yaml file that contains some user-defined hyper-parameters,
templates are in `$PDIR/model_config`. Default hyper-parameters for each model
is defined in `hparams.py`.

example model config file:

```yaml
---
# use Transformer PGN to train SQuAD QG

model_creator: BertPGN  # only two in this package [BertPGN|TransformerPGN]
max_src_len: 500  # maximum allowed source context length
max_tgt_len: 50  # maximum allowed decoded question length
batch_size: 10
learning_rate: 5e-5
tokenizer_type: Bert  # three defined in this package [Bert|NLTK|Albert]
max_snapshot_to_keep: 20  # number of models saved
```

### Hyper-parameter priorities

This package have four methods to set hyper-parameters, and there are priorities.

1. set in hparams.py  (for programmers)

2. set in pre_config_file  (usually for training)

3. set in cmd args  (usually for inference)

4. set in `$MODEL_HOME/hparams.json`  (not recommend to set, only for hparams reading)

If there is no `$MODEL_HOME/hparams.json`, e.g. the first time training, then
3 > 2 > 1. (x > y means x overrides y)

If there is `$MODEL_HOME/hparams.json`, e.g. after training and during inference,
the priority is 4 > 2 > 1. Some hyper-parameters in 3 will override 4, here are
the hyper-parameters:

- model_dir
- batch_size
- learning_rate
- max_snapshot_to_keep
- beam_size
- temperature
- use_greedy

The behavior is defined in `hparams.py`.

Which hyper-parameter is actually working? When you run the code, the hyper-parameters
will output to stderr, e.g.

```
hparams.json exists! some hyper-parameter setting from CMD and pre-config file will be disabled. make sure to clear model_dir first if you want to train a new agent from scratch!
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

## Start dev evaluation

The dev evaluation will run through each model saved in `$MODEL_HOME/last_weights`
and save better models to `$MODEL_HOME/best_weights`.

run
```bash
nohup ./bin/run-dev-eval-nq-pgn.sh &> log-dev.txt &
```

## Start inference

Inference process will read contexts from `src-test.txt` and generate questions.

run
```bash
nohup ./bin/run-infer-nq-pgn.sh &> log-test.txt &
```

Important cmd args for inference:

`--disable-greedy --temperature 0.3 --beam-size 3` - use nucleus sampling during
decoding, temperature set to 0.3, beam size 3.

`--use-greedy --temperature 0.3 --beam-size 3` - use beam search with greedy,
beam size 3, temperature won't affect the decoding process.


## Prepare generated questions for QA system evaluation and human evaluation

The SQuAD QA system we used: BERT: https://github.com/google-research/bert

The NQ QA system we used: BERT-joint: https://github.com/google-research/language/tree/master/language/question_answering/bert_joint

`newsqa_to_nq.py`: transform generated questions in NQ style for NQ QA system

`newsqa_to_squad.py`: transform generated questions in SQuAD style for SQuAD QA system

`csv_for_turk.py`: transform generated questions for Amazon MTurk HIT csv data

`show_diff_qa_nq.py`: read NQ QA system output

`show_diff_qa_squad.py`: read SQuAD QA system outpu