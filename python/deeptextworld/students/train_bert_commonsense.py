import os

import fire
import tensorflow as tf

from deeptextworld.hparams import conventions
from deeptextworld.students.student_learner import BertLearner, CMD
from deeptextworld.students.train_eval_framework import TrainEval

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


if __name__ == '__main__':
    cmd_args = CMD(
        model_dir="",
        model_creator="BertCommonsenseModel",
        vocab_file=conventions.bert_vocab_file,
        bert_ckpt_dir=conventions.bert_ckpt_dir,
        num_tokens=300,
        num_turns=6,
        batch_size=10,
        save_gap_t=5000,
        embedding_size=64,
        learning_rate=5e-5,
        bert_num_hidden_layers=12,
        tokenizer_type="BERT",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True
    )
    train_eval = TrainEval(cmd_args, BertLearner)
    fire.Fire(train_eval)
