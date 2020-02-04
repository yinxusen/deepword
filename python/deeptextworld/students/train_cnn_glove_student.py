import os

import fire
import tensorflow as tf

from deeptextworld.students.student_learner import CMD, DRRNLearner
from deeptextworld.students.train_eval_framework import TrainEval, conventions

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


if __name__ == "__main__":
    cmd_args = CMD(
        model_dir="",
        model_creator="CNNEncoderDSQN",
        vocab_file=conventions.glove_vocab_file,
        bert_ckpt_dir=conventions.bert_ckpt_dir,
        num_tokens=256,
        num_turns=6,
        batch_size=32,
        save_gap_t=5000,
        embedding_size=50,
        learning_rate=5e-5,
        num_conv_filters=32,
        bert_num_hidden_layers=1,
        cls_val="[CLS]",
        cls_val_id=0,
        sep_val="[SEP]",
        sep_val_id=0,
        mask_val="[MASK]",
        mask_val_id=0,
        tokenizer_type="NLTK",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True,
        use_glove_emb=True,
        glove_emb_path=conventions.glove_emb_file,
        glove_trainable=False
    )
    train_eval = TrainEval(cmd_args, DRRNLearner)
    fire.Fire(train_eval)
