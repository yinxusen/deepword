import os

import fire
import tensorflow as tf

from deeptextworld.students.student_learner import GenLearner, CMD
from deeptextworld.students.train_eval_framework import TrainEval, conventions

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


if __name__ == "__main__":
    cmd_args = CMD(
        model_dir="",
        model_creator="TransformerGenDQN",
        vocab_file=conventions.nltk_vocab_file,
        num_tokens=256,
        num_turns=6,
        batch_size=32,
        save_gap_t=50000,
        embedding_size=64,
        learning_rate=5e-5,
        tokenizer_type="NLTK",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True
    )
    train_eval = TrainEval(cmd_args, GenLearner)
    fire.Fire(train_eval)
