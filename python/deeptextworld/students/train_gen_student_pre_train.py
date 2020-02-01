import os
from os.path import join as pjoin

import fire
import tensorflow as tf

from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.students.student_learner import GenPreTrainLearner, CMD
from deeptextworld.students.utils import setup_train_log

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


def train(data_path, n_data, model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    setup_train_log(model_path)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.expanduser("~")
    project_path = pjoin(dir_path, "../../..")
    bert_ckpt_dir = pjoin(home_dir, "local/opt/bert-models/bert-model")
    bert_vocab_file = pjoin(bert_ckpt_dir, "vocab.txt")
    nltk_vocab_file = pjoin(project_path, "resources/vocab.txt")

    cmd_args = CMD(
        model_dir=model_path,
        model_creator="AttnEncoderDecoderDQN",
        vocab_file=nltk_vocab_file,
        num_tokens=512,
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

    hp = load_hparams_for_training(None, cmd_args)
    learner = GenPreTrainLearner(hp, model_path, data_path, n_data)
    learner.train(n_epochs=1000)


if __name__ == "__main__":
    fire.Fire(train)
