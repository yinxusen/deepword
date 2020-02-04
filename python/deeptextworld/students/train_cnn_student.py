import os
from os.path import join as pjoin

import fire
import tensorflow as tf

from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.students.evaluation import WatchDogEvalPlayer
from deeptextworld.students.student_learner import DRRNLearner, CMD
from deeptextworld.students.utils import setup_train_log, setup_eval_log, \
    load_and_split

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class TrainEval(object):
    @staticmethod
    def train(data_path, n_data, model_path):
        cmd_args.set("model_dir", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        setup_train_log(model_path)

        hp = load_hparams_for_training(None, cmd_args)
        learner = DRRNLearner(hp, model_path, data_path, n_data)
        learner.train(n_epochs=1000)

    @staticmethod
    def dev_eval(model_path, game_path, f_games, n_gpus=1):
        cmd_args.set("model_dir", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        setup_eval_log(log_filename="/tmp/eval-logging.txt")

        _, eval_games = load_and_split(game_path, f_games)
        eval_player = WatchDogEvalPlayer()
        eval_player.start(
            cmd_args, model_path, eval_games, n_gpus)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.expanduser("~")
    project_path = pjoin(dir_path, "../../..")
    bert_ckpt_dir = pjoin(home_dir, "local/opt/bert-models/bert-model")
    bert_vocab_file = pjoin(bert_ckpt_dir, "vocab.txt")
    nltk_vocab_file = pjoin(project_path, "resources/vocab.txt")

    cmd_args = CMD(
        model_dir="",
        model_creator="CNNEncoderDSQN",
        vocab_file=bert_vocab_file,
        bert_ckpt_dir=bert_ckpt_dir,
        num_tokens=511,
        num_turns=6,
        batch_size=32,
        save_gap_t=5000,
        embedding_size=64,
        learning_rate=5e-5,
        num_conv_filters=32,
        bert_num_hidden_layers=1,
        cls_val="[CLS]",
        cls_val_id=0,
        sep_val="[SEP]",
        sep_val_id=0,
        mask_val="[MASK]",
        mask_val_id=0,
        tokenizer_type="BERT",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True
    )

    fire.Fire(TrainEval)
