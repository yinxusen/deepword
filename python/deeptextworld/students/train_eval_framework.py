import os
from collections import namedtuple
from os.path import join as pjoin

from deeptextworld.hparams import load_hparams_for_training, \
    load_hparams_for_evaluation
from deeptextworld.students.evaluation import LoopDogEvalPlayer, \
    MultiGPUsEvalPlayer, FullDirEvalPlayer
from deeptextworld.students.utils import setup_train_log, setup_eval_log, \
    load_and_split, load_game_files


class Conventions(namedtuple(
        "Conventions",
        ("bert_ckpt_dir", "bert_vocab_file", "nltk_vocab_file",
         "glove_vocab_file", "glove_emb_file"))):
    pass


dir_path = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")
project_path = pjoin(dir_path, "../../..")
conventions = Conventions(
    bert_ckpt_dir=pjoin(home_dir, "local/opt/bert-models/bert-model"),
    bert_vocab_file=pjoin(
        home_dir, "local/opt/bert-models/bert-model/vocab.txt"),
    nltk_vocab_file=pjoin(project_path, "resources/vocab.txt"),
    glove_vocab_file=pjoin(
        home_dir, "local/opt/glove-models/glove.6B/vocab.glove.6B.4more.txt"),
    glove_emb_file=pjoin(
        home_dir, "local/opt/glove-models/glove.6B/glove.6B.50d.4more.txt")
)


class TrainEval(object):
    def __init__(self, cmd_args, learner_clazz):
        self.cmd_args = cmd_args
        self.learner_clazz = learner_clazz

    def train(self, data_path, n_data, model_path):
        """
        Train an agent with supervised dataset
        :param data_path:
        :param n_data:
        :param model_path:
        :return:
        """
        self.cmd_args.set("model_dir", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        setup_train_log(model_path)

        hp = load_hparams_for_training(None, self.cmd_args)
        learner = self.learner_clazz(hp, model_path, data_path, n_data)
        learner.train(n_epochs=1000)

    def dev_eval(self, model_path, game_path, f_games, n_gpus=1):
        """
        Use dev set to evaluate agent along with the training process
        A dev set evaluator will run when a new model saved
        :param model_path:
        :param game_path:
        :param f_games:
        :param n_gpus:
        :return:
        """
        self.cmd_args.set("model_dir", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        setup_eval_log(log_filename="/tmp/eval-logging.txt")

        _, eval_games = load_and_split(game_path, f_games)
        eval_player = LoopDogEvalPlayer()
        eval_player.start(
            self.cmd_args, model_path, eval_games, n_gpus)

    def eval(self, model_path, game_path, f_games, n_gpus=1):
        """
        Evaluate with a test set once
        :param model_path:
        :param game_path:
        :param f_games:
        :param n_gpus:
        :return:
        """
        self.cmd_args.set("model_dir", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        setup_eval_log(log_filename="/tmp/eval-logging.txt")
        config_file = pjoin(model_path, "hparams.json")
        hp = load_hparams_for_evaluation(config_file, self.cmd_args)
        game_files = load_game_files(game_path, f_games)
        eval_player = MultiGPUsEvalPlayer(
            hp, model_path, game_files, n_gpus, load_best=False)
        eval_player.evaluate(restore_from=None)

    def full_eval(self, model_path, game_path, f_games, n_gpus=1):
        """
        Evaluate all saved models in a directory with a test set
        :param model_path:
        :param game_path:
        :param f_games:
        :param n_gpus:
        :return:
        """
        self.cmd_args.set("model_dir", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        setup_eval_log(log_filename="/tmp/eval-logging.txt")

        _, eval_games = load_and_split(game_path, f_games)
        eval_player = FullDirEvalPlayer()
        eval_player.start(
            self.cmd_args, model_path, eval_games, n_gpus)
