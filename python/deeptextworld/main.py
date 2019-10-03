import argparse
import os
import sys
import time

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deeptextworld.hparams import load_hparams_for_evaluation
from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.utils import setup_logging


parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument('-m', '--model-dir', type=str)
parser.add_argument('-d', '--data-dir', type=str)
parser.add_argument('-c', '--config-file', type=str)
parser.add_argument('--game-path', type=str, help='[a dir|a game file]')
parser.add_argument('--f-games', type=str)
parser.add_argument('--vocab-file', type=str)
parser.add_argument('--tgt-vocab-file', type=str)
parser.add_argument('--action-file', type=str)
parser.add_argument('--mode', default='training', help='[training|evaluation]')
parser.add_argument('--eval-episode', type=int)
parser.add_argument('--init-eps', type=float)
parser.add_argument('--final-eps', type=float)
parser.add_argument('--annealing-eps-t', type=int)
parser.add_argument('--init-gamma', type=float)
parser.add_argument('--final-gamma', type=float)
parser.add_argument('--annealing-gamma-t', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--save-gap-t', type=int)
parser.add_argument('--replay-mem', type=int)
parser.add_argument('--observation-t', type=int)
parser.add_argument('--lstm-num-units', type=int)
parser.add_argument('--lstm-num-layers', type=int)
parser.add_argument('--embedding-size', type=int)
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--total-t', default=sys.maxsize, type=int)
parser.add_argument('--game-episode-terminal-t', type=int)
parser.add_argument('--model-creator', type=str)
parser.add_argument('--tjs-creator', type=str)
parser.add_argument('--game-clazz', type=str)
parser.add_argument('--agent-clazz', type=str)
parser.add_argument('--max-action-len', type=int)
parser.add_argument('--eval-randomness', type=float)
parser.add_argument('--eval-mode', type=str, default="all")
parser.add_argument('--jitter-go', action='store_true')
parser.add_argument('--jitter-train-prob', type=float)
parser.add_argument('--jitter-eval-prob', type=float)
parser.add_argument('--collect-floor-plan', action='store_true')
parser.add_argument('--start-t-ignore-model-t', action='store_true')
parser.add_argument('--bert-ckpt-dir', type=str)
parser.add_argument('--bert-num-hidden-layers', type=int)
parser.add_argument('--apply-dependency-parser', action='store_true')
parser.add_argument('--use-padding-over-lines', action='store_true')
parser.add_argument('--n-actions', type=int)
parser.add_argument('--drop-w-theme-words', action='store_true')


def setup_train_log(model_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_config_file = '{}/../../conf/logging.yaml'.format(current_dir)
    setup_logging(
        default_path=log_config_file,
        local_log_filename=os.path.join(model_dir, 'game_script.log'))


def setup_eval_log(log_filename):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_config_file = '{}/../../conf/logging-eval.yaml'.format(current_dir)
    setup_logging(
        default_path=log_config_file,
        local_log_filename=log_filename)


def get_eval_log_filename(func_name, model_dir, game_path, f_games):
    fn = "tmp-log+{}+{}+{}+{}+{}.txt".format(
        func_name,
        os.path.basename(model_dir),
        os.path.basename(game_path),
        os.path.basename(f_games) if f_games else "None",
        str(round(time.time() * 1000)))
    return os.path.join(os.getcwd(), fn)


if __name__ == '__main__':
    args = parser.parse_args()
    config_file = args.config_file
    model_dir = args.model_dir
    game_path = args.game_path
    if model_dir:
        model_dir = model_dir.rstrip('/')

    if args.mode == "train-drrn":
        setup_train_log(model_dir)
        from deeptextworld.train_drrn import train_n_eval
        hp = load_hparams_for_training(config_file, args)
        train_n_eval(hp, model_dir, game_dir=game_path, f_games=args.f_games)
    elif args.mode == "eval-drrn":
        setup_eval_log(
            get_eval_log_filename(
                "eval_drrn", model_dir, game_path, args.f_games))
        from deeptextworld.train_drrn import run_eval
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        run_eval(
            hp, model_dir, game_path=game_path, f_games=args.f_games,
            eval_randomness=args.eval_randomness, eval_mode=args.eval_mode)
    elif args.mode == "train-dqn":
        setup_train_log(model_dir)
        from deeptextworld.train_dqn import train_n_eval
        hp = load_hparams_for_training(config_file, args)
        train_n_eval(hp, model_dir, game_file=game_path)
    elif args.mode == "eval-dqn":
        setup_eval_log(
            get_eval_log_filename(
                "eval_dqn", model_dir, game_path, args.f_games))
        from deeptextworld.train_dqn import run_eval
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        run_eval(
            hp, model_dir, game_file=game_path,
            eval_randomness=args.eval_randomness)
    elif args.mode == "train-dsqn":
        setup_train_log(model_dir)
        from deeptextworld.train_dsqn import train_n_eval
        hp = load_hparams_for_training(config_file, args)
        train_n_eval(hp, model_dir, game_dir=game_path, f_games=args.f_games)
    elif args.mode == "eval-dsqn":
        setup_eval_log(
            get_eval_log_filename(
                "eval_dsqn", model_dir, game_path, args.f_games))
        from deeptextworld.train_dsqn import run_eval
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        run_eval(
            hp, model_dir, game_path=game_path, f_games=args.f_games,
            eval_randomness=args.eval_randomness, eval_mode=args.eval_mode)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))
