import argparse
import os
import sys

from deeptextworld.hparams import load_hparams_for_evaluation
from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.utils import setup_logging


parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument('-m', '--model_dir', type=str)
parser.add_argument('-d', '--data_dir', type=str)
parser.add_argument('-c', '--config_file', type=str)
parser.add_argument('--game_dir', type=str, help='[a dir|a game file]')
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--tgt_vocab_file', type=str)
parser.add_argument('--action_file', type=str)
parser.add_argument('--mode', default='training', help='[training|evaluation]')
parser.add_argument('--eval_episode', type=int)
parser.add_argument('--init_eps', type=float)
parser.add_argument('--final_eps', type=float)
parser.add_argument('--annealing_eps_t', type=int)
parser.add_argument('--init_gamma', type=float)
parser.add_argument('--final_gamma', type=float)
parser.add_argument('--annealing_gamma_t', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--save_gap_t', type=int)
parser.add_argument('--replay_mem', type=int)
parser.add_argument('--observation_t', type=int)
parser.add_argument('--lstm_num_units', type=int)
parser.add_argument('--lstm_num_layers', type=int)
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--total_t', default=sys.maxsize, type=int)
parser.add_argument('--game_episode_terminal_t', type=int)
parser.add_argument('--model_creator', type=str)
parser.add_argument('--tjs_creator', type=str)
parser.add_argument('--game_clazz', type=str)
parser.add_argument('--delay_target_network', type=int)
parser.add_argument('--max_action_len', type=int)
parser.add_argument('--eval_randomness', type=float)
parser.add_argument('--eval_mode', type=str, default="all")
parser.add_argument('--jitter_go', action='store_true')
parser.add_argument('--collect_floor_plan', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    config_file = args.config_file
    game_path = args.data_dir
    model_dir = args.model_dir

    if args.mode == "train-drrn":
        from deeptextworld.train_drrn import run_main
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=os.path.join(model_dir, 'game_script.log'))
        hp = load_hparams_for_training(config_file, args)
        run_main(hp, model_dir, game_dir=args.game_dir)
    elif args.mode == "eval-drrn":
        from deeptextworld.train_drrn import run_eval
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging-eval.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=None)
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        run_eval(
            hp, model_dir, game_path=args.game_dir,
            eval_randomness=args.eval_randomness, eval_mode=args.eval_mode)
    elif args.mode == "train-dqn":
        from deeptextworld.train_dqn import run_main
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=os.path.join(model_dir, 'game_script.log'))
        hp = load_hparams_for_training(config_file, args)
        run_main(hp, model_dir, game_dir=args.game_dir)
    elif args.mode == "eval-dqn":
        from deeptextworld.train_dqn import run_eval
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging-eval.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=None)
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        run_eval(
            hp, model_dir, game_path=args.game_dir,
            eval_randomness=args.eval_randomness, eval_mode=args.eval_mode)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))

