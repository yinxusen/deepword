import glob
import logging
import os
import sys
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool
from os.path import join as pjoin

import gym
import tensorflow as tf
import textworld.gym
from tensorflow.contrib.training import HParams
from tqdm import trange

from deeptextworld.eval_games import MultiGPUsEvalPlayer, LoopDogEvalPlayer, \
    FullDirEvalPlayer
from deeptextworld.hparams import load_hparams
from deeptextworld.utils import agent_name2clazz, learner_name2clazz
from deeptextworld.utils import load_and_split, load_game_files
from deeptextworld.utils import setup_train_log, setup_eval_log, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


def hp_parser() -> ArgumentParser:
    # TODO: "store_true" defaults to be False, so use an explict default=None
    parser = ArgumentParser(argument_default=None)
    parser.add_argument('--model-creator', type=str)
    parser.add_argument('--agent-clazz', type=str)
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--init-eps', type=float)
    parser.add_argument('--final-eps', type=float)
    parser.add_argument('--annealing-eps-t', type=int)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--eval-episode', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--save-gap-t', type=int)
    parser.add_argument('--replay-mem', type=int)
    parser.add_argument('--observation-t', type=int)
    parser.add_argument('--total-t', default=sys.maxsize, type=int)
    parser.add_argument('--game-episode-terminal-t', type=int)
    parser.add_argument(
        '--collect-floor-plan', action='store_true', default=None)
    parser.add_argument(
        '--start-t-ignore-model-t', action='store_true', default=None)
    parser.add_argument('--n-actions', type=int)
    parser.add_argument(
        '--use-step-wise-reward', action='store_true', default=None)
    parser.add_argument(
        '--compute-policy-action-every-step', action='store_true', default=None)
    parser.add_argument("--tokenizer-type", type=str, help="[bert|albert|nltk]")
    parser.add_argument("--max-snapshot-to-keep", type=int)
    return parser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        argument_default=None, parents=[hp_parser()],
        conflict_handler='resolve')
    parser.add_argument('--model-dir', type=str, required=True)

    subparsers = parser.add_subparsers(dest='mode')

    teacher_parser = subparsers.add_parser('train-dqn')
    teacher_parser.add_argument(
        '--game-path', type=str, help='[a dir|a game file]', required=True)
    teacher_parser.add_argument('--f-games', type=str)

    eval_parser = subparsers.add_parser('eval-dqn')
    eval_parser.add_argument(
        '--eval-mode', type=str, default='eval',
        help='[eval|dev-eval|full-eval]')
    eval_parser.add_argument('--game-path', type=str, required=True)
    eval_parser.add_argument('--f-games', type=str)
    eval_parser.add_argument('--n-gpus', type=int, default=1)
    eval_parser.add_argument('--debug', action='store_true')
    eval_parser.add_argument('--load-best', action='store_true')
    eval_parser.add_argument('--restore-from', type=str)

    student_parser = subparsers.add_parser('train-student')
    student_parser.add_argument('--data-path', type=str, required=True)
    student_parser.add_argument('--learner-clazz', type=str)
    student_parser.add_argument('--n-epochs', type=int, default=1000)

    student_eval_parser = subparsers.add_parser('eval-student')
    student_eval_parser.add_argument('--data-path', type=str, required=True)
    student_eval_parser.add_argument('--learner-clazz', type=str)
    student_eval_parser.add_argument('--n-gpus', type=int, default=1)
    return parser


def run_agent(agent, game_env, nb_games, nb_epochs):
    """
    Run a train agent on given games.
    :param agent:
    :param game_env:
    :param nb_games:
    :param nb_epochs:
    :return:
    """
    logger = logging.getLogger("train")
    for epoch_no in trange(nb_epochs):
        for game_no in trange(nb_games):
            logger.info("playing game epoch_no/game_no: {}/{}".format(
                epoch_no, game_no))

            obs, infos = game_env.reset()
            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            while not all(dones):
                # Increase step counts.
                steps = ([step + int(not done)
                          for step, done in zip(steps, dones)])
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = game_env.step(commands)
            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)


def train(hp, model_dir, game_dir, f_games=None):
    logger = logging.getLogger('train')
    train_games, _ = load_and_split(game_dir, f_games)
    logger.info("load {} game files".format(len(train_games)))
    # nb epochs could only be an estimation since steps per episode is unknown
    nb_epochs = (hp.annealing_eps_t // len(train_games) // 10) + 1

    agent_clazz = agent_name2clazz(hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.train()

    requested_infos = agent.select_additional_infos()

    env_id = textworld.gym.register_games(
        train_games, requested_infos, batch_size=1,
        max_episode_steps=hp.game_episode_terminal_t,
        name="training")
    env = gym.make(env_id)
    try:
        run_agent(agent, env, len(train_games), nb_epochs)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error("error: {}".format(e))
        traceback.print_exception(
            exc_type, exc_value, exc_traceback, limit=None, file=sys.stdout)
    env.close()


def process_hp(args) -> HParams:
    fn_hparams = os.path.join(args.model_dir, "hparams.json")
    if os.path.isfile(fn_hparams):
        model_config_file = fn_hparams
    else:
        model_config_file = None

    hp = load_hparams(
        fn_model_config=model_config_file, cmd_args=vars(args),
        fn_pre_config=args.config_file)
    return hp


def process_train_dqn(args):
    hp = process_hp(args)
    setup_train_log(args.model_dir)
    train(hp, args.model_dir, game_dir=args.game_path, f_games=args.f_games)


def process_train_student(args):
    hp = process_hp(args)
    setup_train_log(args.model_dir)
    learner_clazz = learner_name2clazz(hp.learner_clazz)
    learner = learner_clazz(hp, args.model_dir, args.data_path)
    learner.train(n_epochs=args.n_epochs)


def eval_one_ckpt(hp, model_dir, data_path, learner_clazz, device, ckpt_path):
    tester = learner_clazz(
        hp, model_dir, train_data_dir=None, eval_data_path=data_path)
    acc, total = tester.test(device, ckpt_path)
    return str(acc * 1. / total) if total != 0 else "Nan"


def process_eval_student(args):
    hp = process_hp(args)
    assert hp.learner_clazz == "SwagLearner"
    learner_clazz = learner_name2clazz(hp.learner_clazz)

    n_gpus = args.n_gpus
    gpus = ["/device:GPU:{}".format(i) for i in range(n_gpus)]
    watched_file_regex = pjoin(
        args.model_dir, "last_weights", "after-epoch-*.index")
    files = glob.glob(watched_file_regex)
    ckpt_files = [os.path.splitext(f)[0] for f in files]
    eprint("evaluate {} checkpoints".format(len(ckpt_files)))
    if len(ckpt_files) == 0:
        return

    files_colocate_gpus = [
        ckpt_files[i * n_gpus:(i + 1) * n_gpus]
        for i in range((len(ckpt_files) + n_gpus - 1) // n_gpus)]

    for batch_files in files_colocate_gpus:
        pool = Pool(n_gpus)
        eprint("process: {}".format(batch_files))
        results = []
        for k in range(n_gpus):
            if k < len(batch_files):
                res = pool.apply_async(
                    eval_one_ckpt, args=(
                        hp, args.model_dir, args.data_path, learner_clazz,
                        gpus[k], batch_files[k]))
                results.append(res)

        for k, res in enumerate(results):
            eprint("model: {}, res: {}".format(batch_files[k], res.get()))
        pool.close()
        pool.join()


def process_eval_dqn(args):
    hp = process_hp(args)
    setup_eval_log(log_filename="/tmp/eval-logging.txt")
    if args.eval_mode == "eval":
        game_files = load_game_files(args.game_path, args.f_games)
        eval_player = MultiGPUsEvalPlayer(
            hp, args.model_dir, game_files, args.n_gpus, args.load_best)
        eval_player.evaluate(
            restore_from=args.restore_from, debug=args.debug)
    elif args.eval_mode == "dev-eval":
        _, eval_games = load_and_split(args.game_path, args.f_games)
        eval_player = LoopDogEvalPlayer()
        eval_player.start(hp, args.model_dir, eval_games, args.n_gpus)
    elif args.eval_mode == "full-eval":
        _, eval_games = load_and_split(args.game_path, args.f_games)
        eval_player = FullDirEvalPlayer()
        eval_player.start(hp, args.model_dir, eval_games, args.n_gpus)
    else:
        raise ValueError()


def main(args):
    args.model_dir = args.model_dir.rstrip('/')
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    if args.mode == "train-dqn":
        process_train_dqn(args)
    elif args.mode == "train-student":
        process_train_student(args)
    elif args.mode == "eval-student":
        process_eval_student(args)
    elif args.mode == "eval-dqn":
        process_eval_dqn(args)


if __name__ == '__main__':
    main(get_parser().parse_args())
