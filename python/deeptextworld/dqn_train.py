import argparse
import logging
import os
import sys
import traceback

import gym
import tensorflow as tf
import textworld.gym
from tqdm import trange

from deeptextworld.hparams import load_hparams
from deeptextworld.utils import agent_name2clazz
from deeptextworld.utils import load_and_split
from deeptextworld.utils import setup_logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument('-m', '--model-dir', type=str, required=True)
parser.add_argument(
    '--game-path', type=str, help='[a dir|a game file]', required=True)
parser.add_argument('--f-games', type=str)
parser.add_argument('--model-creator', type=str, default="CnnDRRN")
parser.add_argument('--config-file', type=str)
parser.add_argument('--init-eps', type=float)
parser.add_argument('--final-eps', type=float)
parser.add_argument('--annealing-eps-t', type=int)
parser.add_argument('--gamma', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--save-gap-t', type=int)
parser.add_argument('--replay-mem', type=int)
parser.add_argument('--observation-t', type=int)
parser.add_argument('--total-t', default=sys.maxsize, type=int)
parser.add_argument('--game-episode-terminal-t', type=int)
parser.add_argument('--collect-floor-plan', action='store_true')
parser.add_argument('--start-t-ignore-model-t', action='store_true')
parser.add_argument('--n-actions', type=int)
parser.add_argument('--use-step-wise-reward', action='store_true')
parser.add_argument('--compute-policy-action-every-step', action='store_true')
parser.add_argument("--tokenizer-type", type=str, help="[bert|albert|nltk]")
parser.add_argument("--max-snapshot-to-keep", type=int)


def setup_train_log(model_dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_config_file = '{}/../../conf/logging.yaml'.format(current_dir)
    setup_logging(
        default_path=log_config_file,
        local_log_filename=os.path.join(model_dir, 'game_script.log'))


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


def main(args):
    model_dir = args.model_dir.rstrip('/')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    game_path = args.game_path
    config_file = args.config_file
    if not config_file:
        f_hparams = os.path.join(model_dir, "hparams.json")
        if os.path.isfile(f_hparams):
            config_file = f_hparams

    setup_train_log(model_dir)
    hp = load_hparams(file_args=config_file, cmd_args=args)
    train(hp, model_dir, game_dir=game_path, f_games=args.f_games)


if __name__ == '__main__':
    main(parser.parse_args())
