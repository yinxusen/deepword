import logging
import os

import gym
import numpy as np
import textworld.gym
from tqdm import trange

from deeptextworld.utils import load_and_split, agent_name2clazz

EPOCH_LIMIT = 5


def run_agent_eval(agent, game_files, max_episode_steps, memory_size):
    logger = logging.getLogger("eval")
    requested_infos = agent.select_additional_infos()
    env_id = textworld.gym.register_games(
        game_files, requested_infos, batch_size=1,
        max_episode_steps=max_episode_steps,
        name="eval")
    game_env = gym.make(env_id)

    obs, infos = game_env.reset()
    scores = [0] * len(obs)
    dones = [False] * len(obs)
    for epoch_t in trange(EPOCH_LIMIT):
        for total_t in trange(memory_size // 1000):
            if not all(dones):
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = game_env.step(commands)
            else:
                agent.act(obs, scores, dones, infos)
                obs, infos = game_env.reset()
                scores = [0] * len(obs)
                dones = [False] * len(obs)
                agent.eps = np.random.random() / 2
                logger.info("new randomness: {}".format(agent.eps))
        agent.save_snapshot()
        logger.info("save snapshot epoch: {}".format(epoch_t))


def run_eval(
        hp, model_dir, game_path, f_games=None, eval_randomness=None,
        eval_mode="all"):
    """
    Evaluation an agent.
    :param hp:
    :param model_dir:
    :param game_path:
    :param f_games:
    :param eval_randomness:
    :param eval_mode:
    :return:
    """
    logger = logging.getLogger("eval")
    if os.path.isdir(game_path):
        train_games, dev_games = load_and_split(game_path, f_games)

        game_files = None
        if eval_mode == "all":
            # remove possible repeated games
            game_files = list(set(train_games + dev_games))
        elif eval_mode == "eval-train":
            game_files = train_games
        elif eval_mode == "eval-eval":
            game_files = dev_games
        else:
            print("unknown evaluation mode."
                  " choose from [all|eval-train|eval-eval]")
            exit(-1)
    elif os.path.isfile(game_path):
        game_files = [game_path]
    else:
        print("game path doesn't exist")
        return

    logger.info("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    logger.debug("games for eval: \n{}".format("\n".join(sorted(game_names))))

    agent_clazz = agent_name2clazz(hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.eval(load_best=True)
    if eval_randomness is not None:
        agent.eps = eval_randomness
    logger.info("evaluation randomness: {}".format(agent.eps))
    run_agent_eval(
        agent, game_files, hp.game_episode_terminal_t, hp.replay_mem)