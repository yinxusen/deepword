import glob
import logging
import os
import random
import sys
from threading import Thread, Condition

import gym
import textworld.gym
from textworld import EnvInfos

from deeptextworld.agents import drrn_agent
from deeptextworld.utils import ctime

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = EnvInfos(
    description=True, inventory=True,
    max_score=True, objective=True, entities=True, verbs=True,
    command_templates=True, admissible_commands=True,
    won=True, has_lost=True,
    extras=["recipe"]
)


def validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


def run_agent(cv, agent, game_env, nb_games, nb_epochs):
    """
    Run a train agent on given games.
    :param cv:
    :param agent:
    :param game_env:
    :param nb_games:
    :param nb_epochs:
    :return:
    """
    logger = logging.getLogger("train")
    for epoch_no in range(nb_epochs):
        for game_no in range(nb_games):
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
                if agent.snapshot_saved:
                    agent.snapshot_saved = False
                    with cv:
                        cv.notifyAll()
                obs, scores, dones, infos = game_env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)
    return None


def run_agent_eval(agent, game_files, nb_episodes, max_episode_steps):
    """
    Run an eval agent on given games.
    :param agent:
    :param game_files:
    :param nb_episodes:
    :param max_episode_steps:
    :return:
    """
    logger = logging.getLogger("eval")
    eval_results = dict()
    requested_infos = agent.select_additional_infos()
    validate_requested_infos(requested_infos)
    for game_no, game_file in enumerate(game_files):
        game_name = os.path.basename(game_file)
        env_id = textworld.gym.register_games(
            [game_file], requested_infos,
            max_episode_steps=max_episode_steps,
            name="eval")
        env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
        game_env = gym.make(env_id)

        for episode_no in range(nb_episodes):
            logger.info(
                "episode_no/game_no/game_name: {}/{}/{}".format(
                    episode_no, game_no, game_name))

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

            if not agent.is_training:
                if game_name not in eval_results:
                    eval_results[game_name] = []
                eval_results[game_name].append(
                    (scores[0], infos["max_score"][0], steps[0],
                     infos["won"][0]))
    return eval_results


def train(hp, cv, model_dir, game_files, nb_epochs=sys.maxsize, batch_size=1):
    logger = logging.getLogger('train')
    logger.info("load {} game files".format(len(game_files)))

    agent_clazz = getattr(drrn_agent, hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.train()

    requested_infos = agent.select_additional_infos()
    validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files, requested_infos,
        max_episode_steps=hp.game_episode_terminal_t,
        name="training")
    env_id = textworld.gym.make_batch(
        env_id, batch_size=batch_size, parallel=False)
    env = gym.make(env_id)
    run_agent(cv, agent, env, len(game_files), nb_epochs)


def evaluation(hp, cv, model_dir, game_files, nb_episodes):
    """
    A thread of evaluation.
    :param hp:
    :param cv:
    :param model_dir:
    :param game_files:
    :param nb_episodes:
    :return:
    """
    logger = logging.getLogger("eval")
    logger.info('evaluation worker started ...')
    logger.info("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    logger.debug("games for eval: \n{}".format("\n".join(sorted(game_names))))

    agent_clazz = getattr(drrn_agent, hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    # for eval during training, set load_best=False
    agent.eval(load_best=False)

    prev_total_scores = 0
    prev_total_steps = sys.maxsize

    while True:
        with cv:
            cv.wait()
            logger.info("start evaluation ...")
            agent.reset()
            eval_start_t = ctime()
            eval_results = run_agent_eval(
                agent, game_files, nb_episodes, hp.game_episode_terminal_t)
            eval_end_t = ctime()
            agg_res, total_scores, total_steps, n_won = agg_results(
                eval_results)
            logger.info("eval_results: {}".format(eval_results))
            logger.info("eval aggregated results: {}".format(agg_res))
            logger.info(
                "scores: {:.2f}, steps: {:.2f}, n_won: {:.2f}".format(
                    total_scores, total_steps, n_won))
            logger.info(
                "time to finish eval: {}".format(eval_end_t-eval_start_t))
            if ((total_scores > prev_total_scores) or
                    ((total_scores == prev_total_scores) and
                     (total_steps < prev_total_steps))):
                logger.info("found better agent, save model ...")
                prev_total_scores = total_scores
                prev_total_steps = total_steps
                agent.save_best_model()
            else:
                logger.info("no better model, pass ...")


def split_train_dev(game_files):
    """
    Split train/dev sets from given game files
    sort - shuffle w/ Random(42) - 90%/10% split
      - if #game_files < 10, then use the last one as dev set;
      - if #game_files == 1, then use the one as both train and dev.
    :param game_files: game files
    :return: None if game_files is empty, otherwise (train, dev)
    """
    # have to sort first, otherwise after shuffling the result is different
    # on different platforms, e.g. Linux VS MacOS.
    game_files = sorted(game_files)
    random.Random(42).shuffle(game_files)
    if len(game_files) == 0:
        print("no game files found!")
        return None
    elif len(game_files) == 1:
        train_games = game_files
        dev_games = game_files
    elif len(game_files) < 10:  # use the last one as eval
        train_games = game_files[:-1]
        dev_games = game_files[-1:]
    else:
        num_train = int(len(game_files) * 0.9)
        train_games = game_files[:num_train]
        dev_games = game_files[num_train:]
    return train_games, dev_games


def load_game_files(game_dir, f_games=None):
    """
    Choose games appearing in f_games in a given game dir. Return all games in
    the game dir if f_games is None.
    :param game_dir: a dir
    :param f_games: a file of game names
    :return: a list of games
    """
    if f_games is not None:
        with open(f_games, "r") as f:
            selected_games = map(lambda x: x.strip(), f.readlines())
        game_files = list(map(
            lambda x: os.path.join(game_dir, "{}.ulx".format(x)),
            selected_games))
    else:
        game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    return game_files


def train_n_eval(hp, model_dir, game_dir, f_games=None, batch_size=1):
    """
    Train an agent with periodical evaluation (every time saving a model).
    :param hp:
    :param model_dir:
    :param game_dir:
    :param f_games:
    :param batch_size:
    :return:
    """
    game_files = load_game_files(game_dir, f_games)
    games = split_train_dev(game_files)
    if games is None:
        exit(-1)
    train_games, dev_games = games
    cond_of_eval = Condition()
    eval_worker = Thread(
        name='eval_worker', target=evaluation,
        args=(hp, cond_of_eval, model_dir, dev_games, hp.eval_episode))
    eval_worker.daemon = True
    eval_worker.start()

    # nb epochs could only be an estimation since steps per episode is unknown
    nb_epochs = (hp.annealing_eps_t // len(train_games) // 10) + 1

    train(hp, cond_of_eval, model_dir, train_games,
          nb_epochs=nb_epochs, batch_size=batch_size)


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
        game_files = load_game_files(game_path, f_games)
        games = split_train_dev(game_files)
        if games is None:
            exit(-1)
        train_games, dev_games = games

        game_files = None
        if eval_mode == "all":
            # remove possible repeated games
            game_files = list(set(train_games + dev_games))
        elif eval_mode == "eval-train":
            game_files = train_games
        elif eval_mode == "eval-eval":
            game_files = dev_games
        else:
            print("unknown mode. choose from [all|eval-train|eval-eval]")
            exit(-1)
    elif os.path.isfile(game_path):
        game_files = [game_path]
    else:
        print("game path doesn't exist")
        return

    logger.info("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    logger.debug("games for eval: \n{}".format("\n".join(sorted(game_names))))

    agent_clazz = getattr(drrn_agent, hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.eval(load_best=True)
    if eval_randomness is not None:
        agent.eps = eval_randomness
    logger.info("evaluation randomness: {}".format(agent.eps))

    eval_start_t = ctime()
    eval_results = run_agent_eval(
        agent, game_files, hp.eval_episode, hp.game_episode_terminal_t)
    eval_end_t = ctime()
    agg_res, total_scores, total_steps, n_won = agg_results(eval_results)
    logger.info("eval_results: {}".format(eval_results))
    logger.info("eval aggregated results: {}".format(agg_res))
    logger.info("scores: {:.2f}, steps: {:.2f}, n_won: {:.2f}".format(
        total_scores, total_steps, n_won))
    logger.info("time to finish eval: {}".format(eval_end_t-eval_start_t))


def agg_results(eval_results):
    """
    Aggregate evaluation results.
    :param eval_results:
    :return:
    """
    ret_val = {}
    total_scores = 0
    total_steps = 0
    all_scores = 0
    all_episodes = 0
    all_won = 0
    for game_id in eval_results:
        res = eval_results[game_id]
        agg_score = sum(map(lambda r: r[0], res))
        agg_max_score = sum(map(lambda r: r[1], res))
        all_scores += agg_max_score
        all_episodes += len(res)
        agg_step = sum(map(lambda r: r[2], res))
        agg_nb_won = len(list(filter(lambda r: r[3] , res)))
        all_won += agg_nb_won
        ret_val[game_id] = (agg_score, agg_max_score, agg_step, agg_nb_won)
        total_scores += agg_score
        total_steps += agg_step
    all_steps = all_episodes * 100
    return (ret_val, total_scores * 1. / all_scores,
            total_steps * 1. / all_steps, all_won * 1. / all_episodes)
