import glob
import logging
import os
import random
import sys
from threading import Thread, Condition

import gym
import textworld.gym
from textworld import EnvInfos

from deeptextworld.agents.drrn_agent import DRRNAgent
from deeptextworld.utils import ctime

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = EnvInfos(
    description=True, inventory=True,
    max_score=True, objective=True, entities=True, verbs=True,
    command_templates=True, admissible_commands=True,
    has_won=True, has_lost=True,
    extras=["recipe"]
)


def _validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


def run_agent(cv, agent, game_env, game_names, nb_epochs, batch_size):
    logger = logging.getLogger("train" if agent.is_training else "eval")
    eval_results = dict()
    game_id = None
    for epoch_no in range(nb_epochs):
        for game_no in range(len(game_names)):
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
                if game_id is None:
                    game_id = agent.game_id
                if agent.is_training and agent.snapshot_saved:
                    agent.snapshot_saved = False
                    with cv:
                        cv.notifyAll()
                obs, scores, dones, infos = game_env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            if not agent.is_training:
                if not game_id in eval_results:
                    eval_results[game_id] = []
                eval_results[game_id].append(
                    (scores[0], infos["max_score"][0], steps[0],
                     infos["has_won"][0]))
            game_id = None
    return eval_results


def train(hp, cv, model_dir, game_files, nb_epochs=sys.maxsize, batch_size=1):
    logger = logging.getLogger('train')
    logger.info("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]

    agent = DRRNAgent(hp, model_dir)
    agent.train()

    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files, requested_infos,
        max_episode_steps=hp.game_episode_terminal_t,
        name="training")
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size,
                                      parallel=False)
    env = gym.make(env_id)
    run_agent(cv, agent, env, game_names, nb_epochs, batch_size)


def evaluation(hp, cv, model_dir, game_files, nb_epochs, batch_size):
    logger = logging.getLogger("eval")
    logger.info('evaluation worker started ...')
    logger.info("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]

    agent = DRRNAgent(hp, model_dir)
    agent.eval()
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files, requested_infos,
        max_episode_steps=hp.game_episode_terminal_t,
        name="evaluation")
    env_id = textworld.gym.make_batch(
        env_id, batch_size=batch_size, parallel=False)
    env = None

    prev_total_scores = 0
    prev_total_steps = sys.maxsize

    while True:
        with cv:
            cv.wait()
            logger.info("start evaluation ...")
            agent.reset()
            # delay the env creation for multi-thread problem.
            if not env:
                env = gym.make(env_id)
            else:
                env.reset()
            eval_start_t = ctime()
            eval_results = run_agent(
                cv, agent, env, game_names, nb_epochs, batch_size)
            eval_end_t = ctime()
            agg_res, total_scores, total_steps = agg_results(eval_results)
            logger.info("eval_results: {}".format(eval_results))
            logger.info("eval aggregated results: {}".format(agg_res))
            logger.info("total scores: {}, total steps: {}".format(
                total_scores, total_steps))
            logger.info("time to finish eval: {}".format(eval_end_t-eval_start_t))
            if ((total_scores > prev_total_scores) or
                    ((total_scores == prev_total_scores) and
                     (total_steps < prev_total_steps))):
                logger.info("found better agent, save model ...")
                prev_total_scores = total_scores
                prev_total_steps = total_steps
                agent.save_best_model()
            else:
                logger.info("no better model, pass ...")


def run_main(hp, model_dir, game_dir, batch_size=1, max_games_used=None):
    game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    random.Random(42).shuffle(game_files)
    if max_games_used is not None:
        game_files = game_files[:max_games_used]

    if len(game_files) == 0:
        print("no game files found!")
        return -1
    elif len(game_files) == 1:
        train_games = game_files
        eval_games = game_files
        pass
    elif len(game_files) < 10:  # use the last one as eval
        train_games = game_files[:-1]
        eval_games = game_files[-1:]
    else:
        num_train = int(len(game_files) * 0.9)
        train_games = game_files[:num_train]
        eval_games = game_files[num_train:]
    cond_of_eval = Condition()
    eval_worker = Thread(
        name='eval_worker', target=evaluation,
        args=(hp, cond_of_eval, model_dir, eval_games, hp.eval_episode, 1))
    eval_worker.daemon = True
    eval_worker.start()

    train(hp, cond_of_eval, model_dir, train_games,
          nb_epochs=hp.annealing_eps_t, batch_size=batch_size)


def run_eval(hp, model_dir, game_dir, batch_size=1, eval_randomness=None):
    logger = logging.getLogger("eval")
    logger.info(hp.num_conv_filters)
    game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    logger.info("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]

    agent = DRRNAgent(hp, model_dir)
    agent.eval()
    if eval_randomness is not None:
        agent.eps = eval_randomness
    logger.info("evaluation randomness: {}".format(agent.eps))
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files, requested_infos,
        max_episode_steps=hp.game_episode_terminal_t,
        name="evaluation")
    env_id = textworld.gym.make_batch(
        env_id, batch_size=batch_size, parallel=False)
    env = gym.make(env_id)
    eval_start_t = ctime()
    eval_results = run_agent(
        None, agent, env, game_names, hp.eval_episode, batch_size)
    eval_end_t = ctime()
    logger.info("eval_results: {}".format(eval_results))
    logger.info("eval aggregated results: {}".format(agg_results(eval_results)))
    logger.info("time to finish eval: {}".format(eval_end_t-eval_start_t))


def agg_results(eval_results):
    ret_val = {}
    total_scores = 0
    total_steps = 0
    for game_id in eval_results:
        res = eval_results[game_id]
        agg_score = sum(map(lambda r: r[0], res))
        agg_max_score = sum(map(lambda r: r[1], res))
        agg_step = sum(map(lambda r: r[2], res))
        agg_nb_won = len(list(filter(lambda r: r[3] , res)))
        ret_val[game_id] = (agg_score, agg_max_score, agg_step, agg_nb_won)
        total_scores += agg_score
        total_steps += agg_step
    return ret_val, total_scores, total_steps
