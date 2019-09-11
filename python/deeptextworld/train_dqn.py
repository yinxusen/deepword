import logging
import sys
from threading import Thread, Condition

import gym
import textworld.gym

from deeptextworld.agents.dqn_agent import TabularDQNAgent
from deeptextworld.train_drrn import validate_requested_infos, run_agent, \
    run_agent_eval, agg_results
from deeptextworld.utils import ctime


def train(hp, cv, model_dir, game_file, nb_epochs=sys.maxsize, batch_size=1):
    agent = TabularDQNAgent(hp, model_dir)
    agent.train()

    requested_infos = agent.select_additional_infos()
    validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        [game_file], requested_infos,
        max_episode_steps=hp.game_episode_terminal_t,
        name="training")
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size,
                                      parallel=False)
    env = gym.make(env_id)
    run_agent(cv, agent, env, nb_games=1, nb_epochs=nb_epochs)


def evaluation(hp, cv, model_dir, game_file, nb_episodes):
    logger = logging.getLogger("eval")
    logger.info('evaluation worker started ...')

    agent = TabularDQNAgent(hp, model_dir)
    agent.eval()

    prev_total_scores = 0
    prev_total_steps = sys.maxsize

    while True:
        with cv:
            cv.wait()
            logger.info("start evaluation ...")
            agent.reset()

            eval_start_t = ctime()
            eval_results = run_agent_eval(
                agent, [game_file], nb_episodes, hp.game_episode_terminal_t)
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


def train_n_eval(hp, model_dir, game_file, batch_size=1):
    cond_of_eval = Condition()
    eval_worker = Thread(
        name='eval_worker', target=evaluation,
        args=(hp, cond_of_eval, model_dir, game_file, hp.eval_episode))
    eval_worker.daemon = True
    eval_worker.start()

    nb_epochs = (hp.annealing_eps_t // 10) + 1
    train(hp, cond_of_eval, model_dir, game_file,
          nb_epochs=nb_epochs, batch_size=batch_size)


def run_eval(hp, model_dir, game_file, eval_randomness=None):
    logger = logging.getLogger("eval")
    agent = TabularDQNAgent(hp, model_dir)
    agent.eval(load_best=True)
    if eval_randomness is not None:
        agent.eps = eval_randomness
    logger.info("evaluation randomness: {}".format(agent.eps))

    eval_start_t = ctime()
    eval_results = run_agent_eval(
        agent, [game_file], hp.eval_episode, hp.game_episode_terminal_t)
    eval_end_t = ctime()
    agg_res, total_scores, total_steps, n_won = agg_results(eval_results)
    logger.info("eval_results: {}".format(eval_results))
    logger.info("eval aggregated results: {}".format(agg_res))
    logger.info(
        "total scores: {:.2f}, total steps: {:.2f}, n_won: {:.2f}".format(
            total_scores, total_steps, n_won))
    logger.info("time to finish eval: {}".format(eval_end_t-eval_start_t))
