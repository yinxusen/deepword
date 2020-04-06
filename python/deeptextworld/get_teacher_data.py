import os

import fire
import gym
import numpy as np
import textworld.gym
from tqdm import trange

from deeptextworld.hparams import load_hparams, output_hparams
from deeptextworld.utils import load_and_split, agent_name2clazz, eprint


def run_agent_eval(
        agent, game_files, max_episode_steps, epoch_size, epoch_limit):
    requested_infos = agent.select_additional_infos()
    env_id = textworld.gym.register_games(
        game_files, requested_infos, batch_size=1,
        max_episode_steps=max_episode_steps,
        name="eval")
    game_env = gym.make(env_id)

    obs, infos = game_env.reset()
    scores = [0] * len(obs)
    dones = [False] * len(obs)
    for epoch_t in trange(epoch_limit):
        for _ in trange(epoch_size):
            if not all(dones):
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = game_env.step(commands)
            else:
                agent.act(obs, scores, dones, infos)
                obs, infos = game_env.reset()
                scores = [0] * len(obs)
                dones = [False] * len(obs)
                agent.eps = np.random.random()
                eprint("new randomness: {}".format(agent.eps))
        agent.save_snapshot()
        eprint("save snapshot epoch: {}".format(epoch_t))
    game_env.close()


def run_eval(
        model_dir, game_path, f_games=None, epoch_size=None, epoch_limit=5):
    """
    Evaluation an agent.
    """
    train_games, dev_games = load_and_split(game_path, f_games)
    game_files = train_games
    eprint("load {} game files".format(len(game_files)))
    game_names = [os.path.basename(fn) for fn in game_files]
    eprint("games for eval: \n{}".format("\n".join(sorted(game_names))))

    config_file = os.path.join(model_dir, 'hparams.json')
    hp = load_hparams(config_file, cmd_args=None, fn_pre_config=None)
    # TODO: important setup for gen-data
    hp.set_hparam("compute_policy_action_every_step", True)
    hp.set_hparam("max_snapshot_to_keep", 100)
    hp.set_hparam("agent_clazz", "CompetitionAgent")
    hp.set_hparam("use_step_wise_reward", True)

    eprint("generate data with the following config:")
    eprint(output_hparams(hp))

    agent_clazz = agent_name2clazz(hp.agent_clazz)
    agent = agent_clazz(hp, model_dir)
    agent.eval(load_best=True)
    if epoch_size is None:
        epoch_size = hp.replay_mem
    run_agent_eval(
        agent, game_files, hp.game_episode_terminal_t, epoch_size, epoch_limit)


if __name__ == "__main__":
    fire.Fire(run_eval)
