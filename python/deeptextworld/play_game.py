import sys

import gym
import textworld.gym
from textworld import EnvInfos


def main(game_file):
    request_infos = EnvInfos()
    request_infos.description = True
    request_infos.inventory = True
    request_infos.entities = True
    request_infos.verbs = True
    request_infos.max_score = True
    request_infos.has_won = True
    request_infos.extras = ["recipe"]
    request_infos.admissible_commands = True

    game_files = [game_file]
    env_id = textworld.gym.register_games(
        game_files, request_infos,
        max_episode_steps=100,
        name="human")
    env_id = textworld.gym.make_batch(
        env_id, batch_size=1, parallel=False)
    env = gym.make(env_id)
    obs, infos = env.reset()
    dones = [False] * len(obs)
    print("max score is {}".format(infos["max_score"][0]))
    while not all(dones):
        print("----------------------")
        print(obs[0])
        print("----------------------")
        print("\n".join(infos["admissible_commands"][0]))
        print("----------------------")
        print(infos["extra.recipe"][0])
        print("----------------------")
        print("has won: {}".format(infos["has_won"][0]))
        print("----------------------")
        print(infos["verbs"][0])
        print("----------------------")
        print(infos["entities"][0])
        command = input("> ")
        obs, scores, dones, infos = env.step([command])

    print(obs[0])
    print(infos["extra.recipe"][0])
    print("has won: {}".format(infos["has_won"][0]))


if __name__ == '__main__':
    main(sys.argv[1])
