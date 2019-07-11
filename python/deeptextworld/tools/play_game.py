import sys
import re

import gym
import textworld.gym
from textworld import EnvInfos


def contain_words(sentence, words):
    return any(map(lambda w: w in sentence, words))

def contain_theme_words(theme_words, actions):
    if theme_words is None:
        return actions
    contained = []
    others = []
    for a in actions:
        if contain_words(a, theme_words):
            contained.append(a)
        else:
            others.append(a)

    return contained, others


def main(game_file):
    request_infos = EnvInfos()
    request_infos.description = True
    request_infos.inventory = True
    request_infos.entities = True
    request_infos.verbs = True
    request_infos.command_templates = True
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
    theme_regex = ".*Ingredients:<\|>(.*)<\|>Directions.*"
    theme_words_search = re.search(theme_regex, infos["extra.recipe"][0].replace("\n", "<|>"))
    if theme_words_search:
        theme_words = theme_words_search.group(1)
        theme_words = list(
            filter(lambda w: w != "",
                   map(lambda w: w.strip(), theme_words.split("<|>"))))
    else:
        theme_words = None

    while not all(dones):
        # populate my own admissible actions
        admissible_commands = infos["admissible_commands"][0]
        contained, others = contain_theme_words(theme_words, admissible_commands)
        actions = ["inventory", "look"]
        actions += contained
        actions += list(filter(lambda a: a.startswith("go"), admissible_commands))
        actions = list(filter(lambda c: not c.startswith("examine"), actions))
        actions = list(filter(lambda c: not c.startswith("close"), actions))
        actions = list(filter(lambda c: not c.startswith("insert"), actions))
        actions = list(filter(lambda c: not c.startswith("eat"), actions))
        actions = list(filter(lambda c: not c.startswith("drop"), actions))
        actions = list(filter(lambda c: not c.startswith("put"), actions))
        other_valid_commands = {"prepare meal", "eat meal", "examine cookbook"}
        actions += list(filter(lambda a: a in other_valid_commands, admissible_commands))
        actions += list(filter(
            lambda a: (a.startswith("drop") and
                       all(map(lambda t: t not in a, theme_words))), others))
        actions += list(filter(lambda a: a.startswith("take") and "knife" in a, others))
        actions += list(filter(lambda a: a.startswith("open"), others))
        print("----------------------")
        print(obs[0])
        print("----------------------")
        print("\n".join(admissible_commands))
        print("\n")
        print("{} actions reduced".format(len(admissible_commands) - len(actions)))
        print("{}".format("\n".join(others)))
        print("----------------------")
        print(infos["extra.recipe"][0])
        print("----------------------")
        print("has won: {}".format(infos["has_won"][0]))
        print("----------------------")
        print(infos["verbs"][0])
        print("----------------------")
        print(infos["command_templates"][0])
        print("----------------------")
        print(infos["entities"][0])
        command = input("> ")
        obs, scores, dones, infos = env.step([command])

    print(obs[0])
    print(infos["extra.recipe"][0])
    print("has won: {}".format(infos["has_won"][0]))


if __name__ == '__main__':
    main(sys.argv[1])
