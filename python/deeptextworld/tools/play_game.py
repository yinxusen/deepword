import fire
from textworld import EnvInfos

from deeptextworld.agents.utils import INFO_KEY
from deeptextworld.tools.collect_game_elements import CollectorAgent, run_games


class HumanAgent(CollectorAgent):
    @classmethod
    def request_infos(cls):
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.command_templates = True
        request_infos.max_score = True
        request_infos.won = True
        request_infos.extras = ["recipe"]
        request_infos.admissible_commands = True
        return request_infos

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def act(self, obs, scores, dones, infos):
        actions = infos[INFO_KEY.actions][0]
        print("----------------------")
        print(obs[0])
        state_text = (
            infos[INFO_KEY.desc][0] + "\n" + infos[INFO_KEY.inventory][0])
        print("--------------state text--------------")
        print(state_text)
        print("----------------------")
        print("\n".join(actions))
        print("\n")
        print("----------------------")
        print(infos.keys())
        print("----------------------")
        print("won: {}".format(infos[INFO_KEY.won][0]))
        print("----------------------")
        print(infos[INFO_KEY.verbs][0])
        print("----------------------")
        print(infos[INFO_KEY.templates][0])
        print("----------------------")
        print(infos[INFO_KEY.entities][0])
        print("----------------------")
        print(infos[INFO_KEY.recipe])
        action = input("> ")
        return action


def human_play(game_file, nb_episodes=1, max_steps=100):
    agent = HumanAgent()
    run_games(agent, [game_file], nb_episodes, max_steps)


if __name__ == '__main__':
    fire.Fire(human_play)
