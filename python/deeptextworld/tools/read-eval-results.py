import json
import sys


def main(f_result):
    with open(f_result, "r") as f:
        j_result = json.load(f)

    all_keys = list(j_result["games"].keys())

    # notice there are 10 episodes for each game
    all_score = sum(list(map(
        lambda x: x['max_scores'] * 10, list(j_result['games'].values()))))

    keys_no_go9_go12 = list(filter(
        lambda k: "go9" in k or "go12" in k, all_keys))

    keys_no_go9_go12 = list(filter(lambda k: "go" not in k, all_keys))

    print("\n".join(sorted(keys_no_go9_go12)))
    print("number of eval games: {}".format(len(keys_no_go9_go12)))

    total_scores, total_steps, total_max_scores = 0, 0, 0
    for game_name in keys_no_go9_go12:
        game_res = j_result["games"][game_name]
        max_scores = game_res["max_scores"]
        earned_scores = sum(map(lambda x: x["score"], game_res["runs"]))
        used_steps = sum(map(lambda x: x["steps"], game_res["runs"]))
        total_scores += earned_scores
        total_steps += used_steps
        total_max_scores += max_scores * 10

    print("score: {}".format(total_scores))
    print("adjusted_score: {}".format(total_scores * 0.5))
    print("nb_steps: {}".format(total_steps))
    print("total available score: {}".format(total_max_scores))
    print("total available score for all eval games: {}".format(all_score))


if __name__ == '__main__':
    main(sys.argv[1])
