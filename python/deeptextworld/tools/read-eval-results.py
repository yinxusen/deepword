import json
import sys


def summary_from_keys(keys, j_result):
    total_scores, total_steps, total_max_scores = 0, 0, 0
    for game_name in keys:
        game_res = j_result["games"][game_name]
        max_scores = game_res["max_scores"]
        earned_scores = sum(map(lambda x: x["score"], game_res["runs"]))
        used_steps = sum(map(lambda x: x["steps"], game_res["runs"]))
        total_scores += earned_scores
        total_steps += used_steps
        total_max_scores += max_scores * 10
    total_max_steps = len(keys) * 10 * 100
    return (total_scores * 1. / total_max_scores,
            total_steps * 1. / total_max_steps)


def main(f_result):
    with open(f_result, "r") as f:
        j_result = json.load(f)

    all_keys = list(j_result["games"].keys())

    k_tier1 = list(filter(lambda k: "go" not in k and "recipe1" in k, all_keys))
    k_tier2 = list(filter(lambda k: "go" not in k and "recipe2" in k, all_keys))
    k_tier3 = list(filter(lambda k: "go" not in k and "recipe3" in k, all_keys))
    k_tier4 = list(filter(lambda k: "go6" in k, all_keys))
    k_tier5 = list(filter(lambda k: "go9" in k, all_keys))
    k_tier6 = list(filter(lambda k: "go12" in k, all_keys))

    all_tiers_keys = [k_tier1, k_tier2, k_tier3, k_tier4, k_tier5, k_tier6, all_keys]
    for kk in all_tiers_keys:
        print(",".join(map(
            lambda x: "{:.2f}".format(x), summary_from_keys(kk, j_result))))


if __name__ == '__main__':
    main(sys.argv[1])
