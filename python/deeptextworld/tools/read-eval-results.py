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
    if total_max_steps == 0:
        return 0, 0
    return (total_scores * 1. / total_max_scores,
            total_steps * 1. / total_max_steps)


def main(f_result):
    with open(f_result, "r") as f:
        j_result = json.load(f)

    all_keys = list(j_result["games"].keys())

    k_tier1 = list(filter(lambda k: "go" not in k and "recipe1" in k, all_keys))
    k_tier2 = list(filter(lambda k: "go" not in k and "recipe2" in k, all_keys))
    k_tier3 = list(filter(lambda k: "go" not in k and "recipe3" in k, all_keys))
    k_tier4_1 = list(filter(lambda k: "go6" in k and "recipe1" in k, all_keys))
    k_tier4_2 = list(filter(lambda k: "go6" in k and "recipe2" in k, all_keys))
    k_tier4_3 = list(filter(lambda k: "go6" in k and "recipe3" in k, all_keys))
    k_tier5_1 = list(filter(lambda k: "go9" in k and "recipe1" in k, all_keys))
    k_tier5_2 = list(filter(lambda k: "go9" in k and "recipe2" in k, all_keys))
    k_tier5_3 = list(filter(lambda k: "go9" in k and "recipe3" in k, all_keys))
    k_tier6_1 = list(filter(lambda k: "go12" in k and "recipe1" in k, all_keys))
    k_tier6_2 = list(filter(lambda k: "go12" in k and "recipe2" in k, all_keys))
    k_tier6_3 = list(filter(lambda k: "go12" in k and "recipe3" in k, all_keys))

    k_r1 = list(filter(lambda k: "recipe1" in k, all_keys))

    all_tiers_keys = [k_tier1, k_tier2, k_tier3,
                      k_tier4_1, k_tier4_2, k_tier4_3,
                      k_tier5_1, k_tier5_2, k_tier5_3,
                      k_tier6_1, k_tier6_2, k_tier6_3,
                      all_keys, k_r1]
    all_tiers_names = ["tier1", "tier2", "tier3",
                       "tier4-1", "tier4-2", "tier4-3",
                       "tier5-1", "tier5-2", "tier5-3",
                       "tier6-1", "tier6-2", "tier6-3",
                       "all-tiers",
                       "all-recipe1"]
    for nn, kk in zip(all_tiers_names, all_tiers_keys):
        res = ",".join(
            map(lambda x: "{:.2f}".format(x), summary_from_keys(kk, j_result)))
        print("{:>15}:  {}".format(nn, res))


if __name__ == '__main__':
    main(sys.argv[1])
