import json
import sys


def summary_from_keys(keys, j_result):
    total_scores, total_steps, total_max_scores, total_win = 0, 0, 0, 0
    for game_name in keys:
        game_res = j_result["games"][game_name]
        max_scores = game_res["max_scores"]
        earned_scores = sum(map(lambda x: x["score"], game_res["runs"]))
        used_steps = sum(map(lambda x: x["steps"], game_res["runs"]))
        has_won = len(list(filter(lambda x: x["has_won"], game_res["runs"])))
        total_win += has_won
        total_scores += earned_scores
        total_steps += used_steps
        total_max_scores += max_scores * 10
    total_max_steps = len(keys) * 10 * 100
    if total_max_steps == 0:
        return 0, 0
    return (total_scores * 1. / total_max_scores,
            total_steps * 1. / total_max_steps,
            total_win * 1. / (len(keys) * 10))


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
    # k_tier4_1 = list(filter(lambda k: "go6" in k and "recipe1" in k, all_keys))
    # k_tier4_2 = list(filter(lambda k: "go6" in k and "recipe2" in k, all_keys))
    # k_tier4_3 = list(filter(lambda k: "go6" in k and "recipe3" in k, all_keys))
    # k_tier5_1 = list(filter(lambda k: "go9" in k and "recipe1" in k, all_keys))
    # k_tier5_2 = list(filter(lambda k: "go9" in k and "recipe2" in k, all_keys))
    # k_tier5_3 = list(filter(lambda k: "go9" in k and "recipe3" in k, all_keys))
    # k_tier6_1 = list(filter(lambda k: "go12" in k and "recipe1" in k, all_keys))
    # k_tier6_2 = list(filter(lambda k: "go12" in k and "recipe2" in k, all_keys))
    # k_tier6_3 = list(filter(lambda k: "go12" in k and "recipe3" in k, all_keys))

    k_wo_drop = list(filter(lambda k: "drop" not in k, all_keys))
    k_w_drop = list(filter(lambda k: "drop" in k, all_keys))

    k_r1 = list(filter(lambda k: "recipe1" in k, all_keys))

    all_tiers_keys = [k_tier1, k_tier2, k_tier3,
                      k_tier4,
                      k_tier5,
                      k_tier6,
                      all_keys, k_wo_drop, k_w_drop]
    all_tiers_names = ["tier1", "tier2", "tier3",
                       "tier4",
                       "tier5",
                       "tier6",
                       "all-tiers",
                       "w/o drop", "w/ drop"]
    for nn, kk in zip(all_tiers_names, all_tiers_keys):
        res = ",".join(
            map(lambda x: "{:.2f}".format(x), summary_from_keys(kk, j_result)))
        print("{:>15}:  {}".format(nn, res))


if __name__ == '__main__':
    main(sys.argv[1])
