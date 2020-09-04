import json
import sys

import numpy as np

from deepword.stats import mean_confidence_interval


def summary_from_keys(keys, eval_results):
    total_steps, total_max_scores, total_win = 0, 0, 0
    total_scores = np.zeros(2, dtype=np.float)
    for game_name in keys:
        game_res = eval_results[game_name]
        max_scores = game_res["max_scores"]
        earned_scores = np.asarray(list(map(lambda x: x["score"], game_res["runs"])))
        used_steps = sum(map(lambda x: x["steps"], game_res["runs"]))
        has_won = len(list(filter(lambda x: x["has_won"], game_res["runs"])))
        total_win += has_won
        total_scores += earned_scores
        total_steps += used_steps
        total_max_scores += max_scores
    total_max_steps = len(keys) * 2 * 100
    if total_max_steps == 0:
        return 0, 0
    sample_mean, confidence_interval = mean_confidence_interval(
        total_scores / total_max_scores)
    return (sample_mean,
            confidence_interval,
            total_steps * 1. / total_max_steps,
            total_win * 1. / (len(keys) * 2))


def merge_eval_results(j_res1, j_res2=None):
    eval_res = dict()
    eval_res.update(j_res1['games'])
    if j_res2:
        eval_res.update(j_res2['games'])
    return eval_res


def main(f_result, f_result2=None):
    with open(f_result, "r") as f:
        j_result = json.load(f)

    if f_result2:
        with open(f_result2, 'r') as f:
            j_result2 = json.load(f)
    else:
        j_result2 = None

    eval_results = merge_eval_results(j_result, j_result2)

    all_keys = list(eval_results.keys())

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
            map(lambda x: "{:.2f}".format(x),
                summary_from_keys(kk, eval_results)))
        print("{:>15}:  {}".format(nn, res))


if __name__ == '__main__':
    main(sys.argv[1], None)
