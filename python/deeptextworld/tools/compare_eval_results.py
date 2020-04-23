import json
import sys
from collections import Counter


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


def reason_of_failure(last_commands):
    last_commands = list(filter(lambda a: a != "eat meal", last_commands))
    res = zip(Counter(last_commands).keys(), Counter(last_commands).values())
    res = list(reversed(sorted(res, key=lambda x: x[1])))
    return res


def most_common_steps(scoring_steps):
    """
    1. find the most common scoring steps,
    2. if ties, choose the one with the best score.
    :param scoring_steps:
    :return:
    """
    scoring_steps = list(filter(lambda x: len(x) != 0, scoring_steps))
    if len(scoring_steps) == 0:
        return []
    scoring_steps = map(
        lambda x: ",".join(map(lambda y: str(y), x)), scoring_steps)
    cnt = sorted(list(Counter(scoring_steps).items()), key=lambda x: x[1])
    most_val = cnt[-1][1]
    same_cnt_items = map(lambda k: (k[0], k[1], len(k[0].split(","))),
        filter(lambda x: x[1] == most_val, cnt))
    same_cnt_items = sorted(same_cnt_items, key=lambda x: x[2])
    return list(map(lambda x: int(x), same_cnt_items[-1][0].split(",")))


def format_steps(mcs, max_n_vals):
    mcs = mcs[:max_n_vals]
    retval = []
    prev_dat = 0
    for x in mcs:
        y = x - prev_dat
        retval.append(y)
        prev_dat = x

    return retval + ['-'] * (max_n_vals - len(retval))


def diff_from_keys(keys, j_result, j_result2):
    for game_name in keys:
        game_res = j_result["games"][game_name]
        game_res2 = j_result2["games"][game_name]
        max_scores = game_res["max_scores"]
        earned_scores = sum(map(lambda x: x["score"], game_res["runs"]))
        earned_scores2 = sum(map(lambda x: x["score"], game_res2["runs"]))
        used_steps = sum(map(lambda x: x["steps"], game_res["runs"]))
        used_steps2 = sum(map(lambda x: x["steps"], game_res2["runs"]))
        scoring_steps = list(map(lambda x: x["scoring_steps"], game_res["runs"]))
        scoring_steps2 = list(map(lambda x: x["scoring_steps"], game_res2["runs"]))
        has_won = len(list(filter(lambda x: x["has_won"], game_res["runs"])))
        has_won2 = len(list(filter(lambda x: x["has_won"], game_res2["runs"])))
        # print("{:>100}: {}, {}/{}, {}/{}, {}/{}".format(
        #     game_name,max_scores, earned_scores, earned_scores2,
        #     used_steps, used_steps2, has_won, has_won2))
        # print(reason_of_failure(map(lambda x: x["commands"][-1], game_res["runs"])))
        # print(reason_of_failure(map(lambda x: x["commands"][-1], game_res2["runs"])))
        max_vals = 11
        most_common_1 = most_common_steps(scoring_steps)
        most_common_2 = most_common_steps(scoring_steps2)
        if len(most_common_1) != 0 and len(most_common_1) == len(most_common_2):
            print(sum(most_common_1), ",", sum(most_common_2))
            # most_common_1 = format_steps(most_common_1, max_vals)
            # most_common_2 = format_steps(most_common_2, max_vals)
            # placeholder = " ".join(["0"] * 11)
            # game_name = game_name.split("-")[2]
            # print("{} {} {} {}".format(game_name, "CNN", " ".join([str(x) for x in most_common_1]), placeholder))
            # print("{} {} {} {}".format(game_name, "BERT", placeholder, " ".join([str(x) for x in most_common_2])))


def main(f_result, f_result2):
    with open(f_result, "r") as f:
        j_result = json.load(f)

    with open(f_result2, "r") as f:
        j_result2 = json.load(f)


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

    k_r1 = list(filter(lambda k: "recipe1" in k, all_keys))

    all_tiers_keys = [k_tier1, k_tier2, k_tier3
                      ]
    all_tiers_names = ["tier1", "tier2", "tier3"
                       ]
    for nn, kk in zip(all_tiers_names, all_tiers_keys):
        # print("-------{}-----------".format(nn))
        diff_from_keys(kk, j_result, j_result2)
        # print("------------------")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
