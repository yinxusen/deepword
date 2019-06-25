import json
import sys


def parse_one_game(g_name, j_game):
    run1 = j_game["runs"][0]
    run2 = j_game["runs"][1]
    g_type = g_name.split("-")[2]
    if run1["has_won"] or run2["has_won"]:
        return ("{:>40}".format(g_type),
                "{:6}".format(""),
                "{:30}".format(""),
                "{:3}".format(""),
                "{:6}".format(""),
                "{:30}".format(""),
                "{:3}".format(""))
    return ("{:>40}".format(g_type),
            "{:6}".format("true" if run1["has_won"] else "false"),
            "{:30}".format(run1["commands"][-1][:30]),
            "{:3d}".format(run1["steps"]),
            "{:6}".format("true" if run2["has_won"] else "false"),
            "{:30}".format(run2["commands"][-1][:30]),
            "{:3d}".format(run2["steps"]))


def count_failed_twice(g_name, j_game):
    run1 = j_game["runs"][0]
    run2 = j_game["runs"][1]
    g_type = g_name.split("-")[2]
    return ("{:>40}".format(g_type),
            "{:6}".format("true" if run1["has_won"] else "false"),
            "{:6}".format("true" if run2["has_won"] else "false"))


def repeat_action_error(g_name, j_game):
    run1 = j_game["runs"][0]
    run2 = j_game["runs"][1]
    g_type = g_name.split("-")[2]
    if run1["has_won"] or run2["has_won"]:
        pass
    else:
        if run1["steps"] == 100:
            print("{}, exceed step limit".format(g_type))
        elif run1["commands"][-1] in run1["commands"][:-1]:
            print("{}, repeat dangerous action: {}".format(g_type, run1["commands"][-1]))
        else:
            print("{}, unknown error: {}".format(g_type, run1["commands"][-1]))


def summary_from_keys(keys, j_result):
    for game_name in keys:
        game_res = j_result["games"][game_name]
        # repeat_action_error(game_name, game_res)
        res = parse_one_game(game_name, game_res)
        print(" ".join(res))


def main(f_result):
    with open(f_result, "r") as f:
        j_result = json.load(f)

    all_keys = sorted(list(j_result["games"].keys()))
    summary_from_keys(all_keys, j_result)


if __name__ == '__main__':
    main(sys.argv[1])
