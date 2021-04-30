#!/usr/bin/env python

import argparse
import glob
import json
import os
from os.path import join as pjoin

from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.info import ALFWORLD_DATA
from tqdm import tqdm


def main(problems, domain_path, grammar_path):
    for problem in tqdm(problems):
        game_logic = {
            "pddl_domain": open(domain_path).read(),
            "grammar": open(grammar_path).read(),
        }
        # load state and trajectory files
        pddl_file = os.path.join(problem, 'initial_state.pddl')
        json_file = os.path.join(problem, 'traj_data.json')
        with open(json_file, 'r') as f:
            traj_data = json.load(f)
        game_logic['grammar'] = add_task_to_grammar(
            game_logic['grammar'], traj_data)

        # dump game file
        gamedata = dict(**game_logic, pddl_problem=open(pddl_file).read())
        gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
        json.dump(gamedata, open(gamefile, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_dir", type=str, default=None)
    parser.add_argument("--domain",
                        default=pjoin(ALFWORLD_DATA, "logic", "alfred.pddl"),
                        help="Path to a PDDL file describing the domain."
                             " Default: `%(default)s`.")
    parser.add_argument("--grammar",
                        default=pjoin(ALFWORLD_DATA, "logic", "alfred.twl2"),
                        help="Path to a TWL2 file defining the grammar used to generated text feedbacks."
                             " Default: `%(default)s`.")
    args = parser.parse_args()
    problem_dirs = [
        os.path.dirname(x) for x in glob.glob(
            pjoin(args.game_dir, "**", "initial_state.pddl"), recursive=True)]
    main(problem_dirs, args.domain, args.grammar)
