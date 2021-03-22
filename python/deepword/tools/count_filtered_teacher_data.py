import glob
import os

import fire
import numpy as np
from deeptextworld.agents.base_agent import DRRNMemoTeacher

from deepword.utils import load_uniq_lines


def count_data(data_dir, fn_allowed_gids):
    # filter allowed gids from memory during training
    # if allowed_gids is empty, use all memory
    # allowed_gids.txt:
    # game name [TAB] game ID
    allowed_gids = set()
    if os.path.isfile(fn_allowed_gids):
        allowed_gids = set(
            [x.split("\t")[1] for x in load_uniq_lines(fn_allowed_gids)])

    memo_files = glob.glob(os.path.join(data_dir, "memo*npz"))
    counter_allowed = 0
    counter_all = 0
    for fn_memo in memo_files:
        memo = np.load(fn_memo, allow_pickle=True)['data']
        memo = [x for x in memo if isinstance(x, DRRNMemoTeacher)]
        cnt_all = len(memo)
        if allowed_gids:
            memo = [x for x in memo if x.gid in allowed_gids]
        cnt_allowed = len(memo)
        print("file: {}, allowed: {}, all: {}, percentage: {}".format(
            os.path.basename(fn_memo), cnt_allowed, cnt_all,
            cnt_allowed * 100. / cnt_all))

        counter_all += cnt_all
        counter_allowed += cnt_allowed

    print("overall, allowed: {}, all: {}, percentage: {}".format(
        counter_allowed, counter_all, counter_allowed * 100. / counter_all))


if __name__ == '__main__':
    fire.Fire(count_data)
