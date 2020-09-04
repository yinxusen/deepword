import os
from os.path import join as pjoin
from threading import Thread

import fire
import numpy as np

from deepword.utils import eprint


def clean_hs2tj(hash_states2tjs, tjs):
    """
    Due to code problem, some hash_states2tjs contains states that don't have
    tjs companions.
    These states should be removed, otherwise the request for their companions
    would cause error.

    :param hash_states2tjs: a map from states to tjs. States are composed of
      observation + inventory
    :param tjs:
    :return:
    """
    cnt_trashed = 0
    empty_keys = []
    all_tids = set(tjs.keys())
    for k in hash_states2tjs.keys():
        empty_tids = []
        for tid in hash_states2tjs[k].keys():
            if tid not in all_tids:
                empty_tids.append(tid)
                cnt_trashed += len(hash_states2tjs[k][tid])
        for tid in empty_tids:
            hash_states2tjs[k].pop(tid, None)
        if hash_states2tjs[k] == {}:  # delete the dict if empty
            empty_keys.append(k)
    eprint("hs2tj deletes {} items".format(cnt_trashed))
    for k in empty_keys:
        hash_states2tjs.pop(k, None)
    eprint("hs2tj deletes {} keys".format(len(empty_keys)))
    return hash_states2tjs


def load_snapshot(raw_tjs_path, hs2tj_path):
    trajectories = {}
    tjs = np.load(raw_tjs_path, allow_pickle=True)
    curr_tid = tjs["curr_tid"][0]
    curr_tj = list(tjs["curr_tj"][0])
    tids = tjs["tids"]
    vals = tjs["vals"]
    assert len(tids) == len(vals), "incompatible trajectory ids and values"
    for i in range(len(tids)):
        trajectories[tids[i]] = list(vals[i])
    if curr_tid not in trajectories:
        trajectories[curr_tid] = curr_tj

    hs2tj = np.load(hs2tj_path, allow_pickle=True)
    hash_states2tjs = hs2tj["hs2tj"][0]

    return trajectories, hash_states2tjs


def clean_data(tp, hs):
    eprint("cleaning\n{}\n{}".format(tp, hs))
    hs_prefix = os.path.splitext(hs)[0]
    tjs, hs2tj = load_snapshot(tp, hs)
    cleaned_hs2tj = clean_hs2tj(hs2tj, tjs)
    np.savez("{}.clean.npz".format(hs_prefix), hs2tj=[cleaned_hs2tj])


def main(data_path, n_data):
    tjs_prefix = "raw-trajectories"
    hs2tj_prefix = "hs2tj"

    combined_data_path = []
    for i in range(n_data):
        combined_data_path.append(
            (pjoin(data_path, "{}-{}.npz".format(tjs_prefix, i)),
             pjoin(data_path, "{}-{}.npz".format(hs2tj_prefix, i))))

    threads = [
        Thread(target=clean_data, args=(tp, hs))
        for tp, hs in combined_data_path]

    [t.start() for t in threads]

    [t.join() for t in threads]


if __name__ == '__main__':
    fire.Fire(main)
