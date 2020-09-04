import glob
import os
import random
import time
from collections import Counter
from os.path import join as pjoin
from queue import Queue
from threading import Thread

import fire
import numpy as np
import tensorflow as tf

from deepword.models import drrn_model
from deepword.action import ActionCollector
from deepword.agents.base_agent import BaseAgent, DRRNMemoTeacher
from deepword.models.drrn_model import create_eval_model
from deepword.hparams import load_hparams, output_hparams
from deepword.trajectory import Trajectory
from deepword.utils import flatten, eprint, load_uniq_lines

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def prepare_model(fn_create_model, hp, device_placement, load_model_from):
    model_clazz = getattr(drrn_model, hp.model_creator)
    model = fn_create_model(
        model_creator=model_clazz, hp=hp)
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=hp.max_snapshot_to_keep,
            save_relative_paths=True)
        global_step = tf.train.get_or_create_global_step()

    try:
        ckpt_path = tf.train.latest_checkpoint(load_model_from)
        saver.restore(sess, ckpt_path)
        trained_steps = sess.run(global_step)
        eprint("load student from ckpt: {}".format(ckpt_path))
    except Exception as e:
        eprint("load model failed: {}".format(e))
        trained_steps = 0
    return sess, model, saver, trained_steps


def fetch_features(
        hp, tokenizer, model_path, combined_data_path, selected_hs_keys):
    load_drrn_from = pjoin(model_path, "best_weights")

    sess_drrn, model_drrn, saver_drrn, train_steps_drrn = prepare_model(
        create_eval_model, hp, "/device:GPU:0", load_drrn_from)

    queue = Queue(maxsize=-1)

    batch_size = hp.batch_size
    epoch_size = hp.save_gap_t
    num_epochs = 1000

    # add_batch(
    #     combined_data_path, queue, hp, batch_size, tokenizer,
    #     selected_hs_keys)

    t = Thread(
        target=add_batch,
        args=(combined_data_path, queue, hp, batch_size, tokenizer, selected_hs_keys))
    t.setDaemon(True)
    t.start()

    wait_times = 10
    while wait_times > 0 and queue.empty():
        eprint("waiting data ... (retry times: {})".format(wait_times))
        time.sleep(10)
        wait_times -= 1

    eprint("start training")
    while True:
        try:
            data = queue.get(timeout=10)
            (src, src_len, labels) = data
            h_state = sess_drrn.run(
                model_drrn.h_state,
                feed_dict={
                    model_drrn.src_: src,
                    model_drrn.src_len_: src_len})
            for features, y in zip(list(h_state), labels):
                eprint("data: {} {}".format(",".join(
                    list(map(lambda x: "{:.5f}".format(x), features))), y))
        except Exception as e:
            eprint("no more data: {}".format(e))
            break
    return


def add_batch(
        combined_data_path, queue, hp, batch_size, tokenizer, selected_hs_keys):
    """
    Fetch batches from data, without random orders of data, for eval purpose.
    fetch all data once, and stop if there is no more data.
    """
    for tp, ap, mp, hs in sorted(combined_data_path):
        memory, tjs, action_collector, lst_hs2tj = load_snapshot(
            hp, mp, tp, ap, hs, tokenizer, selected_hs_keys)
        eprint("len of hs2tj: {}".format(len(lst_hs2tj)))
        keys = [x[0] for x in lst_hs2tj]
        ge50 = []
        c = Counter(keys)
        for k in c:
            if c.get(k) >= 50:
                ge50.append(k)
        eprint(len(ge50))
        lst_hs2tj = list(filter(lambda x: x[0] in ge50, lst_hs2tj))

        i = 0
        while i < len(lst_hs2tj) // batch_size:
            batch_hs2tj = (
                lst_hs2tj[i*batch_size: min((i+1)*batch_size, len(memory))])
            queue.put(
                prepare_data_w_label(
                    batch_hs2tj, tjs, hp.num_tokens, tokenizer),
            )
            i += 1


def hs2tj2list(hs2tj, selected_hs_keys=None):
    lst = []
    for hs_key in hs2tj:
        if (selected_hs_keys is None) or (hs_key in selected_hs_keys):
            for tid in hs2tj[hs_key]:
                for sid in hs2tj[hs_key][tid]:
                    lst.append((hs_key, tid, sid))
        else:
            pass
    return lst


def load_snapshot(
        hp, memo_path, raw_tjs_path, action_path, hs2tj_path,
        tokenizer, selected_hs_keys):
    memory = np.load(memo_path, allow_pickle=True)['data']
    memory = list(filter(lambda x: isinstance(x, DRRNMemoTeacher), memory))

    tjs = Trajectory(hp.num_turns)
    tjs.load_tjs(raw_tjs_path)

    actions = ActionCollector(
        tokenizer, hp.n_actions, hp.n_tokens_per_action,
        unk_val_id=hp.unk_val_id, padding_val_id=hp.padding_val_id)
    actions.load_actions(action_path)

    hs2tj = np.load(hs2tj_path, allow_pickle=True)
    hash_states2tjs = hs2tj["hs2tj"][0]
    lst_hs2tj = hs2tj2list(hash_states2tjs, None)

    eprint(
        "snapshot data loaded:\nmemory path: {}\ntjs path:: {}\n"
        "action path: {}\nhs2tj path: {}".format(
            memo_path, raw_tjs_path, action_path, hs2tj_path))

    return memory, tjs, actions, lst_hs2tj


def prepare_master(master_str, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(master_str))


def prepare_action(action_str, tokenizer):
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(action_str))
    tokens = tokens[:max(10, len(tokens))]
    return tokens


def prepare_trajectory(trajectory_lst, tokenizer, num_tokens):
    rst_lst = flatten(
        [prepare_master(s, tokenizer) if i % 2 == 0 else
         prepare_action(s, tokenizer) for i, s in enumerate(trajectory_lst)])
    len_rst = len(rst_lst)
    if len_rst > num_tokens:
        rst_lst = rst_lst[len_rst-num_tokens:]
    else:
        rst_lst += [0] * (num_tokens - len_rst)
    len_rst = min(len_rst, num_tokens)
    return rst_lst, len_rst


def prepare_data_w_label(b_hs2tj, tjs, num_tokens, tokenizer):
    tid_sids = [(x[1], x[2]) for x in b_hs2tj]
    labels = [x[0] for x in b_hs2tj]

    states = tjs.fetch_batch_states_impl(tid_sids)
    src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    src = [x[0] for x in src_n_len]
    src_len = [x[1] for x in src_n_len]

    return src, src_len, labels


def load_game_files(game_dir, f_games=None):
    """
    Choose games appearing in f_games in a given game dir. Return all games in
    the game dir if f_games is None.
    :param game_dir: a dir
    :param f_games: a file of game names
    :return: a list of games
    """
    if f_games is not None:
        with open(f_games, "r") as f:
            selected_games = map(lambda x: x.strip(), f.readlines())
        game_files = list(map(
            lambda x: os.path.join(game_dir, "{}.ulx".format(x)),
            selected_games))
    else:
        game_files = glob.glob(os.path.join(game_dir, "*.ulx"))
    return game_files


def split_train_dev(game_files):
    """
    Split train/dev sets from given game files
    sort - shuffle w/ Random(42) - 90%/10% split
      - if #game_files < 10, then use the last one as dev set;
      - if #game_files == 1, then use the one as both train and dev.
    :param game_files: game files
    :return: None if game_files is empty, otherwise (train, dev)
    """
    # have to sort first, otherwise after shuffling the result is different
    # on different platforms, e.g. Linux VS MacOS.
    game_files = sorted(game_files)
    random.Random(42).shuffle(game_files)
    if len(game_files) == 0:
        print("no game files found!")
        return None
    elif len(game_files) == 1:
        train_games = game_files
        dev_games = game_files
    elif len(game_files) < 10:  # use the last one as eval
        train_games = game_files[:-1]
        dev_games = game_files[-1:]
    else:
        num_train = int(len(game_files) * 0.9)
        train_games = game_files[:num_train]
        dev_games = game_files[num_train:]
    return train_games, dev_games


def fetch(cmd_args, combined_data_path, model_path, selected_hs_keys_path):
    config_file = pjoin(model_path, "hparams.json")
    hp = load_hparams(config_file, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    selected_hs_keys = load_uniq_lines(selected_hs_keys_path)
    fetch_features(
        hp, tokenizer, model_path, combined_data_path, selected_hs_keys)


def main(data_path, n_data, model_path, selected_hs_keys_path):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.expanduser("~")
    bert_ckpt_dir = pjoin(home_dir, "local/opt/bert-models/bert-model")
    bert_vocab_file = pjoin(bert_ckpt_dir, "vocab.txt")
    nltk_vocab_file = pjoin(dir_path, "../resources/vocab.txt")

    tjs_prefix = "raw-trajectories"
    action_prefix = "actions"
    memo_prefix = "memo"
    hs2tj_prefix = "hs2tj"

    combined_data_path = []
    for i in sorted(range(n_data), key=lambda k: random.random()):
        combined_data_path.append(
            (pjoin(data_path, "{}-{}.npz".format(tjs_prefix, i)),
             pjoin(data_path, "{}-{}.npz".format(action_prefix, i)),
             pjoin(data_path, "{}-{}.npz".format(memo_prefix, i)),
             pjoin(data_path, "{}-{}.clean.npz".format(hs2tj_prefix, i))))

    cmd_args = CMD(
        model_dir=model_path,
        model_creator="AttnEncoderDSQN",
        vocab_file=bert_vocab_file,
        bert_ckpt_dir=bert_ckpt_dir,
        num_tokens=511,
        num_turns=6,
        batch_size=32,
        save_gap_t=5000,
        embedding_size=64,
        learning_rate=5e-5,
        num_conv_filters=32,
        bert_num_hidden_layers=1,
        cls_val="[CLS]",
        cls_val_id=0,
        sep_val="[SEP]",
        sep_val_id=0,
        mask_val="[MASK]",
        mask_val_id=0,
        tokenizer_type="BERT",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True
    )

    fetch(cmd_args, combined_data_path, model_path, selected_hs_keys_path)


if __name__ == "__main__":
    fire.Fire(main)
