import os
import random
import time
from os.path import join as pjoin
from queue import Queue
from threading import Thread

import fire
import numpy as np
import tensorflow as tf
from numpy.random import choice as npc
from tqdm import trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent, DRRNMemoTeacher
from deeptextworld import dsqn_model
from deeptextworld.dsqn_model import create_train_student_dsqn_model, create_train_student_drrn_model
from deeptextworld.hparams import load_hparams_for_training, output_hparams, \
    load_hparams_for_evaluation, save_hparams
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import flatten, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def prepare_model(fn_create_model, hp, device_placement, load_model_from):
    model_clazz = getattr(dsqn_model, hp.model_creator)
    model = fn_create_model(
        model_creator=model_clazz, hp=hp,
        device_placement=device_placement)
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


def train_bert_student(
        hp, tokenizer, model_path, combined_data_path, combined_dev_data_path):

    load_dsqn_from = pjoin(model_path, "dsqn_last_weights")
    load_drrn_from = pjoin(model_path, "drrn_last_weights")
    ckpt_dsqn_prefix = pjoin(load_dsqn_from, "after-epoch")
    ckpt_drrn_prefix = pjoin(load_drrn_from, "after-epoch")

    # load dsqn model
    sess_dsqn, model_dsqn, saver_dsqn, train_steps_dsqn = prepare_model(
        create_train_student_dsqn_model, hp, "/device:GPU:0", load_dsqn_from)
    # load drrn model
    sess_drrn, model_drrn, saver_drrn, train_steps_drrn = prepare_model(
        create_train_student_drrn_model, hp, "/device:GPU:1", load_drrn_from)

    # save the very first model to verify the Bert weight has been loaded
    if train_steps_drrn == 0:
        saver_drrn.save(
            sess_drrn, ckpt_drrn_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model_drrn.graph))
        saver_dsqn.save(
            sess_dsqn, ckpt_dsqn_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model_dsqn.graph))
    else:
        pass

    sw_path_dsqn = pjoin(model_path, "dsqn_summaries", "train")
    sw_path_drrn = pjoin(model_path, "drrn_summaries", "train")
    sw_dsqn = tf.summary.FileWriter(sw_path_dsqn, sess_dsqn.graph)
    sw_drrn = tf.summary.FileWriter(sw_path_drrn, sess_drrn.graph)

    queue = Queue(maxsize=100)

    batch_size = 32
    epoch_size = 10000
    num_epochs = 100

    t = Thread(
        target=add_batch,
        args=(combined_data_path, queue, hp, batch_size, tokenizer))
    t.setDaemon(True)
    t.start()

    wait_times = 10
    while wait_times > 0 and queue.empty():
        eprint("waiting data ... (retry times: {})".format(wait_times))
        time.sleep(10)
        wait_times -= 1

    eprint("start training")
    data_in_queue = True
    for et in trange(num_epochs, ascii=True, desc="epoch"):
        for it in trange(epoch_size, ascii=True, desc="step"):
            try:
                data = queue.get(timeout=1)
                (p_states, p_len, action_matrix, action_mask_t, action_len,
                 expected_qs) = data[0]
                (src, src_len, src2, src2_len, labels) = data[1]

                def run_dsqn():
                    _, summ = sess_dsqn.run(
                        [model_dsqn.train_op, model_dsqn.train_summary_op],
                        feed_dict={
                            model_dsqn.src_: p_states,
                            model_dsqn.src_len_: p_len,
                            model_dsqn.actions_mask_: action_mask_t,
                            model_dsqn.actions_: action_matrix,
                            model_dsqn.actions_len_: action_len,
                            model_dsqn.expected_qs_: expected_qs,
                            model_dsqn.snn_src_: src,
                            model_dsqn.snn_src_len_: src_len,
                            model_dsqn.snn_src2_: src2,
                            model_dsqn.snn_src2_len_: src2_len,
                            model_dsqn.labels_: labels})
                    sw_dsqn.add_summary(
                        summ, train_steps_dsqn + et * epoch_size + it)

                t_dsqn = Thread(target=run_dsqn)
                t_dsqn.start()

                # model 2
                _, summaries = sess_drrn.run(
                    [model_drrn.train_op, model_drrn.train_summary_op],
                    feed_dict={
                        model_drrn.src_: p_states,
                        model_drrn.src_len_: p_len,
                        model_drrn.actions_mask_: action_mask_t,
                        model_drrn.actions_: action_matrix,
                        model_drrn.actions_len_: action_len,
                        model_drrn.expected_qs_: expected_qs})
                sw_drrn.add_summary(
                    summaries, train_steps_drrn + et * epoch_size + it)
                t_dsqn.join()
            except Exception as e:
                data_in_queue = False
                eprint("no more data: {}".format(e))
                break
        saver_dsqn.save(
            sess_dsqn, ckpt_dsqn_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model_dsqn.graph))
        saver_drrn.save(
            sess_drrn, ckpt_drrn_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model_drrn.graph))
        eprint("finish and save {} epoch".format(et))
        if not data_in_queue:
            break
    return


def clean_hs2tj(hash_states2tjs, tjs):
    cnt_trashed = 0
    empty_keys = []
    all_tids = list(tjs.trajectories.keys())
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


def add_batch(
        combined_data_path, queue, hp, batch_size, tokenizer):
    while True:
        for tp, ap, mp, hs in sorted(
                combined_data_path, key=lambda k: random.random()):
            memory, tjs, action_collector, hash_states2tjs = load_snapshot(
                hp, mp, tp, ap, hs, tokenizer)
            # clean hs2tj
            hash_states2tjs = clean_hs2tj(hash_states2tjs, tjs)
            random.shuffle(memory)
            i = 0
            while i < len(memory) // batch_size:
                batch_memory = (
                    memory[i*batch_size: min((i+1)*batch_size, len(memory))])
                queue.put(
                    [
                        prepare_data(
                            batch_memory, tjs, action_collector, tokenizer,
                            hp.num_tokens),
                        prepare_snn_pairs(
                            batch_size // 2, hash_states2tjs, tjs,
                            hp.num_tokens, tokenizer)
                    ])
                i += 1


def load_snapshot(
        hp, memo_path, raw_tjs_path, action_path, hs2tj_path, tokenizer):
    memory = np.load(memo_path)['data']
    memory = list(filter(lambda x: isinstance(x, DRRNMemoTeacher), memory))

    tjs = RawTextTrajectory(hp)
    tjs.load_tjs(raw_tjs_path)

    actions = ActionCollector(
        tokenizer, hp.n_actions, hp.n_tokens_per_action,
        unk_val_id=hp.unk_val_id, padding_val_id=hp.padding_val_id)
    actions.load_actions(action_path)

    hs2tj = np.load(hs2tj_path)
    hash_states2tjs = hs2tj["hs2tj"][0]

    eprint(
        "snapshot data loaded:\nmemory path: {}\ntjs path:: {}\n"
        "action path: {}\nhs2tj path: {}".format(
            memo_path, raw_tjs_path, action_path, hs2tj_path))

    return memory, tjs, actions, hash_states2tjs


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


def prepare_snn_pairs(batch_size, hash_states2tjs, tjs, num_tokens, tokenizer):
    non_empty_keys = list(
        filter(lambda x: hash_states2tjs[x] != {},
               hash_states2tjs.keys()))
    hs_keys = npc(non_empty_keys, size=batch_size)
    diff_keys = [
        npc(list(filter(lambda x: x != k, non_empty_keys)), size=None)
        for k in hs_keys]

    target_tids = []
    same_tids = []
    for k in hs_keys:
        try:
            tid_pair = npc(
                list(hash_states2tjs[k].keys()), size=2, replace=False)
        except ValueError:
            tid_pair = list(hash_states2tjs[k].keys()) * 2

        target_tids.append(tid_pair[0])
        same_tids.append(tid_pair[1])

    diff_tids = [npc(list(hash_states2tjs[k])) for k in diff_keys]

    target_sids = [npc(list(hash_states2tjs[k][tid]))
                   for k, tid in zip(hs_keys, target_tids)]
    same_sids = [npc(list(hash_states2tjs[k][tid]))
                 for k, tid in zip(hs_keys, same_tids)]
    diff_sids = [npc(list(hash_states2tjs[k][tid]))
                 for k, tid in zip(diff_keys, diff_tids)]

    target_state = tjs.fetch_batch_states(target_tids, target_sids)
    same_state = tjs.fetch_batch_states(same_tids, same_sids)
    diff_state = tjs.fetch_batch_states(diff_tids, diff_sids)

    target_src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in target_state]
    target_src = [x[0] for x in target_src_n_len]
    target_src_len = [x[1] for x in target_src_n_len]

    same_src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in same_state]
    same_src = [x[0] for x in same_src_n_len]
    same_src_len = [x[1] for x in same_src_n_len]

    diff_src_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in diff_state]
    diff_src = [x[0] for x in diff_src_n_len]
    diff_src_len = [x[1] for x in diff_src_n_len]

    src = np.concatenate([target_src, target_src], axis=0)
    src_len = np.concatenate([target_src_len, target_src_len], axis=0)
    src2 = np.concatenate([same_src, diff_src], axis=0)
    src2_len = np.concatenate([same_src_len, diff_src_len], axis=0)
    labels = np.concatenate([np.zeros(batch_size), np.ones(batch_size)], axis=0)
    return src, src_len, src2, src2_len, labels


def prepare_data(b_memory, tjs, action_collector, tokenizer, num_tokens):
    trajectory_id = [m.tid for m in b_memory]
    state_id = [m.sid for m in b_memory]
    game_id = [m.gid for m in b_memory]
    action_mask = [m.action_mask for m in b_memory]
    expected_qs = [m.q_actions for m in b_memory]
    action_mask_t = BaseAgent.from_bytes(action_mask)

    states = tjs.fetch_batch_states(trajectory_id, state_id)
    states_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    p_states = list(map(lambda x: x[0], states_n_len))
    p_len = list(map(lambda x: x[1], states_n_len))

    action_len = (
        [action_collector.get_action_len(gid) for gid in game_id])
    max_action_len = np.max(action_len)
    action_matrix = (
        [action_collector.get_action_matrix(gid)[:, :max_action_len]
         for gid in game_id])

    return (
        p_states, p_len, action_matrix, action_mask_t, action_len, expected_qs)


def train(cmd_args, combined_data_path, model_path, combined_dev_data_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    hp = load_hparams_for_training(None, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    save_hparams(hp, pjoin(model_path, "hparams.json"))
    train_bert_student(
        hp, tokenizer, model_path, combined_data_path, combined_dev_data_path)


# def evaluate(cmd_args, combined_data_path, model_path):
#     config_file = pjoin(model_path, "hparams.json")
#     hp = load_hparams_for_evaluation(config_file, cmd_args)
#     hp, tokenizer = BaseAgent.init_tokens(hp)
#     last_weights = os.path.join(model_path, "last_weights")
#     eprint(output_hparams(hp))
#     for tp, ap, mp in combined_data_path:
#         eval_gen_student(
#             hp, tokenizer, mp, tp, ap, last_weights)


def main(data_path, n_data, model_path, dev_data_path):
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
             pjoin(data_path, "{}-{}.npz".format(hs2tj_prefix, i))))

    dev_tp = pjoin(dev_data_path, "{}-0.npz".format(tjs_prefix))
    dev_ap = pjoin(dev_data_path, "{}-0.npz".format(action_prefix))
    dev_mp = pjoin(dev_data_path, "{}-0.npz".format(memo_prefix))
    dev_hs = pjoin(dev_data_path, "{}-0.npz".format(hs2tj_prefix))

    combined_dev_data_path = (dev_mp, dev_tp, dev_ap, dev_hs)

    cmd_args = CMD(
        model_dir=model_path,
        model_creator="BertAttnEncoderDSQN",
        vocab_file=bert_vocab_file,
        bert_ckpt_dir=bert_ckpt_dir,
        num_tokens=511,
        num_turns=6,
        batch_size=32,
        save_gap_t=1000,
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
        max_snapshot_to_keep=100
    )

    train(cmd_args, combined_data_path, model_path, combined_dev_data_path)
    # evaluate(combined_data_path, model_path)


if __name__ == "__main__":
    fire.Fire(main)
