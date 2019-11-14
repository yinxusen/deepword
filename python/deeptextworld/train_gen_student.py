import os
import sys
import random
import time
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
from tqdm import trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.dqn_func import get_best_2Daction
from deeptextworld.dqn_model import AttnEncoderDecoderDQN
from deeptextworld.dqn_model import create_train_gen_model, \
    create_eval_gen_model
from deeptextworld.hparams import load_hparams_for_training, output_hparams, \
    load_hparams_for_evaluation, save_hparams
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import flatten, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_gen_student(
        hp, tokenizer, memo_path, tjs_path, action_path,
        ckpt_prefix, summary_writer_path, load_student_from):
    model = create_train_gen_model(
        model_creator=AttnEncoderDecoderDQN, hp=hp,
        device_placement="/device:GPU:0")
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    summary_writer = tf.summary.FileWriter(summary_writer_path, sess.graph)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=hp.max_snapshot_to_keep,
            save_relative_paths=True)
        global_step = tf.train.get_or_create_global_step()

    try:
        ckpt_path = tf.train.latest_checkpoint(load_student_from)
        saver.restore(sess, ckpt_path)
        trained_steps = sess.run(global_step)
        eprint("load student from ckpt: {}".format(ckpt_path))
    except Exception as e:
        eprint("load model failed: {}".format(e))
        trained_steps = 0

    queue = Queue(maxsize=100)

    memory, tjs, action_collector = load_snapshot(
        hp, memo_path, tjs_path, action_path, tokenizer)
    total_size = len(memory)
    eprint("loaded memory size: {}".format(total_size))
    batch_size = hp.batch_size
    epoch_size = 5000
    num_epochs = int(max(np.ceil(total_size / batch_size / epoch_size), 1))

    t = Thread(
        target=add_batch,
        args=(memory, tjs, action_collector, queue, batch_size, tokenizer,
              hp.num_tokens))
    t.setDaemon(True)
    t.start()
    wait_cnt = 0
    while wait_cnt < 10 and queue.empty():
        eprint("waiting data ...")
        time.sleep(10)
        wait_cnt += 1

    eprint("start training")
    data_in_queue = True
    for et in trange(num_epochs, ascii=True, desc="epoch"):
        for it in trange(epoch_size, ascii=True, desc="step"):
            try:
                data = queue.get(timeout=10)
                (p_states, p_len, actions_in, actions_out, action_len,
                 expected_qs, b_weights) = data
                _, summaries = sess.run(
                    [model.train_seq2seq_op, model.train_seq2seq_summary_op],
                    feed_dict={model.src_: p_states,
                               model.src_len_: p_len,
                               model.action_idx_: actions_in,
                               model.action_idx_out_: actions_out,
                               model.action_len_: action_len,
                               model.b_weight_: b_weights})
                summary_writer.add_summary(
                    summaries, trained_steps + et * epoch_size + it)
            except Exception as e:
                data_in_queue = False
                eprint("no more data: {}".format(e))
                break
        saver.save(
            sess, ckpt_prefix,
            global_step=tf.train.get_or_create_global_step(
                graph=model.graph))
        eprint("finish and save {} epoch".format(et))
        if not data_in_queue:
            break


def eval_gen_student(
        hp, tokenizer, memo_path, tjs_path, action_path, load_student_from):
    model = create_eval_gen_model(
        model_creator=AttnEncoderDecoderDQN, hp=hp,
        device_placement="/device:GPU:0")
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=hp.max_snapshot_to_keep,
            save_relative_paths=True)

    try:
        ckpt_path = tf.train.latest_checkpoint(load_student_from)
        saver.restore(sess, ckpt_path)
        eprint("load student from ckpt: {}".format(ckpt_path))
    except Exception as e:
        eprint("load model failed: {}".format(e))

    queue = Queue(maxsize=100)

    memory, tjs, action_collector = load_snapshot(
        hp, memo_path, tjs_path, action_path, tokenizer)
    total_size = len(memory)
    eprint("loaded memory size: {}".format(total_size))
    batch_size = hp.batch_size
    epoch_size = max((total_size // batch_size) // 10, 1)

    t = Thread(
        target=add_batch,
        args=(memory, tjs, action_collector, queue, batch_size, tokenizer,
              hp.num_tokens))
    t.setDaemon(True)
    t.start()
    wait_cnt = 0
    while wait_cnt < 10 and queue.empty():
        eprint("waiting data ...")
        time.sleep(10)
        wait_cnt += 1

    eprint("start eval")
    total_correct = 0
    total_data = 0
    for it in trange(epoch_size):
        try:
            data = queue.get(timeout=10)
            (p_states, p_len, actions_in, actions_out, action_len,
             expected_qs, b_weights) = data
            res = sess.run(
                [model.q_actions_infer, model.p_gen_infer,
                 model.gen_dist, model.copy_dist],
                feed_dict={
                    model.src_: p_states,
                    model.src_len_: p_len,
                    model.temperature: 1.})
            q_actions = res[0]
            p_gen = res[1]
            gen_dist = res[2]
            eprint("total prob")
            eprint(gen_dist[0][np.where(gen_dist[0] > 1e-3)])
            eprint(np.sum(gen_dist[0], axis=1))
            copy_dist = res[3]
            eprint("copy prob")
            eprint(copy_dist[0][np.where(copy_dist[0] != 0)])
            eprint(np.sum(copy_dist[0], axis=1))
        except Exception as e:
            eprint("no more data: {}".format(e))
            break
        decoded = list(map(
            lambda qa: get_best_2Daction(qa, tokenizer.inv_vocab, hp.eos_id),
            list(q_actions)))
        decoded_action = list(map(lambda x: x[-1], decoded))
        decoded_action_str = [
            " ".join(["{}[{:.2f}]".format(a, p)
                      for a, p in zip(da.split(), list(np.squeeze(pg)))])
            for da, pg in zip(decoded_action, list(p_gen))]
        true_action = [
            q_idx_to_action(q_idx, l, tokenizer.inv_vocab)
            for q_idx, l in zip(list(actions_out), list(action_len))]
        eprint("\n")
        eprint("\n".join(
            ["{} - {}".format(t, d)
             for t, d in zip(true_action, decoded_action_str)]))
        eprint("-----------")
        correct = np.sum(
            [a1 == a2 for a1, a2 in
             zip(true_action, decoded_action)])
        total_correct += correct
        total_data += len(decoded)
    eprint("total data: {}, total correct: {}, ratio: {:.2f}".format(
        total_data, total_correct, total_correct / total_data))


def q_idx_to_action(q_idx, valid_len, tokens):
    """
    :param q_idx:
    :param valid_len: length includes </S>
    :param tokens:
    :return:
    """
    if valid_len <= 1:
        return " "
    return " ".join(map(lambda t: tokens[t], q_idx[:valid_len-1]))


def add_batch(
        memory, tjs, action_collector, queue, batch_size, tokenizer,
        num_tokens):
    while True:
        random.shuffle(memory)
        i = 0
        while i < len(memory) // batch_size:
            batch_memory = (
                memory[i*batch_size: min((i+1)*batch_size, len(memory))])
            queue.put(prepare_data_v2(
                batch_memory, tjs, action_collector, tokenizer, num_tokens))
            i += 1


def load_snapshot(hp, memo_path, raw_tjs_path, action_path, tokenizer):
    memory = np.load(memo_path)['data']
    memory = list(filter(lambda x: isinstance(x, tuple), memory))

    tjs = RawTextTrajectory(hp)
    tjs.load_tjs(raw_tjs_path)

    actions = ActionCollector(
        tokenizer, hp.n_actions, hp.n_tokens_per_action,
        unk_val_id=hp.unk_val_id, padding_val_id=hp.padding_val_id)
    actions.load_actions(action_path)
    return memory, tjs, actions


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


def get_action_idx_pair(action_matrix, action_len, sos_id, eos_id):
    action_id_in = np.concatenate(
        [np.asarray([[sos_id]] * len(action_len)),
         action_matrix[:, :-1]], axis=1)
    action_id_out = action_matrix[:, :]
    n_rows, max_col_size = action_matrix.shape
    new_action_len = np.min(
        [action_len + 1, np.zeros_like(action_len) + max_col_size], axis=0)
    action_id_out[list(range(n_rows)), new_action_len-1] = eos_id
    return action_id_in, action_id_out, new_action_len


def get_q_per_token(action_idx, expected_qs, vocab_size):
    """
    :param action_idx: action idx after masking
    :param expected_qs: expected_qs after masking
    :return:
    """
    n_cols = action_idx.shape[1]
    n_rows = vocab_size
    n_actions = action_idx.shape[0]
    expected_q_mat = np.full([n_rows, n_cols], fill_value=-np.inf)
    for i in range(n_cols):
        for k in n_actions:
            if expected_qs[k] > expected_q_mat[action_idx[k, i], i]:
                expected_q_mat[action_idx[k, i], i] = expected_qs[k]
            else:
                pass
    return expected_q_mat


def get_best_q_idx(expected_qs, mask_idx):
    best_q_idx = []
    for i, mk_id in enumerate(mask_idx):
        bq_idx = mk_id[np.argmax(expected_qs[i][mk_id])]
        best_q_idx.append(bq_idx)
    return best_q_idx


def prepare_data(b_memory, tjs, action_collector, tokenizer, num_tokens):
    """
    ("tid", "sid", "gid", "aid", "reward", "is_terminal",
     "action_mask", "next_action_mask", "q_actions")
    """
    trajectory_id = [m[0] for m in b_memory]
    state_id = [m[1] for m in b_memory]
    game_id = [m[2] for m in b_memory]
    action_mask = [m[6] for m in b_memory]
    expected_qs = [m[8] for m in b_memory]
    action_mask_t = list(BaseAgent.from_bytes(action_mask))
    mask_idx = list(map(lambda m: np.where(m == 1)[0], action_mask_t))

    states = tjs.fetch_batch_states(trajectory_id, state_id)
    states_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    p_states = list(map(lambda x: x[0], states_n_len))
    p_len = list(map(lambda x: x[1], states_n_len))

    action_len = np.concatenate(
        [action_collector.get_action_len(gid)[mid]
         for gid, mid in zip(game_id, mask_idx)], axis=0)
    actions = np.concatenate(
        [action_collector.get_action_matrix(gid)[mid, :]
         for gid, mid in zip(game_id, mask_idx)], axis=0)
    actions_in, actions_out, action_len = get_action_idx_pair(
        actions, action_len, tokenizer.vocab["<S>"], tokenizer.vocab["</S>"])
    repeats = np.sum(action_mask_t, axis=1)
    repeated_p_states = np.repeat(p_states, repeats, axis=0)
    repeated_p_len = np.repeat(p_len, repeats, axis=0)
    expected_qs = np.concatenate(
        [qs[mid] for qs, mid in zip(expected_qs, mask_idx)], axis=0)
    b_weights = np.ones_like(action_len, dtype="float32")
    return (
        repeated_p_states, repeated_p_len,
        actions_in, actions_out, action_len, expected_qs, b_weights)


def prepare_data_v2(b_memory, tjs, action_collector, tokenizer, num_tokens):
    """
    ("tid", "sid", "gid", "aid", "reward", "is_terminal",
     "action_mask", "next_action_mask", "q_actions")
    """
    trajectory_id = [m[0] for m in b_memory]
    state_id = [m[1] for m in b_memory]
    game_id = [m[2] for m in b_memory]
    action_mask = [m[6] for m in b_memory]
    expected_qs = [m[8] for m in b_memory]
    action_mask_t = list(BaseAgent.from_bytes(action_mask))
    mask_idx = list(map(lambda m: np.where(m == 1)[0], action_mask_t))

    best_q_idx = get_best_q_idx(expected_qs, mask_idx)

    states = tjs.fetch_batch_states(trajectory_id, state_id)
    states_n_len = [
        prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    p_states = np.asarray(list(map(lambda x: x[0], states_n_len)))
    p_len = np.asarray(list(map(lambda x: x[1], states_n_len)))

    action_len = np.asarray(
        [action_collector.get_action_len(gid)[mid]
         for gid, mid in zip(game_id, best_q_idx)])
    actions = np.asarray(
        [action_collector.get_action_matrix(gid)[mid, :]
         for gid, mid in zip(game_id, best_q_idx)])
    actions_in, actions_out, action_len = get_action_idx_pair(
        actions, action_len, tokenizer.vocab["<S>"], tokenizer.vocab["</S>"])
    b_weights = np.ones_like(action_len, dtype="float32")
    return (
        p_states, p_len,
        actions_in, actions_out, action_len, expected_qs, b_weights)


def train(combined_data_path, model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    hp = load_hparams_for_training(None, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    last_weights = os.path.join(model_path, "last_weights")
    ckpt_prefix = os.path.join(last_weights, "after-epoch")
    summary_writer_path = os.path.join(model_path, "summaries")

    save_hparams(hp, "{}/hparams.json".format(model_path))
    for tp, ap, mp in combined_data_path:
        train_gen_student(
            hp, tokenizer, mp, tp, ap,
            ckpt_prefix, summary_writer_path, last_weights)


def evaluate(combined_data_path, model_path):
    hp = load_hparams_for_evaluation(config_file, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    last_weights = os.path.join(model_path, "last_weights")
    eprint(output_hparams(hp))
    for tp, ap, mp in combined_data_path:
        eval_gen_student(
            hp, tokenizer, mp, tp, ap, last_weights)


if __name__ == "__main__":
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    n_data = int(sys.argv[3])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    VOCAB_FILE = dir_path + "/../../resources/vocab.txt"
    config_file = os.path.join(model_path, "hparams.json")

    tjs_prefix = "raw-trajectories"
    action_prefix = "actions"
    memo_prefix = "memo"

    combined_data_path = []
    for i in range(n_data):
        combined_data_path.append(
            (os.path.join(data_path, "{}-{}.npz".format(tjs_prefix, i)),
             os.path.join(data_path, "{}-{}.npz".format(action_prefix, i)),
             os.path.join(data_path, "{}-{}.npz".format(memo_prefix, i))))

    cmd_args = CMD(
        model_dir=model_path,
        model_creator="AttnEncoderDecoderDQN",
        vocab_file=VOCAB_FILE,
        num_tokens=512,
        num_turns=6,
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=5e-5,
        tokenizer_type="NLTK"
    )
    train(combined_data_path, model_path)
