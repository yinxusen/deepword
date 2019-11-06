import os
import random
import time
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
from tqdm import trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent, DRRNMemoTeacher
from deeptextworld.dqn_model import AttnEncoderDecoderDQN
from deeptextworld.dqn_model import create_train_gen_model
from deeptextworld.hparams import load_hparams_for_training, output_hparams
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
    batch_size = 32
    epoch_size = 10000
    num_epochs = max((total_size // batch_size) // epoch_size, 1)

    t = Thread(
        target=add_batch,
        args=(memory, tjs, action_collector, queue, batch_size, tokenizer,
              hp.num_tokens))
    t.setDaemon(True)
    t.start()
    while queue.empty():
        eprint("waiting data ...")
        time.sleep(10)

    eprint("start training")
    data_in_queue = True
    for et in trange(num_epochs, ascii=True, desc="epoch"):
        for it in trange(epoch_size, ascii=True, desc="step"):
            try:
                data = queue.get(timeout=10)
                (p_states, p_len, actions_in, actions_out, action_len,
                 expected_qs, b_weights) = data
                _, summaries, weighted_loss = sess.run(
                    [model.train_op, model.train_summary_op, model.loss],
                    feed_dict={model.src_: p_states,
                               model.src_len_: p_len,
                               model.action_idx_: actions_in,
                               model.action_idx_out_: actions_out,
                               model.action_len_: action_len,
                               model.expected_q_: expected_qs,
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


def add_batch(
        memory, tjs, action_collector, queue, batch_size, tokenizer,
        num_tokens):
    while True:
        random.shuffle(memory)
        i = 0
        while i < len(memory) // batch_size:
            batch_memory = (
                memory[i*batch_size: min((i+1)*batch_size, len(memory))])
            queue.put(prepare_data(
                batch_memory, tjs, action_collector, tokenizer, num_tokens))
            i += 1


def load_snapshot(hp, memo_path, raw_tjs_path, action_path, tokenizer):
    memory = np.load(memo_path)['data']
    memory = list(filter(lambda x: isinstance(x, DRRNMemoTeacher), memory))

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
    max_col_size = action_matrix.shape[1]
    idx1 = np.where(action_len < max_col_size)[0]
    idx2 = np.where(action_len >= max_col_size)[0]
    action_id_out[idx1, action_len] = eos_id
    action_id_out[idx2, -1] = eos_id
    new_action_len = action_len[:]
    new_action_len[idx1] += 1
    return action_id_in, action_id_out, new_action_len


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

    action_len = np.concatenate(
        [action_collector.get_action_len(gid)[action_mask_t]
         for gid in game_id], axis=0)
    max_action_len = np.max(action_len)
    actions = np.concatenate(
        [action_collector.get_action_matrix(gid)[action_mask_t, :max_action_len]
         for gid in game_id], axis=0)
    actions_in, actions_out, action_len = get_action_idx_pair(
        actions, action_len, tokenizer.vocab["<S>"], tokenizer.vocab["</S>"])
    repeats = np.sum(action_mask_t, axis=1)
    repeated_p_states = np.repeat(p_states, repeats, axis=0)
    repeated_p_len = np.repeat(p_len, repeats, axis=0)
    expected_qs = np.concatenate(
        [qs[action_mask_t] for qs in expected_qs], axis=0)
    b_weights = np.ones_like(action_len, dtype="float32")
    return (
        repeated_p_states, repeated_p_len,
        actions_in, actions_out, action_len, expected_qs, b_weights)


if __name__ == "__main__":
    HOME = "/home/rcf-40/xusenyin/"
    MODEL_HOME = HOME + "git-store/experiments-drrn-bak5/agent-drrn-TDRRN-fine-tune-drop-no-theme-w-cookbook-teacher/"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    VOCAB_FILE = dir_path + "../../resources/vocab.txt"
    config_file = MODEL_HOME + "hparams.json"
    tjs_path = MODEL_HOME + "raw-trajectories-0.npz"
    action_path = MODEL_HOME + "actions-0.npz"
    memo_path = MODEL_HOME + "memo-0.npz"
    ckpt_prefix = MODEL_HOME + "gen-student/after-epoch"
    summary_writer_path = MODEL_HOME + "gen-student-summary/"
    load_student_from = MODEL_HOME + "gen-student/"
    cmd_args = CMD(
        model_creator="AttnEncoderDecoderDQN",
        vocab_file=VOCAB_FILE,
        num_tokens=1000,
        num_turns=6,
        batch_size=16,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=5e-5,
        tokenizer_type="NLTK"
    )
    hp = load_hparams_for_training(config_file, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    eprint(output_hparams(hp))
    train_gen_student(
        hp, tokenizer, memo_path, tjs_path, action_path,
        ckpt_prefix, summary_writer_path, load_student_from)
