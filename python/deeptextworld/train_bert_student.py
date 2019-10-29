import os
import random
import time
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
from tqdm import trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent, DRRNMemo
from deeptextworld.dsqn_model import BertAttnEncoderDSQN
from deeptextworld.dsqn_model import create_train_student_drrn_model
from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import flatten, eprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_bert_student(
        hp, tokenizer, memo_path, tjs_path, action_path,
        ckpt_prefix, summary_writer_path, load_student_from):
    model = create_train_student_drrn_model(
        model_creator=BertAttnEncoderDSQN, hp=hp,
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
    batch_size = 32
    epoch_size = 10000
    num_epochs = total_size // epoch_size

    t = Thread(
        target=add_batch,
        args=(memory, tjs, action_collector, queue, batch_size, tokenizer,
              hp.num_tokens))
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
                (p_states, p_len, action_matrix, action_mask_t, action_len,
                 expected_qs) = data
                _, summaries, weighted_loss = sess.run(
                    [model.train_op, model.train_summary_op, model.loss],
                    feed_dict={model.src_: p_states,
                               model.src_len_: p_len,
                               model.actions_mask_: action_mask_t,
                               model.actions_: action_matrix,
                               model.actions_len_: action_len,
                               model.expected_qs: expected_qs})
                summary_writer.add_summary(summaries, et * epoch_size + it)
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

    eprint("wait to join")
    t.join(timeout=10)


def train_bert_student_no_queue(
        hp, tokenizer, memo_path, tjs_path, action_path,
        ckpt_prefix, summary_writer_path):
    model = create_train_student_drrn_model(
        model_creator=BertAttnEncoderDSQN, hp=hp,
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

    memory, tjs, action_collector = load_snapshot(
        hp, memo_path, tjs_path, action_path, tokenizer)
    total_size = len(memory)
    batch_size = 32
    epoch_size = 10000
    num_epochs = total_size // epoch_size

    eprint("start training")
    total_t = 0
    while True:
        random.shuffle(memory)
        for i in trange(len(memory) // batch_size):
            batch_memory = (
                memory[i*batch_size: min((i+1)*batch_size, len(memory))])
            data = prepare_data(
                batch_memory, tjs, action_collector, tokenizer, hp.num_tokens)
            (p_states, p_len, action_matrix, action_mask_t, action_len,
             expected_qs) = data
            _, summaries, weighted_loss = sess.run(
                [model.train_op, model.train_summary_op, model.loss],
                feed_dict={model.src_: p_states,
                           model.src_len_: p_len,
                           model.actions_mask_: action_mask_t,
                           model.actions_: action_matrix,
                           model.actions_len_: action_len,
                           model.expected_qs: expected_qs})
            summary_writer.add_summary(summaries, total_t)
            total_t += 1
            if total_t % epoch_size == 0:
                saver.save(
                    sess, ckpt_prefix,
                    global_step=tf.train.get_or_create_global_step(
                        graph=model.graph))


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
    memory = list(filter(lambda x: isinstance(x, DRRNMemo), memory))

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


if __name__ == "__main__":
    HOME = "/home/rcf-40/xusenyin/"
    MODEL_HOME = HOME + "git-store/experiments-drrn-bak5/agent-drrn-TDRRN-fine-tune-drop-no-theme-w-cookbook-teacher/"
    VOCAB_FILE = HOME + "local/opt/bert-models/bert-model/vocab.txt"
    bert_ckpt_dir = HOME + "local/opt/bert-models/bert-model"
    config_file = MODEL_HOME + "hparams.json"
    tjs_path = MODEL_HOME + "raw-trajectories-0.npz"
    action_path = MODEL_HOME + "actions-0.npz"
    memo_path = MODEL_HOME + "memo-0.npz"
    ckpt_prefix = MODEL_HOME + "bert-student/after-epoch"
    summary_writer_path = MODEL_HOME + "bert-student-summary/"
    load_student_from = MODEL_HOME + "bert-student/"
    cmd_args = CMD(
        model_creator="BertAttnEncoderDSQN",
        vocab_file=VOCAB_FILE,
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
        mask_val_id=0
    )
    hp = load_hparams_for_training(config_file, cmd_args)
    hp, tokenizer = BaseAgent.init_tokens(hp)
    train_bert_student(
        hp, tokenizer, memo_path, tjs_path, action_path,
        ckpt_prefix, summary_writer_path, load_student_from)
