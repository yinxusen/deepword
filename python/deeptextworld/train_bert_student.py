import os
import collections
import random
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
from tqdm import tqdm, trange

from deeptextworld.action import ActionCollector
from deeptextworld.agents.base_agent import BaseAgent, DRRNMemo
from deeptextworld.dsqn_model import BertAttnEncoderDSQN
from deeptextworld.dsqn_model import create_train_student_drrn_model
from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.trajectory import RawTextTrajectory
from deeptextworld.utils import flatten

VOCAB_FILE = "/Users/xusenyin/local/opt/bert-models/bert-model/vocab.txt"


class CMD:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_bert_student(hp, tokenizer, memo_path, tjs_path, action_path):
    model = create_train_student_drrn_model(
        model_creator=BertAttnEncoderDSQN, hp=hp,
        device_placement="/device:GPU:0")
    conf = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(graph=model.graph, config=conf)
    summary_writer = tf.summary.FileWriter(
        '/tmp/tf-writer', sess.graph)
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            max_to_keep=hp.max_snapshot_to_keep,
            save_relative_paths=True)
    queue = Queue(maxsize=100)

    memory, tjs, action_collector = load_snapshot(
        hp, memo_path, tjs_path, action_path, tokenizer)
    batch_size = 32
    epoch_size = 10000
    ckpt_prefix = "/tmp/bert-drrn"

    t = Thread(
        target=add_batch,
        args=(memory, tjs, action_collector, queue, batch_size, tokenizer,
              hp.num_tokens))
    t.start()

    print("start training")
    pbar = tqdm(total=epoch_size)
    data_in_queue = True
    step_t = 0
    while data_in_queue:
        try:
            data = queue.get(timeout=10)
            p_states, p_len, action_matrix, action_mask_t, action_len, expected_qs = data
            _, summaries, weighted_loss = sess.run(
                [model.train_op, model.train_summary_op, model.loss],
                feed_dict={model.src_: p_states,
                           model.src_len_: p_len,
                           model.actions_mask_: action_mask_t,
                           model.actions_: action_matrix,
                           model.actions_len_: action_len,
                           model.expected_qs: expected_qs})
            step_t += 1
            pbar.update(step_t)
            summary_writer.add_summary(summaries, step_t)
            if step_t >= epoch_size:
                step_t = 0
                saver.save(
                    sess, ckpt_prefix,
                    global_step=tf.train.get_or_create_global_step(
                        graph=model.graph))
        except Exception as e:
            data_in_queue = False
            print("no more data: {}".format(e))

    print("wait to join")
    t.join(timeout=10)
    return


def add_batch(memory, tjs, action_collector, queue, batch_size, tokenizer, num_tokens):
    while True:
        random.shuffle(memory)
        i = 0
        while i < len(memory) // batch_size:
            batch_memory = memory[i*batch_size: min((i+1)*batch_size, len(memory))]
            queue.put(prepare_data(batch_memory, tjs, action_collector, tokenizer, num_tokens))
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
    if len(tokens) > 10:
        return tokens[:10]
    else:
        return tokens + [0] * len(tokens)


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
    states_n_len = [prepare_trajectory(s, tokenizer, num_tokens) for s in states]
    p_states = list(map(lambda x: x[0], states_n_len))
    p_len = list(map(lambda x: x[1], states_n_len))

    action_len = (
        [action_collector.get_action_len(gid) for gid in game_id])
    max_action_len = np.max(action_len)
    action_matrix = (
        [action_collector.get_action_matrix(gid)[:, :max_action_len]
         for gid in game_id])

    return p_states, p_len, action_matrix, action_mask_t, action_len, expected_qs


if __name__ == "__main__":
    HOME = "/Users/xusenyin/Downloads/submissions-to-codelab/submission_deepdnd-32/agent-drrn-textworld/"
    bert_ckpt_dir = "/Users/xusenyin/local/opt/bert-models/bert-model"
    config_file = HOME + "hparams.json"
    tjs_path = HOME + "raw-trajectories-99.npz"
    action_path = HOME + "actions-99.npz"
    memo_path = HOME + "memo-99.npz"
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
    train_bert_student(hp, tokenizer, memo_path, tjs_path, action_path)