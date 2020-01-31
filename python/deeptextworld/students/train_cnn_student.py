import os
from os.path import join as pjoin

import fire
import numpy as np
import tensorflow as tf

from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.students.student_learner import StudentLearner, CMD
from deeptextworld.students.utils import setup_train_log

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class CNNLearner(StudentLearner):
    def train_impl(self, data):
        (p_states, p_len, action_matrix, action_mask_t, action_len,
         expected_qs) = data
        _, summaries = self.sess.run(
            [self.model.train_op, self.model.train_summary_op],
            feed_dict={
                self.model.src_: p_states,
                self.model.src_len_: p_len,
                self.model.actions_mask_: action_mask_t,
                self.model.actions_: action_matrix,
                self.model.actions_len_: action_len,
                self.model.expected_qs_: expected_qs})
        self.sw.add_summary(summaries)
        return

    def prepare_data(self, b_memory, tjs, action_collector):
        trajectory_id = [m.tid for m in b_memory]
        state_id = [m.sid for m in b_memory]
        game_id = [m.gid for m in b_memory]
        action_mask = [m.action_mask for m in b_memory]
        expected_qs = [m.q_actions for m in b_memory]
        action_mask_t = BaseAgent.from_bytes(action_mask)

        states = tjs.fetch_batch_states(trajectory_id, state_id)
        p_states = [self.prepare_trajectory(s) for s in states]
        p_len = [len(state) for state in p_states]

        action_len = (
            [action_collector.get_action_len(gid) for gid in game_id])
        max_action_len = np.max(action_len)
        action_matrix = (
            [action_collector.get_action_matrix(gid)[:, :max_action_len]
             for gid in game_id])

        return (
            p_states, p_len, action_matrix, action_mask_t, action_len,
            expected_qs)


def train(data_path, n_data, model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    setup_train_log(model_path)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.expanduser("~")
    project_path = pjoin(dir_path, "../../..")
    bert_ckpt_dir = pjoin(home_dir, "local/opt/bert-models/bert-model")
    bert_vocab_file = pjoin(bert_ckpt_dir, "vocab.txt")
    nltk_vocab_file = pjoin(project_path, "resources/vocab.txt")

    cmd_args = CMD(
        model_dir=model_path,
        model_creator="CNNEncoderDSQN",
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

    hp = load_hparams_for_training(None, cmd_args)
    learner = CNNLearner(hp, model_path, data_path, n_data)
    learner.train(n_epochs=1000)


if __name__ == "__main__":
    fire.Fire(train)
