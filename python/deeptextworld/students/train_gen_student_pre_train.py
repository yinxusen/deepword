import os
from os.path import join as pjoin

import fire
import numpy as np
import tensorflow as tf

from deeptextworld.agents.base_agent import BaseAgent
from deeptextworld.hparams import load_hparams_for_training
from deeptextworld.students.student_learner import StudentLearner, CMD
from deeptextworld.students.utils import get_action_idx_pair
from deeptextworld.students.utils import setup_train_log

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


class GenPreTrainLearner(StudentLearner):
    def train_impl(self, data):
        (p_states, p_len, actions_in, actions_out, action_len,
         expected_qs, b_weights) = data
        _, summaries = self.sess.run(
            [self.model.train_seq2seq_op, self.model.train_seq2seq_summary_op],
            feed_dict={self.model.src_: p_states,
                       self.model.src_len_: p_len,
                       self.model.action_idx_: actions_in,
                       self.model.action_idx_out_: actions_out,
                       self.model.action_len_: action_len,
                       self.model.b_weight_: b_weights})
        self.sw.add_summary(summaries)
        return

    def prepare_data(self, b_memory, tjs, action_collector):
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
        # mask_idx = list(map(lambda m: np.where(m == 1)[0], action_mask_t))
        selected_mask_idx = list(map(
            lambda m: np.random.choice(np.where(m == 1)[0], size=[2, ]),
            action_mask_t))

        states = tjs.fetch_batch_states(trajectory_id, state_id)
        p_states = [self.prepare_trajectory(s) for s in states]
        p_len = [len(state) for state in p_states]

        action_len = np.concatenate(
            [action_collector.get_action_len(gid)[mid]
             for gid, mid in zip(game_id, selected_mask_idx)], axis=0)
        actions = np.concatenate(
            [action_collector.get_action_matrix(gid)[mid, :]
             for gid, mid in zip(game_id, selected_mask_idx)], axis=0)
        actions_in, actions_out, action_len = get_action_idx_pair(
            actions, action_len, self.tokenizer.vocab["<S>"],
            self.tokenizer.vocab["</S>"])
        # repeats = np.sum(action_mask_t, axis=1)
        repeats = 2
        repeated_p_states = np.repeat(p_states, repeats, axis=0)
        repeated_p_len = np.repeat(p_len, repeats, axis=0)
        expected_qs = np.concatenate(
            [qs[mid] for qs, mid in zip(expected_qs, selected_mask_idx)],
            axis=0)
        b_weights = np.ones_like(action_len, dtype="float32")
        return (
            repeated_p_states, repeated_p_len,
            actions_in, actions_out, action_len, expected_qs, b_weights)


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
        model_creator="AttnEncoderDecoderDQN",
        vocab_file=nltk_vocab_file,
        num_tokens=512,
        num_turns=6,
        batch_size=32,
        save_gap_t=50000,
        embedding_size=64,
        learning_rate=5e-5,
        tokenizer_type="NLTK",
        max_snapshot_to_keep=100,
        eval_episode=5,
        game_episode_terminal_t=100,
        replay_mem=500000,
        collect_floor_plan=True
    )

    hp = load_hparams_for_training(None, cmd_args)
    learner = GenPreTrainLearner(hp, model_path, data_path, n_data)
    learner.train(n_epochs=1000)


if __name__ == "__main__":
    fire.Fire(train)
