import tensorflow as tf

import deeptextworld.models.utils as dqn
from deeptextworld.models.drrn_model import CnnDRRN
from deeptextworld.models.encoders import TxEncoder
from deeptextworld.models.export_models import DSQNModel


class CnnDSQN(CnnDRRN):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
          actions_mask: a 0-1 vector of size |actions|, using 0 to eliminate
                        some actions for a certain state.
        :param hp:
        :param is_infer:
        """
        super(CnnDSQN, self).__init__(hp, is_infer)
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_seg": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(
                tf.int32, [None, self.n_tokens_per_action]),
            "actions_repeats": tf.placeholder(tf.int32, [None]),
            "actions_len": tf.placeholder(tf.float32, [None]),
            "snn_src": tf.placeholder(tf.int32, [None, None]),
            "snn_src_len": tf.placeholder(tf.float32, [None]),
            "snn_src2": tf.placeholder(tf.int32, [None, None]),
            "snn_src2_len": tf.placeholder(tf.float32, [None]),
            "labels": tf.placeholder(tf.float32, [None])
        }

        self.w_snn = tf.layers.Dense(
            units=1, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def is_semantic_same(self):
        h_state = self.enc_tj(self.inputs["snn_src"])
        h_state2 = self.enc_tj(self.inputs["snn_src2"])
        h_states_diff = tf.abs(h_state - h_state2)
        semantic_same = self.w_snn(h_states_diff)
        return semantic_same, h_states_diff

    def get_snn_train_op(self, semantic_same):
        labels = self.inputs["labels"]
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=semantic_same)
        loss = tf.reduce_mean(losses)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op

    def get_merged_train_op(self, loss, snn_loss):
        # Multi-Task Learning Using Uncertainty to Weigh Losses
        s1 = tf.get_variable("s1", shape=[], dtype=tf.float32)
        s2 = tf.get_variable("s2", shape=[], dtype=tf.float32)
        weighted_loss = (
            0.5 * tf.exp(-s1) * loss + tf.exp(-s2) * snn_loss +
            0.5 * s1 + 0.5 * s2)
        merged_train_op = self.optimizer.minimize(
            weighted_loss, global_step=self.global_step)
        return weighted_loss, merged_train_op, s1, s2

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        return create_train_model(cls, hp, device_placement)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        return create_eval_model(cls, hp, device_placement)


class TransformerDSQN(CnnDSQN):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
          actions_mask: a 0-1 vector of size |actions|, using 0 to eliminate
                        some actions for a certain state.
        :param hp:
        :param is_infer:
        """
        super(TransformerDSQN, self).__init__(hp, is_infer)
        self.enc_tj = TxEncoder(
            num_layers=1, d_model=128, num_heads=8, dff=256,
            input_vocab_size=self.hp.vocab_size)

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state
        and hidden actions
        :return:
        """
        _, pooled = self.enc_tj(
            self.inputs["src"], training=(not self.is_infer))
        h_state = self.wt(pooled)
        h_state_expanded = dqn.repeat(h_state, self.inputs["actions_repeats"])
        _, h_actions = self.enc_actions(self.inputs["actions"])
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions[0]), axis=-1)
        return q_actions


class TransformerDSQNWithFactor(TransformerDSQN):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
          actions_mask: a 0-1 vector of size |actions|, using 0 to eliminate
                        some actions for a certain state.
        :param hp:
        :param is_infer:
        """
        super(TransformerDSQNWithFactor, self).__init__(hp, is_infer)
        # trajectory pooler
        self.wt_var = tf.layers.Dense(
            units=self.h_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        _, pooled = self.enc_tj(self.inputs["src"])
        h_state = self.wt(pooled)
        h_state_var = self.wt_var(pooled)
        h_state_sum = h_state + h_state_var
        _, h_actions = self.enc_actions(self.inputs["actions"])
        h_state_expanded = dqn.repeat(
            h_state_sum, self.inputs["actions_repeats"])
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions[0]), axis=-1)
        return q_actions

    def is_semantic_same(self):
        _, pooled = self.enc_tj(self.inputs["snn_src"])
        h_state = self.wt(pooled)
        _, pooled2 = self.enc_tj(self.inputs["snn_src2"])
        h_state2 = self.wt(pooled2)
        h_states_diff = tf.abs(h_state - h_state2)
        semantic_same = self.w_snn(h_states_diff)
        return semantic_same, h_states_diff


def create_train_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            inputs = model.inputs
            q_actions = model.get_q_actions()
            semantic_same, h_states_diff = model.is_semantic_same()
            loss, train_op, abs_loss = model.get_train_op(q_actions)
            snn_loss, snn_train_op = model.get_snn_train_op(semantic_same)
            weighted_loss, merged_train_op, s1, s2 = model.get_merged_train_op(
                loss, snn_loss)
            loss_summary = tf.summary.scalar("loss", loss)
            snn_loss_summary = tf.summary.scalar("snn_loss", snn_loss)
            weighted_loss_summary = tf.summary.scalar(
                "weighted_loss", weighted_loss)
            s1_summary = tf.summary.scalar("w_dqn", 0.5 * tf.exp(-s1))
            s2_summary = tf.summary.scalar("w_snn", tf.exp(-s2))
            train_summary_op = tf.summary.merge([loss_summary])
            snn_train_summary_op = tf.summary.merge([snn_loss_summary])
            weighted_train_summary_op = tf.summary.merge(
                [loss_summary, snn_loss_summary, weighted_loss_summary,
                 s1_summary, s2_summary])
    return DSQNModel(
        graph=graph, q_actions=q_actions, semantic_same=semantic_same,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        snn_src_=inputs["snn_src"],
        snn_src_len_=inputs["snn_src_len"],
        snn_src2_=inputs["snn_src2"],
        snn_src2_len_=inputs["snn_src2_len"],
        actions_repeats_=inputs["actions_repeats"],
        src_seg_=inputs["src_seg"],
        labels_=inputs["labels"],
        b_weight_=inputs["b_weight"],
        abs_loss=abs_loss,
        train_op=train_op, action_idx_=inputs["action_idx"],
        expected_q_=inputs["expected_q"], loss=loss,
        snn_train_op=snn_train_op,
        weighted_loss=weighted_loss,
        snn_loss=snn_loss,
        merged_train_op=merged_train_op,
        train_summary_op=train_summary_op,
        snn_train_summary_op=snn_train_summary_op,
        weighted_train_summary_op=weighted_train_summary_op,
        h_states_diff=h_states_diff,
        h_state=None)


def create_eval_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp, is_infer=True)
            inputs = model.inputs
            q_actions = model.get_q_actions()
            semantic_same, h_states_diff = model.is_semantic_same()
    return DSQNModel(
        graph=graph, q_actions=q_actions, semantic_same=semantic_same,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        snn_src_=inputs["snn_src"],
        snn_src_len_=inputs["snn_src_len"],
        snn_src2_=inputs["snn_src2"],
        snn_src2_len_=inputs["snn_src2_len"],
        actions_repeats_=inputs["actions_repeats"],
        src_seg_=inputs["src_seg"],
        labels_=inputs["labels"],
        b_weight_=inputs["b_weight"],
        abs_loss=None,
        train_op=None, action_idx_=inputs["action_idx"],
        expected_q_=inputs["expected_q"], loss=None,
        snn_train_op=None,
        weighted_loss=None,
        snn_loss=None,
        merged_train_op=None,
        train_summary_op=None,
        snn_train_summary_op=None,
        weighted_train_summary_op=None,
        h_states_diff=h_states_diff,
        h_state=None)
