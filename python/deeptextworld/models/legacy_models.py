"""
Legacy models are those models trained with deep-textworld v1.0 - v3.0 code

These models are not longer valid for code after v3.0, but sometimes we need
to load the old models to play games, either for comparison, or for teacher
model forcing.
"""

import numpy as np
import tensorflow as tf

import deeptextworld.models.utils as dqn
from deeptextworld.models.dqn_model import BaseDQN
from deeptextworld.models.export_models import DRRNModel


def get_best_1D_q(q_actions_t, mask=None):
    if mask is not None:
        inv_mask = np.logical_not(mask)
        min_q_val = np.min(q_actions_t)
        q_actions_t = q_actions_t * mask + inv_mask * min_q_val
    action_idx = np.argmax(q_actions_t)
    q_val = q_actions_t[action_idx]
    return action_idx, q_val


def get_best_1Daction(q_actions_t, actions, mask=None):
    action_idx, q_val = get_best_1D_q(q_actions_t, mask)
    action = actions[action_idx]
    return action_idx, q_val, action


def l2_loss_1Daction(q_actions, action_idx, expected_q, n_actions, b_weight):
    """
    l2 loss for 1D action space.
    e.g. "go east" would be one whole action.
    :param q_actions: Q-vector of a state for all actions
    :param action_idx: placeholder, the action chose for the state,
           in a format of (tf.int32, [None])
    :param expected_q: placeholder, the expected reward gained from the step,
           in a format of (tf.float32, [None])
    :param n_actions: number of total actions
    :param b_weight:
    """
    actions_mask = tf.one_hot(indices=action_idx, depth=n_actions)
    predicted_q = tf.reduce_sum(
        tf.multiply(q_actions, actions_mask), axis=1)
    loss = tf.reduce_mean(b_weight * tf.square(expected_q - predicted_q))
    abs_loss = tf.abs(expected_q - predicted_q)
    return loss, abs_loss


def decoder_dense_classification(inner_states, n_actions):
    """
    :param inner_states:
    :param n_actions:
    :return:
    """
    q_actions = tf.layers.dense(inner_states, units=n_actions, use_bias=True)
    return q_actions


def encoder_lstm(src, src_len, src_embeddings, num_units, num_layers):
    """
    encode state with LSTM
    :param src: placeholder, (tf.int32, [None, None])
    :param src_len: placeholder, (tf.float32, [None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param num_units: number of LSTM units
    :param num_layers: number of LSTM layers
    """
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(num_layers)])

    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    _, inner_states = tf.nn.dynamic_rnn(
        encoder_cell, src_emb, sequence_length=src_len,
        initial_state=None, dtype=tf.float32)
    return inner_states


def encoder_cnn_prepare_input_two_facets(src, src_embeddings, pos_embeddings):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [batch_size, src_len])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings: (tf.float32, [pos_emb_len, embedding_size])
    """
    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    pos_emb = tf.slice(pos_embeddings, [0, 0], [tf.shape(src_emb)[1], -1])
    src_pos_emb = src_emb + pos_emb
    src_emb_expanded = tf.stack([src_emb, src_pos_emb], axis=-1)
    return src_emb_expanded


def encoder_cnn_base(
        input_tensor, filter_sizes, num_filters, num_channels, embedding_size,
        is_infer=False):
    layer_outputs = []
    for i, fs in enumerate(filter_sizes):
        with tf.variable_scope("conv-block-%s" % fs):
            src_paddings = tf.constant([[0, 0], [fs - 1, 0], [0, 0], [0, 0]])
            src_w_pad = tf.pad(
                input_tensor, paddings=src_paddings, mode="CONSTANT")
            # Convolution Layer
            filter_shape = [fs, embedding_size, num_channels, num_filters]
            w = tf.get_variable(
                name="W",
                initializer=lambda: tf.truncated_normal(
                    filter_shape, stddev=0.1))
            b = tf.get_variable(
                name="b",
                initializer=lambda: tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(
                input=src_w_pad, filter=w, strides=[1, 1, 1, 1],
                padding="VALID", name="conv")
            # Apply nonlinearity
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
            dropout_h = tf.layers.dropout(
                inputs=h, rate=0.4,
                training=(not is_infer), name="dropout")
            layer_outputs.append(dropout_h)

    # Combine all the pooled features
    # Squeeze the 3rd dim that is the col of conv result
    inner_state = tf.squeeze(tf.concat(layer_outputs, axis=-1), axis=[2])
    return inner_state


def encoder_cnn_block(
        src, src_embeddings, pos_embeddings,
        filter_sizes, num_filters,
        embedding_size, is_infer=False):
    in_tn = encoder_cnn_prepare_input_two_facets(
        src, src_embeddings, pos_embeddings)
    return encoder_cnn_base(
        in_tn, filter_sizes, num_filters, num_channels=2,
        embedding_size=embedding_size, is_infer=is_infer)


def encoder_cnn(
        src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
        embedding_size, is_infer=False):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [None, None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings:
     position embedding, (tf.float32, [src_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    :param is_infer:
    """
    with tf.variable_scope("cnn_encoder"):
        h_cnn = encoder_cnn_block(
            src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
            embedding_size, is_infer)
        pooled = tf.reduce_max(h_cnn, axis=1)
        num_filters_total = num_filters * len(filter_sizes)
        inner_states = tf.reshape(pooled, [-1, num_filters_total])
    return inner_states


class LegacyCnnDRRN(BaseDQN):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
        :param hp:
        :param is_infer:
        """
        super(LegacyCnnDRRN, self).__init__(hp, is_infer)
        self.filter_sizes = [3, 4, 5]
        self.num_filters = hp.num_conv_filters
        self.num_tokens = hp.num_tokens
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.5

        self.src_embeddings = tf.get_variable(
            name="src_embeddings", dtype=tf.float32,
            shape=[hp.vocab_size, hp.embedding_size])

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

        self.n_actions = self.hp.n_actions
        self.n_tokens_per_action = self.hp.n_tokens_per_action
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(
                tf.int32, [None, self.n_tokens_per_action]),
            "actions_repeats": tf.placeholder(tf.int32, [None]),
            "actions_len": tf.placeholder(tf.float32, [None])
        }

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of
         hidden state and hidden actions
        h_state: (batch_size, n_hidden_state)
        h_state_expanded: (batch_size, 1, n_hidden_state)
        h_actions_expanded: (batch_size, n_actions, n_hidden_state)
        q_actions: (batch_size, n_actions)
        **q_actions = sum_k h_state_expanded_{ijk} * h_actions_expanded_{ijk}**
        i: batch_size
        j: n_actions
        k: n_hidden_state
        :return:
        """
        with tf.variable_scope("drrn-encoder", reuse=False):
            h_state = encoder_cnn(
                self.inputs["src"], self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            new_h = decoder_dense_classification(h_state, 32)
            h_state_expanded = dqn.repeat(new_h, self.inputs["actions_repeats"])

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                h_actions = encoder_lstm(
                    self.inputs["actions"],
                    self.inputs["actions_len"],
                    self.src_embeddings,
                    num_units=32,
                    num_layers=1)[-1].h

            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions, new_h

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return DRRNModel(
            graph=graph,
            q_actions=q_actions,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            actions_=inputs["actions"],
            actions_len_=inputs["actions_len"],
            actions_repeats_=inputs["actions_repeats"],
            b_weight_=inputs["b_weight"],
            h_state=new_h,
            abs_loss=abs_loss,
            train_op=train_op,
            action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"],
            loss=loss,
            train_summary_op=train_summary_op,
            src_seg_=None)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
        return DRRNModel(
            graph=graph, q_actions=q_actions,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            actions_=inputs["actions"],
            actions_len_=inputs["actions_len"],
            actions_repeats_=inputs["actions_repeats"],
            h_state=new_h,
            b_weight_=None,
            abs_loss=None,
            train_op=None,
            action_idx_=None,
            expected_q_=None,
            loss=None,
            train_summary_op=None,
            src_seg_=None)
