import numpy as np
import tensorflow as tf

import deepword.models.utils as dqn
from deepword.hparams import conventions
from deepword.log import Logging
from deepword.models.models import DQNModel


class BaseDQN(Logging):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(BaseDQN, self).__init__()
        self.is_infer = is_infer
        self.hp = hp
        if src_embeddings is None:
            if hp.use_glove_emb:
                _, glove_emb = self.init_glove(conventions.glove_emb_file)
                self.src_embeddings = tf.get_variable(
                    name="src_embeddings", dtype=tf.float32,
                    initializer=glove_emb, trainable=hp.glove_trainable)
            else:
                self.src_embeddings = tf.get_variable(
                    name="src_embeddings", dtype=tf.float32,
                    shape=[hp.vocab_size, hp.embedding_size])
        else:
            self.src_embeddings = src_embeddings

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(self.hp.learning_rate)
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None])
        }

    @classmethod
    def init_glove(cls, glove_path):
        with open(glove_path, "r") as f:
            glove = list(map(lambda s: s.strip().split(), f.readlines()))
        glove_tokens = list(map(lambda x: x[0], glove))
        glove_embeddings = np.asarray(
            list(map(lambda x: x[1:], glove)), dtype=np.float32)
        return glove_tokens, glove_embeddings

    def get_q_actions(self):
        raise NotImplementedError()

    def get_train_op(self, q_actions):
        raise NotImplementedError()

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                src_placeholder = inputs["src"]
                src_len_placeholder = inputs["src_len"]
                action_idx_placeholder = inputs["action_idx"]
                expected_q_placeholder = inputs["expected_q"]
                b_weight_placeholder = inputs["b_weight"]
                q_actions = model.get_q_actions()
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return DQNModel(
            graph=graph,
            q_actions=q_actions,
            src_=src_placeholder,
            src_len_=src_len_placeholder,
            train_op=train_op,
            action_idx_=action_idx_placeholder,
            expected_q_=expected_q_placeholder,
            b_weight_=b_weight_placeholder,
            loss=loss,
            train_summary_op=train_summary_op,
            abs_loss=abs_loss,
            src_seg_=None,
            h_state=None)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                src_placeholder = inputs["src"]
                src_len_placeholder = inputs["src_len"]
                q_actions = model.get_q_actions()
        return DQNModel(
            graph=graph,
            q_actions=q_actions,
            src_=src_placeholder,
            src_len_=src_len_placeholder,
            train_op=None,
            action_idx_=None,
            expected_q_=None,
            b_weight_=None,
            loss=None,
            train_summary_op=None,
            abs_loss=None,
            src_seg_=None,
            h_state=None)


class LstmDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(LstmDQN, self).__init__(hp, src_embeddings, is_infer)

    def get_q_actions(self):
        inner_states = dqn.encoder_lstm(
            self.inputs["src"], self.inputs["src_len"], self.src_embeddings,
            self.hp.lstm_num_units, self.hp.lstm_num_layers)
        q_actions = tf.layers.dense(
            inner_states[-1].c, units=self.hp.n_actions, use_bias=True)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action_v2(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class CnnDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(CnnDQN, self).__init__(hp, src_embeddings, is_infer)
        self.filter_sizes = [3, 4, 5]
        self.num_filters = hp.num_conv_filters
        self.num_tokens = hp.num_tokens
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.5

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

    def get_q_actions(self):
        inner_states = dqn.encoder_cnn(
            self.inputs["src"], self.src_embeddings, self.pos_embeddings,
            self.filter_sizes, self.num_filters, self.hp.embedding_size,
            self.is_infer)
        q_actions = tf.layers.dense(
            inner_states, units=self.hp.n_actions, use_bias=True)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action_v2(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss
