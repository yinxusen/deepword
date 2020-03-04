import tensorflow as tf

import deeptextworld.models.utils as dqn
from deeptextworld.models.encoders import CnnEncoder, LstmEncoder
from deeptextworld.models.export_models import DQNModel


class BaseDQN(object):
    def __init__(self, hp, is_infer=False):
        self.is_infer = is_infer
        self.hp = hp
        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(self.hp.learning_rate)
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None])
        }

    def get_q_actions(self):
        raise NotImplementedError()

    def get_train_op(self, q_actions):
        raise NotImplementedError()

    @classmethod
    def get_train_model(cls, hp, device_placement):
        return create_train_model(cls, hp, device_placement)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        return create_eval_model(cls, hp, device_placement)


class LstmDQN(BaseDQN):
    def __init__(self, hp, is_infer=False):
        super(LstmDQN, self).__init__(hp, is_infer)
        self.enc_tj = LstmEncoder(
            self.hp.lstm_num_units, self.hp.lstm_num_layers,
            self.hp.vocab_size, self.hp.embedding_size)
        self.enc_actions = tf.layers.Dense(
            units=self.hp.n_actions, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        h_state = self.enc_tj(self.inputs["src"])
        q_actions = self.enc_actions(h_state)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class CnnDQN(BaseDQN):
    def __init__(self, hp, is_infer=False):
        super(CnnDQN, self).__init__(hp, is_infer)
        filter_sizes = [3, 4, 5]
        num_filters = hp.num_conv_filters
        self.enc_tj = CnnEncoder(
            filter_sizes=filter_sizes, num_filters=num_filters,
            num_layers=1, input_vocab_size=self.hp.vocab_size)
        self.enc_actions = tf.layers.Dense(
            units=self.hp.n_actions, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        h_state = self.enc_tj(self.inputs["src"])
        q_actions = self.enc_actions(h_state)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


def create_train_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
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


def create_eval_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp, is_infer=True)
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
