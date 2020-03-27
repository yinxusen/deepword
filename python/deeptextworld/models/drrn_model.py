import bert.modeling as b_model
import tensorflow as tf

import deeptextworld.models.utils as dqn
from deeptextworld.hparams import conventions
from deeptextworld.models.dqn_model import CnnDQN
from deeptextworld.models.encoders import LstmEncoder, TxEncoder
from deeptextworld.models.export_models import DRRNModel


class CnnDRRN(CnnDQN):
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
        super(CnnDRRN, self).__init__(hp, is_infer)
        self.n_tokens_per_action = self.hp.n_tokens_per_action
        self.h_state_size = self.hp.hidden_state_size
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
            "actions_len": tf.placeholder(tf.float32, [None])
        }
        self.enc_actions = LstmEncoder(
            num_units=self.h_state_size,
            num_layers=self.hp.lstm_num_layers,
            input_vocab_size=self.hp.vocab_size,
            embedding_size=self.hp.embedding_size)
        self.wt = tf.layers.Dense(
            units=self.h_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state
        and hidden actions
        :return:
        """
        _, pooled = self.enc_tj(self.inputs["src"])
        h_state = self.wt(pooled)
        _, h_actions = self.enc_actions(self.inputs["actions"])
        h_state_expanded = dqn.repeat(h_state, self.inputs["actions_repeats"])
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions

    @classmethod
    def get_train_model(cls, hp, device_placement):
        return create_train_model(cls, hp, device_placement)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        return create_eval_model(cls, hp, device_placement)

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        return cls.get_eval_model(hp, device_placement)


class TransformerDRRN(CnnDRRN):
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
        super(TransformerDRRN, self).__init__(hp, is_infer)
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
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


class BertDRRN(CnnDRRN):
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
        super(BertDRRN, self).__init__(hp, is_infer)
        self.bert_init_ckpt_dir = conventions.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.h_state_size = self.hp.hidden_state_size
        self.wt = tf.layers.Dense(
            units=self.h_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.hp.num_tokens, dtype=tf.int32)
        bert_config = b_model.BertConfig.from_json_file(self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        src_masks_w_pad = tf.pad(
            src_masks, paddings=paddings, mode="CONSTANT",
            constant_values=1)

        with tf.variable_scope("tj-bert-encoder"):
            bert_model = b_model.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src_w_pad, input_mask=src_masks_w_pad)
        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "tj-bert-encoder/bert/"})
        h_state = self.wt(bert_model.pooled_output)
        _, h_actions = self.enc_actions(self.inputs["actions"])
        h_state_expanded = dqn.repeat(h_state, self.inputs["actions_repeats"])
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


def create_train_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            inputs = model.inputs
            q_actions = model.get_q_actions()
            loss, train_op, abs_loss = model.get_train_op(q_actions)
            loss_summary = tf.summary.scalar("loss", loss)
            train_summary_op = tf.summary.merge([loss_summary])
    return DRRNModel(
        graph=graph, q_actions=q_actions,
        src_=inputs["src"],
        src_seg_=inputs["src_seg"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_repeats_=inputs["actions_repeats"],
        b_weight_=inputs["b_weight"],
        abs_loss=abs_loss,
        train_op=train_op, action_idx_=inputs["action_idx"],
        expected_q_=inputs["expected_q"], loss=loss,
        train_summary_op=train_summary_op,
        h_state=None)


def create_eval_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp, is_infer=True)
            inputs = model.inputs
            q_actions = model.get_q_actions()
    return DRRNModel(
        graph=graph, q_actions=q_actions,
        src_=inputs["src"],
        src_seg_=inputs["src_seg"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_repeats_=inputs["actions_repeats"],
        b_weight_=inputs["b_weight"],
        abs_loss=None,
        train_op=None, action_idx_=inputs["action_idx"],
        expected_q_=None, loss=None,
        train_summary_op=None,
        h_state=None)
