import albert.modeling as ab_model
import bert.modeling as b_model
import tensorflow as tf

import deeptextworld.models.utils as dqn
import deeptextworld.models.transformer as txf
from deeptextworld.models.dqn_model import BaseDQN, CnnDQN
from deeptextworld.models.export_models import DRRNModel, CommonsenseModel
from deeptextworld.models.simple_lstm import LstmEncoder


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
        self.n_actions = self.hp.n_actions
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
            self.hp.h_state_size, self.hp.lstm_num_layers,
            self.hp.vocab_size, self.hp.embedding_size)
        self.wt = tf.layers.Dense(
            units=self.h_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state
        and hidden actions
        :return:
        """
        h_state = self.enc_tj(self.inputs["src"])
        h_state = self.wt(h_state)
        h_actions = self.enc_actions(self.inputs["actions"])
        h_state_expanded = tf.repeat(
            h_state, self.inputs["actions_repeats"], axis=0)
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


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
        self.enc_tj = txf.Encoder(
            num_layers=1, d_model=128, num_heads=8, dff=256,
            input_vocab_size=self.hp.vocab_size)

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state
        and hidden actions
        :return:
        """
        padding_mask = txf.create_padding_mask(self.inputs["src"])
        inner_state = self.enc_tj(
            self.inputs["src"],
            training=(not self.is_infer), mask=padding_mask)
        pooled = tf.reduce_max(inner_state, axis=1)
        h_state = self.wt(pooled)
        h_state_expanded = tf.repeat(
            h_state, self.inputs["actions_repeats"], axis=0)
        h_actions = self.enc_actions(self.inputs["actions"])
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


def create_train_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
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


def create_eval_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
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


class BertDRRN(BaseDQN):
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
        self.bert_init_ckpt_dir = self.hp.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)

        self.h_state_size = self.hp.hidden_state_size
        self.enc_actions = tf.layers.Dense(
            units=self.hp.n_actions, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
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
        h_actions = self.enc_actions(self.inputs["actions"])
        h_state_expanded = tf.repeat(
            h_state, self.inputs["actions_repeats"], axis=0)
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class BertCommonsenseModel(BaseDQN):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode,
           has paddings, [CLS], and [SEP] prepared
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
        super(BertCommonsenseModel, self).__init__(hp, is_infer)
        self.num_tokens = hp.num_tokens
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.int32, [None]),
            "seg_tj_action": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None])
        }
        self.bert_init_ckpt_dir = self.hp.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        seg_tj_action = self.inputs["seg_tj_action"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        bert_config = b_model.BertConfig.from_json_file(self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

        with tf.variable_scope("bert-state-encoder"):
            bert_model = b_model.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks,
                token_type_ids=seg_tj_action)
            pooled = bert_model.pooled_output
            q_actions = tf.layers.dense(pooled, units=1, use_bias=True)[:, 0]

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return q_actions

    def get_train_op(self, q_actions):
        losses = tf.squared_difference(self.inputs["expected_q"], q_actions)
        loss = tf.reduce_mean(losses)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return create_train_bert_commonsense_model(cls, hp, device_placement)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        return create_eval_bert_commonsense_model(cls, hp, device_placement)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        return cls.get_train_student_model(hp, device_placement)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        return cls.get_eval_student_model(hp, device_placement)


class AlbertCommonsenseModel(BertCommonsenseModel):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode,
           has paddings, [CLS], and [SEP] prepared
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
        super(AlbertCommonsenseModel, self).__init__(hp, is_infer)
        self.bert_config_file = "{}/albert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/model.ckpt-best".format(
            self.bert_init_ckpt_dir)

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        seg_tj_action = self.inputs["seg_tj_action"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        bert_config = ab_model.AlbertConfig.from_json_file(
            self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

        with tf.variable_scope("bert-state-encoder"):
            bert_model = ab_model.AlbertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks,
                token_type_ids=seg_tj_action)
            pooled = bert_model.pooled_output
            q_actions = tf.layers.dense(pooled, units=1, use_bias=True)[:, 0]

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return q_actions


def create_train_bert_commonsense_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            inputs = model.inputs
            q_actions = model.get_q_actions()
            loss, train_op = model.get_train_op(q_actions)
            loss_summary = tf.summary.scalar("loss", loss)
            train_summary_op = tf.summary.merge([loss_summary])
    return CommonsenseModel(
        graph=graph,
        q_actions=q_actions,
        src_seg_=None,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        loss=loss,
        train_op=train_op,
        train_summary_op=train_summary_op,
        expected_q_=inputs["expected_q_"],
        action_idx_=inputs["action_idx"],
        abs_loss=None,
        seg_tj_action_=inputs["seg_tj_action"],
        h_state=None,
        b_weight_=None)


def create_eval_bert_commonsense_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            inputs = model.inputs
            q_actions = model.get_q_actions()
    return CommonsenseModel(
        graph=graph,
        q_actions=q_actions,
        src_seg_=None,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        loss=None,
        train_op=None,
        train_summary_op=None,
        expected_q_=inputs["expected_q_"],
        action_idx_=inputs["action_idx"],
        abs_loss=None,
        seg_tj_action_=inputs["seg_tj_action"],
        h_state=None,
        b_weight_=None)
