import collections

import tensorflow as tf
from bert import modeling

import deeptextworld.dqn_func as dqn
import deeptextworld.transformer as txf
from deeptextworld.dqn_model import BaseDQN, CNNEncoderDQN


class TrainDRRNModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions','train_op', 'loss', 'train_summary_op',
         'src_', 'src_seg_', 'src_len_',
         'actions_', 'actions_len_', 'actions_mask_',
         'action_idx_', 'expected_q_', 'b_weight_', 'abs_loss',
         'initializer'))):
    pass


class EvalDRRNModel(
    collections.namedtuple(
        'EvalModel',
        ('graph', 'model', 'q_actions',
         'src_', 'src_seg_', 'src_len_',
         'actions_', 'actions_len_', 'actions_mask_',
         'initializer'))):
    pass


class CNNEncoderDRRN(CNNEncoderDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
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
        :param src_embeddings:
        :param is_infer:
        """
        super(CNNEncoderDRRN, self).__init__(hp, src_embeddings, is_infer)
        self.n_actions = self.hp.n_actions
        self.n_tokens_per_action = self.hp.n_tokens_per_action
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_seg": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(tf.int32, [None, self.n_actions,
                                                 self.n_tokens_per_action]),
            "actions_len": tf.placeholder(tf.float32, [None, self.n_actions]),
            "actions_mask": tf.placeholder(tf.float32, [None, self.n_actions])
        }

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state and hidden actions
        h_state: (batch_size, n_hidden_state)
        h_state_expanded: (batch_size, 1, n_hidden_state)

        h_actions_expanded: (1, n_actions, n_hidden_state)
        actions_mask: (batch_size, n_actions, 1)
        h_actions_masked: (batch_size, n_actions, n_hidden_state)

        **h_actions_masked = h_actions_expanded * actions_mask**

        q_actions: (batch_size, n_actions)

        **q_actions = \sum_k h_state_expanded_{ijk} * h_actions_masked_{ijk}**

        i: batch_size
        j: n_actions
        k: n_hidden_state
        :return:
        """
        batch_size = tf.shape(self.inputs["src_len"])[0]

        with tf.variable_scope("drrn-encoder", reuse=False):
            h_state = dqn.encoder_cnn_3(
                self.inputs["src"], self.inputs["src_seg"],
                self.src_embeddings, self.pos_embeddings, self.seg_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state_expanded = tf.expand_dims(new_h, axis=1)

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                flat_actions = tf.reshape(
                    self.inputs["actions"],
                    shape=(-1, self.n_tokens_per_action))
                flat_actions_len = tf.reshape(
                    self.inputs["actions_len"],
                    shape=(-1,))
                flat_h_actions = dqn.encoder_lstm(
                    flat_actions, flat_actions_len,
                    self.src_embeddings,
                    num_units=32,
                    num_layers=1)[-1].h
                h_actions = tf.reshape(flat_h_actions,
                                       shape=(batch_size, self.n_actions, -1))
            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


class AttnEncoderDRRN(CNNEncoderDRRN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
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
        :param src_embeddings:
        :param is_infer:
        """
        super(AttnEncoderDRRN, self).__init__(hp, src_embeddings, is_infer)

    def get_q_actions(self):
        batch_size = tf.shape(self.inputs["src_len"])[0]

        with tf.variable_scope("drrn-attn-encoder", reuse=False):
            attn_encoder = txf.Encoder(
                num_layers=1, d_model=128, num_heads=8, dff=256,
                input_vocab_size=self.hp.vocab_size)
            padding_mask = txf.create_padding_mask(self.inputs["src"])
            inner_state = attn_encoder(
                self.inputs["src"], x_seg=self.inputs["src_seg"],
                training=(not self.is_infer), mask=padding_mask)
            pooled = tf.reduce_max(inner_state, axis=1)
            h_state = tf.reshape(pooled, [-1, 128])
            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state_expanded = tf.expand_dims(new_h, axis=1)

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                flat_actions = tf.reshape(
                    self.inputs["actions"],
                    shape=(-1, self.n_tokens_per_action))
                flat_actions_len = tf.reshape(
                    self.inputs["actions_len"],
                    shape=(-1,))
                flat_h_actions = dqn.encoder_lstm(
                    flat_actions, flat_actions_len,
                    self.src_embeddings,
                    num_units=32,
                    num_layers=1)[-1].h
                h_actions = tf.reshape(
                    flat_h_actions,
                    shape=(batch_size, self.n_actions, -1))
            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


class CNNAttnEncoderDRRN(CNNEncoderDRRN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
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
        :param src_embeddings:
        :param is_infer:
        """
        super(CNNAttnEncoderDRRN, self).__init__(hp, src_embeddings, is_infer)
        self.num_tokens = self.hp.num_tokens
        self.num_features = len(self.filter_sizes) * self.num_filters

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state and hidden actions
        h_state: (batch_size, n_hidden_state)
        h_state_expanded: (batch_size, 1, n_hidden_state)

        h_actions_expanded: (1, n_actions, n_hidden_state)
        actions_mask: (batch_size, n_actions, 1)
        h_actions_masked: (batch_size, n_actions, n_hidden_state)

        **h_actions_masked = h_actions_expanded * actions_mask**

        q_actions: (batch_size, n_actions)

        **q_actions = \sum_k h_state_expanded_{ijk} * h_actions_masked_{ijk}**

        i: batch_size
        j: n_actions
        k: n_hidden_state
        :return:
        """
        batch_size = tf.shape(self.inputs["src_len"])[0]

        with tf.variable_scope("drrn-encoder", reuse=False):
            h_state = dqn.encoder_cnn(
                self.inputs["src"], self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            h_state_expanded = tf.expand_dims(h_state, axis=1)
        with tf.variable_scope("drrn-encoder", reuse=True):
            flat_actions = tf.reshape(
                self.inputs["actions"], shape=(-1, self.n_tokens_per_action))
            flat_h_actions = dqn.encoder_cnn(
                flat_actions, self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            h_actions = tf.reshape(
                flat_h_actions,
                shape=(batch_size, self.n_actions, -1))

        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


class BertEncoderDRRN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
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
        :param src_embeddings:
        :param is_infer:
        """
        super(BertEncoderDRRN, self).__init__(hp, src_embeddings, is_infer)
        self.n_actions = self.hp.n_actions
        self.num_tokens = hp.num_tokens
        self.n_tokens_per_action = self.hp.n_tokens_per_action
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(
                tf.int32, [None, self.n_actions, self.n_tokens_per_action]),
            "actions_len": tf.placeholder(tf.float32, [None, self.n_actions]),
            "actions_mask": tf.placeholder(tf.float32, [None, self.n_actions])
        }
        self.bert_init_ckpt_dir = self.hp.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(self.bert_init_ckpt_dir)

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state and hidden actions
        h_state: (batch_size, n_hidden_state)
        h_state_expanded: (batch_size, 1, n_hidden_state)

        h_actions_expanded: (1, n_actions, n_hidden_state)
        actions_mask: (batch_size, n_actions, 1)
        h_actions_masked: (batch_size, n_actions, n_hidden_state)

        **h_actions_masked = h_actions_expanded * actions_mask**

        q_actions: (batch_size, n_actions)

        **q_actions = \sum_k h_state_expanded_{ijk} * h_actions_masked_{ijk}**

        i: batch_size
        j: n_actions
        k: n_hidden_state
        :return:
        """
        batch_size = tf.shape(self.inputs["src_len"])[0]

        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        actions = tf.reshape(
            self.inputs["actions"], shape=(-1, self.n_tokens_per_action))
        actions_len = tf.reshape(
            self.inputs["actions_len"], shape=(-1,))
        actions_token_masks = tf.sequence_mask(
            actions_len, maxlen=self.n_tokens_per_action, dtype=tf.int32)

        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        src_masks_w_pad = tf.pad(
            src_masks, paddings=paddings, mode="CONSTANT",
            constant_values=1)
        actions_w_pad = tf.pad(
            actions, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        actions_token_masks_w_pad = tf.pad(
            actions_token_masks, paddings=paddings, mode="CONSTANT",
            constant_values=1)

        with tf.variable_scope("bert-state-encoder"):
            bert_model = modeling.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src_w_pad, input_mask=src_masks_w_pad)
            h_state = bert_model.pooled_output
        with tf.variable_scope("bert-action-encoder"):
            bert_action_model = modeling.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=actions_w_pad, input_mask=actions_token_masks_w_pad)
            flat_h_actions = bert_action_model.pooled_output
            h_actions = tf.reshape(
                flat_h_actions,
                shape=(batch_size, self.n_actions, -1))

        with tf.variable_scope("drrn-encoder", reuse=False):
            new_h = dqn.decoder_dense_classification(h_state, 768)
            h_state_expanded = tf.expand_dims(new_h, axis=1)
            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})
        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-action-encoder/bert/"})

        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        tvars_bert_state = tf.trainable_variables(scope="bert-state-encoder")
        tvars_bert_action = tf.trainable_variables(scope="bert-action-encoder")

        allowed_tvars_state = list(filter(
            lambda v: "layer_11" in v.name or "pooler" in v.name,
            tvars_bert_state))
        allowed_tvars_action = list(filter(
            lambda v: "layer_11" in v.name or "pooler" in v.name,
            tvars_bert_action))

        tvars_drrn = tf.trainable_variables(scope="drrn-encoder")
        tvars = tvars_drrn + allowed_tvars_state + allowed_tvars_action
        train_op = self.optimizer.minimize(
            loss, global_step=self.global_step, var_list=tvars)
        return loss, train_op, abs_loss


class BertCNNEncoderDRRN(CNNEncoderDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
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
        :param src_embeddings:
        :param is_infer:
        """
        super(BertCNNEncoderDRRN, self).__init__(hp, src_embeddings, is_infer)
        self.n_actions = self.hp.n_actions
        self.n_tokens_per_action = self.hp.n_tokens_per_action
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(
                tf.int32, [None, self.n_actions, self.n_tokens_per_action]),
            "actions_len": tf.placeholder(tf.float32, [None, self.n_actions]),
            "actions_mask": tf.placeholder(tf.float32, [None, self.n_actions])
        }
        self.bert_init_ckpt_dir = self.hp.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(self.bert_init_ckpt_dir)

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state and hidden actions
        h_state: (batch_size, n_hidden_state)
        h_state_expanded: (batch_size, 1, n_hidden_state)

        h_actions_expanded: (1, n_actions, n_hidden_state)
        actions_mask: (batch_size, n_actions, 1)
        h_actions_masked: (batch_size, n_actions, n_hidden_state)

        **h_actions_masked = h_actions_expanded * actions_mask**

        q_actions: (batch_size, n_actions)

        **q_actions = \sum_k h_state_expanded_{ijk} * h_actions_masked_{ijk}**

        i: batch_size
        j: n_actions
        k: n_hidden_state
        :return:
        """
        batch_size = tf.shape(self.inputs["src_len"])[0]

        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        actions = tf.reshape(
            self.inputs["actions"], shape=(-1, self.n_tokens_per_action))
        actions_len = tf.reshape(
            self.inputs["actions_len"], shape=(-1,))
        actions_token_masks = tf.sequence_mask(
            actions_len, maxlen=self.n_tokens_per_action, dtype=tf.int32)

        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers
        with tf.variable_scope("bert-embedding"):
            bert_model = modeling.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks, scope="bert")
            src_bert_embeddings = bert_model.sequence_output
        # TODO: I don't understand the reuse behavior for nested variable_scope
        with tf.variable_scope("bert-embedding", reuse=True):
            bert_action_model = modeling.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=actions, input_mask=actions_token_masks, scope="bert")
            actions_bert_embeddings = bert_action_model.sequence_output

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-embedding/bert/"})

        with tf.variable_scope("drrn-encoder", reuse=False):
            with tf.variable_scope("cnn-encoder", reuse=False):
                src_bert_embeddings = tf.expand_dims(src_bert_embeddings,
                                                     axis=-1)
                # when use bert, there is no need to add pos emb since bert
                # contains that.
                h_cnn = dqn.encoder_cnn_base(
                    src_bert_embeddings, self.filter_sizes, self.num_filters,
                    num_channels=1, embedding_size=self.hp.embedding_size,
                    is_infer=self.is_infer)
                pooled = tf.reduce_max(h_cnn, axis=1)
                num_filters_total = self.num_filters * len(self.filter_sizes)
                h_state = tf.reshape(pooled, [-1, num_filters_total])

            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state_expanded = tf.expand_dims(new_h, axis=1)

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(32) for _ in range(1)])
                sequence_output, inner_state = tf.nn.dynamic_rnn(
                    encoder_cell, actions_bert_embeddings,
                    sequence_length=actions_len,
                    initial_state=None, dtype=tf.float32)
                flat_h_actions = inner_state[-1].h
                h_actions = tf.reshape(flat_h_actions,
                                       shape=(batch_size, self.n_actions, -1))
            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        tvars_bert = tf.trainable_variables(scope="bert-embedding")
        # allow the last layer of bert to be fine-tuned.
        train_layer = self.hp.bert_num_hidden_layers - 1
        allowed_tvars_bert = list(filter(
            lambda v: "layer_{}".format(train_layer) in v.name,
            tvars_bert))
        tvars_drrn = tf.trainable_variables(scope="drrn-encoder")
        tvars = tvars_drrn + allowed_tvars_bert
        train_op = self.optimizer.minimize(
            loss, global_step=self.global_step, var_list=tvars)
        return loss, train_op, abs_loss


def create_train_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        q_actions = model.get_q_actions()
        loss, train_op, abs_loss = model.get_train_op(q_actions)
        loss_summary = tf.summary.scalar("loss", loss)
        train_summary_op = tf.summary.merge([loss_summary])
    return TrainDRRNModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=inputs["src"],
        src_seg_=inputs["src_seg"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_mask_=inputs["actions_mask"],
        b_weight_=inputs["b_weight"],
        abs_loss=abs_loss,
        train_op=train_op, action_idx_=inputs["action_idx"],
        expected_q_=inputs["expected_q"], loss=loss,
        train_summary_op=train_summary_op,
        initializer=initializer)


def create_eval_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp, is_infer=True)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        q_actions = model.get_q_actions()
        # still need to put them here, otherwise the loaded model could not be trained
        _ = model.get_train_op(q_actions)
    return EvalDRRNModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=inputs["src"],
        src_seg_=inputs["src_seg"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_mask_=inputs["actions_mask"],
        initializer=initializer)
