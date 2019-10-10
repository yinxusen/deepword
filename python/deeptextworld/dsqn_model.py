import collections

import tensorflow as tf
from bert import modeling

import deeptextworld.dqn_func as dqn
from deeptextworld import transformer as txf
from deeptextworld.dqn_model import CNNEncoderDQN


class TrainDSQNModel(
    collections.namedtuple(
        'TrainDSQNModel',
        ('graph', 'model', 'q_actions', 'train_op', 'loss', 'train_summary_op',
         'snn_train_summary_op', 'weighted_train_summary_op',
         'src_', 'src_len_', 'actions_', 'actions_len_', 'actions_mask_',
         'action_idx_', 'expected_q_', 'b_weight_', 'abs_loss', 'pred',
         'snn_src_', "snn_src_len_", "snn_src2_", "snn_src2_len_", "labels_",
         'snn_loss', 'weighted_loss', 'merged_train_op', 'snn_train_op',
         'diff_two_states',
         'initializer'))):
    pass


class EvalDSQNModel(
    collections.namedtuple(
        'EvalDSQNModel',
        ('graph', 'model', 'q_actions', 'pred',
         'src_', 'src_len_', 'actions_', 'actions_len_', 'actions_mask_',
         'snn_src_', "snn_src_len_", "snn_src2_", "snn_src2_len_", "labels_",
         'diff_two_states',
         'initializer'))):
    pass


class CNNEncoderDSQN(CNNEncoderDQN):
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
        super(CNNEncoderDSQN, self).__init__(hp, src_embeddings, is_infer)
        self.n_actions = self.hp.n_actions
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(tf.int32, [None, self.n_actions, None]),
            "actions_len": tf.placeholder(tf.float32, [None, self.n_actions]),
            "actions_mask": tf.placeholder(tf.float32, [None, self.n_actions]),
            "snn_src": tf.placeholder(tf.int32, [None, None]),
            "snn_src_len": tf.placeholder(tf.float32, [None]),
            "snn_src2": tf.placeholder(tf.int32, [None, None]),
            "snn_src2_len": tf.placeholder(tf.float32, [None]),
            "labels": tf.placeholder(tf.float32, [None])
        }

    def get_q_actions(self):
        """
        compute the Q-vector from the relevance of hidden state and actions
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
            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state_expanded = tf.expand_dims(new_h, axis=1)

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                flat_actions = tf.reshape(
                    self.inputs["actions"],
                    shape=(batch_size * self.n_actions, -1))
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

    def get_pred(self):
        with tf.variable_scope("drrn-encoder", reuse=True):
            h_state = dqn.encoder_cnn(
                self.inputs["snn_src"],
                self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            h_state2 = dqn.encoder_cnn(
                self.inputs["snn_src2"],
                self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)

        diff_two_states = tf.abs(h_state - h_state2)
        pred = tf.squeeze(tf.layers.dense(
            diff_two_states, activation=tf.nn.sigmoid, units=1, use_bias=True,
            name="snn_dense"))
        return pred, diff_two_states

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    def get_snn_train_op(self, pred):
        labels = self.inputs["labels"]
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=pred)
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


class AttnEncoderDSQN(CNNEncoderDSQN):
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
        super(AttnEncoderDSQN, self).__init__(hp, src_embeddings, is_infer)
        self.attn_encoder = txf.Encoder(
            num_layers=1, d_model=128, num_heads=8, dff=256,
            input_vocab_size=self.hp.vocab_size)

    def get_q_actions(self):
        batch_size = tf.shape(self.inputs["src_len"])[0]

        with tf.variable_scope("drrn-attn-encoder", reuse=False):
            padding_mask = txf.create_padding_mask(self.inputs["src"])
            inner_state = self.attn_encoder(
                self.inputs["src"], x_seg=None,
                training=(not self.is_infer), mask=padding_mask)
            pooled = tf.reduce_max(inner_state, axis=1)
            h_state = tf.reshape(pooled, [-1, 128])
            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state_expanded = tf.expand_dims(new_h, axis=1)

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                flat_actions = tf.reshape(
                    self.inputs["actions"],
                    shape=(batch_size * self.n_actions, -1))
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

    def get_h_state(self, src, src_len):
        padding_mask = txf.create_padding_mask(src)
        inner_state = self.attn_encoder(
            src, x_seg=None,
            training=(not self.is_infer), mask=padding_mask)
        pooled = tf.reduce_max(inner_state, axis=1)
        h_state = tf.reshape(pooled, [-1, 128])
        return h_state

    def get_pred(self):
        with tf.variable_scope("drrn-attn-encoder", reuse=True):
            h_state = self.get_h_state(
                self.inputs["snn_src"], self.inputs["snn_src_len"])
            h_state2 = self.get_h_state(
                self.inputs["snn_src2"], self.inputs["snn_src2_len"])

        diff_two_states = tf.abs(h_state - h_state2)
        pred = tf.squeeze(tf.layers.dense(
            diff_two_states, activation=tf.nn.sigmoid, units=1, use_bias=True,
            name="snn_dense"))
        return pred, diff_two_states


class BertAttnEncoderDSQN(AttnEncoderDSQN):
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
        super(BertAttnEncoderDSQN, self).__init__(hp, src_embeddings, is_infer)
        self.bert_init_ckpt_dir = self.hp.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.bert_config = modeling.BertConfig.from_json_file(
            self.bert_config_file)
        self.bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers
        self.enc_layer = txf.EncoderLayer(d_model=768, num_heads=8, dff=768)
        self.pooler_layer = tf.layers.Dense(
            units=32, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        batch_size = tf.shape(self.inputs["src_len"])[0]

        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        actions = tf.reshape(
            self.inputs["actions"], shape=(batch_size * self.n_actions, -1))
        actions_mask = txf.create_padding_mask(actions)
        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        src_masks_w_pad = tf.pad(
            src_masks, paddings=paddings, mode="CONSTANT",
            constant_values=1)

        with tf.variable_scope("bert-state-encoder"):
            bert_model = modeling.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src_w_pad, input_mask=src_masks_w_pad)
            enc_out = self.enc_layer(
                bert_model.get_sequence_output(), (not self.is_infer),
                mask=None)
            first_token_tensor = tf.squeeze(enc_out[:, 0:1, :], axis=1)
            h_state = self.pooler_layer(first_token_tensor)
        with tf.variable_scope("attn-action-encoder"):
            attn_encoder = txf.Encoder(
                num_layers=1, d_model=32, num_heads=8, dff=64,
                input_vocab_size=self.hp.vocab_size)
            flat_inner_state = attn_encoder(
                actions, None, (not self.is_infer), actions_mask)
            pooled = tf.reduce_max(flat_inner_state, axis=1)
            h_actions = tf.reshape(
                pooled, shape=(batch_size, self.n_actions, -1))

        with tf.variable_scope("drrn-encoder", reuse=False):
            h_state_expanded = tf.expand_dims(h_state, axis=1)
            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return q_actions

    def get_h_state(self, src, src_len):
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        src_masks_w_pad = tf.pad(
            src_masks, paddings=paddings, mode="CONSTANT",
            constant_values=1)
        with tf.variable_scope("bert-state-encoder", reuse=True):
            bert_model = modeling.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src_w_pad, input_mask=src_masks_w_pad)
            h_state = bert_model.get_pooled_output()
        return h_state

    def get_pred(self):
        h_state = self.get_h_state(
            self.inputs["snn_src"], self.inputs["snn_src_len"])
        h_state2 = self.get_h_state(
            self.inputs["snn_src2"], self.inputs["snn_src2_len"])

        diff_two_states = tf.abs(h_state - h_state2)
        pred = tf.squeeze(tf.layers.dense(
            diff_two_states, activation=tf.nn.sigmoid, units=1, use_bias=True,
            name="snn_dense"))
        return pred, diff_two_states

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        tvars_bert_state = tf.trainable_variables(scope="bert-state-encoder")
        tvars_attn_action = tf.trainable_variables(scope="attn-action-encoder")

        if self.hp.ft_bert_layers == 0:
            allowed_tvars_state = []
        elif self.hp.ft_bert_layers == -1:
            allowed_tvars_state = tvars_bert_state
        else:
            allowed_tvars_state = []
            for t_layer in range(
                    min(self.hp.ft_bert_layers,
                        self.hp.bert_num_hidden_layers)):
                allowed_tvars_state += list(filter(
                    lambda v: "layer_{}".format(
                        self.hp.bert_num_hidden_layers - t_layer - 1) in v.name,
                    tvars_bert_state))
            allowed_tvars_state += list(filter(
                lambda v: "pooler" in v.name, tvars_bert_state))

        allowed_tvars_action = tvars_attn_action

        tvars_drrn = tf.trainable_variables(scope="drrn-encoder")
        tvars = tvars_drrn + allowed_tvars_state + allowed_tvars_action
        train_op = self.optimizer.minimize(
            loss, global_step=self.global_step, var_list=tvars)
        return loss, train_op, abs_loss

    def get_snn_train_op(self, pred):
        labels = self.inputs["labels"]
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=pred)
        loss = tf.reduce_mean(losses)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op


def create_train_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        q_actions = model.get_q_actions()
        pred, diff_two_states = model.get_pred()
        loss, train_op, abs_loss = model.get_train_op(q_actions)
        snn_loss, snn_train_op = model.get_snn_train_op(pred)
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
    return TrainDSQNModel(
        graph=graph, model=model, q_actions=q_actions, pred=pred,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_mask_=inputs["actions_mask"],
        snn_src_=inputs["snn_src"],
        snn_src_len_=inputs["snn_src_len"],
        snn_src2_=inputs["snn_src2"],
        snn_src2_len_=inputs["snn_src2_len"],
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
        diff_two_states=diff_two_states,
        initializer=initializer)


def create_eval_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp, is_infer=True)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        q_actions = model.get_q_actions()
        pred, diff_two_states = model.get_pred()
        loss, train_op, abs_loss = model.get_train_op(q_actions)
        snn_loss, snn_train_op = model.get_snn_train_op(pred)
        _ = model.get_merged_train_op(loss, snn_loss)
    return EvalDSQNModel(
        graph=graph, model=model, q_actions=q_actions, pred=pred,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_mask_=inputs["actions_mask"],
        snn_src_=inputs["snn_src"],
        snn_src_len_=inputs["snn_src_len"],
        snn_src2_=inputs["snn_src2"],
        snn_src2_len_=inputs["snn_src2_len"],
        labels_=inputs["labels"],
        diff_two_states=diff_two_states,
        initializer=initializer)
