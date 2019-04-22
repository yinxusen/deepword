import collections

import tensorflow as tf

import deeptextworld.dqn_func as dqn
from deeptextworld.dqn_model import CNNEncoderDQN
from deeptextworld.bert_layer import BertLayer


class TrainBertDRRNModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions','train_op', 'loss', 'train_summary_op',
         'src_', 'src_len_', 'actions_', 'actions_len_', 'actions_mask_',
         'action_idx_', 'expected_q_', 'b_weight_', 'abs_loss',
         'initializer'))):
    pass


class EvalBertDRRNModel(
    collections.namedtuple(
        'EvalModel',
        ('graph', 'model', 'q_actions',
         'src_', 'src_len_', 'actions_', 'actions_len_', 'actions_mask_',
         'initializer'))):
    pass


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

        # src = tf.pad(
        #     self.inputs["src"], paddings=tf.constant([[0, 0], [1, 0]]),
        #     mode="CONSTANT", constant_values=self.hp.cls_val_id)
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)
        src_segment_ids = tf.zeros_like(src, dtype=tf.int32)

        actions = tf.reshape(
            self.inputs["actions"], shape=(-1, self.n_tokens_per_action))
        # actions = tf.pad(
        #     actions, paddings=tf.constant([[0, 0], [1, 0]]),
        #     mode="CONSTANT", constant_values=self.hp.cls_val_id)
        actions_len = tf.reshape(
            self.inputs["actions_len"], shape=(-1,))
        actions_token_masks = tf.sequence_mask(
            actions_len, maxlen=self.n_tokens_per_action, dtype=tf.int32)
        actions_segment_ids = tf.zeros_like(actions, dtype=tf.int32)

        with tf.variable_scope("drrn-encoder", reuse=False):
            bert_layer = BertLayer(n_fine_tune_layers=3)
            with tf.variable_scope("bert-cnn-encoder", reuse=False):
                bert_output = bert_layer([src, src_masks, src_segment_ids])
                h_cnn = dqn.encoder_cnn_base(
                    bert_output, self.filter_sizes, self.num_filters,
                    self.hp.embedding_size, self.is_infer)
                pooled = tf.reduce_max(h_cnn, axis=1)
                num_filters_total = self.num_filters * len(self.filter_sizes)
                h_state = tf.reshape(pooled, [-1, num_filters_total])

            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state_expanded = tf.expand_dims(new_h, axis=1)

            with tf.variable_scope("drrn-action-encoder", reuse=False):
                bert_output_actions = bert_layer(
                    [actions, actions_token_masks, actions_segment_ids])
                encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(32) for _ in range(1)])
                sequence_output, inner_state = tf.keras.layers.RNN(
                    encoder_cell, bert_output_actions,
                    sequence_length=actions_len,
                    initial_state=None, dtype=tf.float32)
                flat_h_actions = inner_state[-1].h
                h_actions = tf.reshape(flat_h_actions,
                                       shape=(batch_size, self.n_actions, -1))
            q_actions = tf.reduce_sum(
                tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions


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
    return TrainBertDRRNModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=inputs["src"],
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
    return EvalBertDRRNModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_mask_=inputs["actions_mask"],
        initializer=initializer)
