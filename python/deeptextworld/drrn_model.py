import collections

import tensorflow as tf

import deeptextworld.dqn_func as dqn
from deeptextworld.dqn_model import CNNEncoderDQN


class TrainDRRNModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions','train_op', 'loss', 'train_summary_op',
         'src_', 'src_len_', 'actions_', 'actions_len_', 'actions_mask_',
         'action_idx_', 'expected_q_', 'b_weight_', 'abs_loss',
         'initializer'))):
    pass


class EvalDRRNModel(
    collections.namedtuple(
        'EvalModel',
        ('graph', 'model', 'q_actions',
         'src_', 'src_len_', 'actions_', 'actions_len_', 'actions_mask_',
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
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(tf.int32, [None, self.n_actions,
                                                 self.n_tokens_per_action]),
            "actions_len": tf.placeholder(tf.float32, [None, self.n_actions]),
            "actions_mask": tf.placeholder(tf.float32, [None, self.n_actions])
        }

        # use smaller config for action encoding
        self.filter_sizes_action = [1, 2, 3]
        self.num_filters_action = hp.num_conv_filters
        self.num_tokens_actions = hp.num_tokens
        self.embedding_size_action = 8

        self.src_embeddings_action = tf.get_variable(
            name="src_embeddings_action", dtype=tf.float32,
            shape=[self.hp.vocab_size, self.embedding_size_action])
        self.pos_embeddings_action = tf.get_variable(
            name="pos_embeddings_action", dtype=tf.float32,
            shape=[self.num_tokens, self.embedding_size_action])


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
    return EvalDRRNModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        actions_=inputs["actions"],
        actions_len_=inputs["actions_len"],
        actions_mask_=inputs["actions_mask"],
        initializer=initializer)
