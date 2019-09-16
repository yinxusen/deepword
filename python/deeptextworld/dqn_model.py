import collections
import tensorflow as tf

import deeptextworld.dqn_func as dqn


class TrainDQNModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions',
         'src_', 'src_len_',
         'train_op', 'loss', 'action_idx_', 'expected_q_', 'b_weight_',
         'train_summary_op', 'abs_loss',
         'initializer'))):
    pass


class EvalDQNModel(
    collections.namedtuple(
        'EvalModel',
        ('graph', 'model',
         'q_actions', 'src_', 'src_len_',
         'initializer'))):
    pass


class BaseDQN(object):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        self.is_infer = is_infer
        self.hp = hp
        if src_embeddings is None:
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

    def get_q_actions(self):
        raise NotImplementedError()

    def get_train_op(self, q_actions):
        raise NotImplementedError()


class LSTMEncoderDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(LSTMEncoderDQN, self).__init__(hp, src_embeddings, is_infer)

    def get_q_actions(self):
        inner_states = dqn.encoder_lstm(
            self.inputs["src"], self.inputs["src_len"], self.src_embeddings,
            self.hp.lstm_num_units, self.hp.lstm_num_layers)
        q_actions = dqn.decoder_dense_classification(
            inner_states[-1].c, self.hp.n_actions)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class CNNEncoderDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(CNNEncoderDQN, self).__init__(hp, src_embeddings, is_infer)
        self.filter_sizes = [3, 4, 5]
        self.num_filters = hp.num_conv_filters
        self.num_tokens = hp.num_tokens
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.5

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

        self.seg_embeddings = tf.stack(
            [tf.zeros(self.hp.embedding_size), tf.ones(self.hp.embedding_size)],
            name="seg_embeddings")

    def get_q_actions(self):
        inner_states = dqn.encoder_cnn(
            self.inputs["src"], self.src_embeddings, self.pos_embeddings,
            self.filter_sizes, self.num_filters, self.hp.embedding_size,
            self.is_infer)
        q_actions = dqn.decoder_dense_classification(inner_states,
                                                     self.hp.n_actions)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class CNNEncoderMultiLayerDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(CNNEncoderMultiLayerDQN, self).__init__(
            hp, src_embeddings, is_infer)
        self.filter_size = 3
        self.num_layers = hp.num_layers
        self.num_tokens = hp.num_tokens

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

    def get_q_actions(self):
        h_cnn = dqn.encoder_cnn_multilayers(
            self.inputs["src"], self.src_embeddings, self.pos_embeddings,
            self.num_layers, self.filter_size, self.hp.embedding_size)
        pooled = tf.reduce_max(h_cnn, axis=1)
        inner_states = tf.reshape(pooled, [-1, self.hp.embedding_size])
        q_actions = dqn.decoder_dense_classification(inner_states,
                                                     self.hp.n_actions)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class MultiChannelCNNEncoderDQN(CNNEncoderDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(MultiChannelCNNEncoderDQN, self).__init__(
            hp, src_embeddings, is_infer)

        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None, None]),
            "src_len": tf.placeholder(tf.float32, [None, None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None])
        }

    def get_q_actions(self):
        inner_states = dqn.encoder_cnn_multichannels(
            self.inputs["src"], self.inputs["src_len"], self.src_embeddings,
            self.filter_sizes, self.num_filters, self.hp.embedding_size,
            self.hp.num_channels)
        q_actions = dqn.decoder_dense_classification(inner_states,
                                                     self.hp.n_actions)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


def create_train_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp)
        initializer = tf.global_variables_initializer
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
    return TrainDQNModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=src_placeholder,
        src_len_=src_len_placeholder,
        train_op=train_op, action_idx_=action_idx_placeholder,
        expected_q_=expected_q_placeholder,
        b_weight_=b_weight_placeholder,
        loss=loss,
        train_summary_op=train_summary_op,
        abs_loss=abs_loss,
        initializer=initializer)


def create_eval_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp, is_infer=True)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        src_placeholder = inputs["src"]
        src_len_placeholder = inputs["src_len"]
        q_actions = model.get_q_actions()
        _ = model.get_train_op(q_actions)
    return EvalDQNModel(
        graph=graph, model=model,
        q_actions=q_actions,
        src_=src_placeholder,
        src_len_=src_len_placeholder,
        initializer=initializer)


class TrainDQNGenModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions',
         'src_', 'src_len_',
         'train_op', 'loss', 'action_idx_', 'expected_q_',
         'action_len_', 'b_weight_',
         'train_summary_op', 'abs_loss',
         'initializer'))):
    pass


class EvalDQNGenModel(
    collections.namedtuple(
        'EvalModel',
        ('graph', 'model',
         'q_actions', 'src_', 'src_len_',
         'initializer'))):
    pass


class LSTMEncoderDecoderDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, tgt_embeddings=None,
                 is_infer=False):
        super(LSTMEncoderDecoderDQN, self).__init__(
            hp, src_embeddings, is_infer)

        # redefine inputs, notice the shape of action_idx
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_len": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None])
        }

        if tgt_embeddings is None:
            self.tgt_embeddings = tf.get_variable(
                name="tgt_embeddings", dtype=tf.float32,
                shape=[self.hp.tgt_vocab_size, self.hp.embedding_size])
        else:
            self.tgt_embeddings = tgt_embeddings


    def get_q_actions(self):
        inner_states = dqn.encoder_lstm(
            self.inputs["src"], self.inputs["src_len"], self.src_embeddings,
            self.hp.lstm_num_units, self.hp.lstm_num_layers)
        q_actions = dqn.decoder_fix_len_lstm(
            inner_states, self.hp.tgt_vocab_size, self.tgt_embeddings,
            self.hp.lstm_num_units, self.hp.lstm_num_layers,
            self.hp.tgt_sos_id, self.hp.tgt_eos_id, self.hp.max_action_len)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_2Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.tgt_vocab_size, self.inputs["action_len"],
            self.hp.max_action_len, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class CNNEncoderDecoderDQN(CNNEncoderDQN):
    def __init__(
            self, hp, src_embeddings=None, tgt_embeddings=None, is_infer=False):
        super(CNNEncoderDecoderDQN, self).__init__(hp, src_embeddings, is_infer)

        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_len": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None])
        }

        if tgt_embeddings is None:
            self.tgt_embeddings = tf.get_variable(
                name="tgt_embeddings", dtype=tf.float32,
                shape=[self.hp.tgt_vocab_size, self.hp.embedding_size])
        else:
            self.tgt_embeddings = tgt_embeddings

    def get_q_actions(self):
        inner_states = dqn.encoder_cnn_block(
            self.inputs["src"], self.src_embeddings, self.pos_embeddings,
            self.filter_sizes, self.num_filters, self.hp.embedding_size)
        q_actions = dqn.decoder_fix_len_cnn(
            inner_states, self.tgt_embeddings, self.pos_embeddings,
            self.hp.tgt_vocab_size, self.hp.embedding_size,
            self.filter_sizes, self.num_filters, self.hp.tgt_sos_id,
            self.hp.max_action_len)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_2Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.tgt_vocab_size, self.inputs["action_len"],
            self.hp.max_action_len, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class CNNEDMultiLayerDQN(BaseDQN):
    def __init__(
            self, hp, src_embeddings=None, tgt_embeddings=None, is_infer=False):
        super(CNNEDMultiLayerDQN, self).__init__(hp, src_embeddings, is_infer)

        self.filter_size = 3
        self.num_layers = hp.num_layers
        self.num_tokens = hp.num_tokens

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_len": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None])
        }

        if tgt_embeddings is None:
            self.tgt_embeddings = tf.get_variable(
                name="tgt_embeddings", dtype=tf.float32,
                shape=[self.hp.tgt_vocab_size, self.hp.embedding_size])
        else:
            self.tgt_embeddings = tgt_embeddings

    def get_q_actions(self):
        inner_states = dqn.encoder_cnn_multilayers(
            self.inputs["src"], self.src_embeddings, self.pos_embeddings,
            self.num_layers, self.filter_size, self.hp.embedding_size)
        q_actions = dqn.decoder_fix_len_cnn_multilayers(
            inner_states, self.tgt_embeddings, self.pos_embeddings,
            self.hp.tgt_vocab_size, self.hp.embedding_size,
            self.num_layers, self.filter_size, self.hp.tgt_sos_id,
            self.hp.max_action_len)
        return q_actions

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_2Daction(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.tgt_vocab_size, self.inputs["action_len"],
            self.hp.max_action_len, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


def create_train_gen_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        src_placeholder = inputs["src"]
        src_len_placeholder = inputs["src_len"]
        action_idx_placeholder = inputs["action_idx"]
        expected_q_placeholder = inputs["expected_q"]
        action_len_placeholder = inputs["action_len"]
        b_weight_placeholder = inputs["b_weight"]
        q_actions = model.get_q_actions()
        loss, train_op, abs_loss = model.get_train_op(q_actions)
        loss_summary = tf.summary.scalar("loss", loss)
        train_summary_op = tf.summary.merge([loss_summary])
    return TrainDQNGenModel(
        graph=graph, model=model, q_actions=q_actions,
        src_=src_placeholder,
        src_len_=src_len_placeholder,
        train_op=train_op, action_idx_=action_idx_placeholder,
        action_len_=action_len_placeholder,
        b_weight_=b_weight_placeholder,
        expected_q_=expected_q_placeholder, loss=loss,
        abs_loss=abs_loss,
        train_summary_op=train_summary_op,
        initializer=initializer)


def create_eval_gen_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp, is_infer=True)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        src_placeholder = inputs["src"]
        src_len_placeholder = inputs["src_len"]
        q_actions = model.get_q_actions()
        _ = model.get_train_op(q_actions)
    return EvalDQNGenModel(
        graph=graph, model=model,
        q_actions=q_actions,
        src_=src_placeholder,
        src_len_=src_len_placeholder,
        initializer=initializer)
