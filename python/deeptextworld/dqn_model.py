import collections
import numpy as np
import tensorflow as tf

import deeptextworld.dqn_func as dqn
from deeptextworld import transformer as txf


class TrainDQNModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions',
         'src_', 'src_len_', 'action_idx_',
         'train_op', 'loss', 'expected_q_', 'b_weight_',
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
            if hp.use_glove_emb:
                _, glove_emb = self.init_glove(hp.glove_emb_path)
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
    return EvalDQNModel(
        graph=graph, model=model,
        q_actions=q_actions,
        src_=src_placeholder,
        src_len_=src_len_placeholder,
        initializer=initializer)


class TrainDQNGenModel(
    collections.namedtuple(
        'TrainModel',
        ('graph', 'model', 'q_actions', 'decoded_idx_infer',
         'src_', 'src_len_', 'action_idx_', 'action_idx_out_',
         'train_op', 'loss', 'expected_q_',
         'action_len_', 'b_weight_', 'temperature_',
         'train_summary_op', 'abs_loss',
         'p_gen', 'p_gen_infer',
         'beam_size_', 'use_greedy_', 'col_eos_idx', 'decoded_logits_infer',
         'loss_seq2seq', 'train_seq2seq_summary_op', 'train_seq2seq_op',
         'initializer'))):
    pass


class EvalDQNGenModel(
    collections.namedtuple(
        'EvalModel',
        ('graph', 'model', 'temperature_',
         'p_gen', 'p_gen_infer',
         'q_actions', 'decoded_idx_infer', 'src_', 'src_len_', 'action_idx_',
         'beam_size_', 'use_greedy_', 'col_eos_idx', 'decoded_logits_infer',
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


class AttnEncoderDecoderDQN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(AttnEncoderDecoderDQN, self).__init__(
            hp, src_embeddings, is_infer)

        # redefine inputs, notice the shape of action_idx
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None, None]),
            "action_idx_out": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_len": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "temperature": tf.placeholder(tf.float32, []),
            "beam_size": tf.placeholder(tf.int32, []),
            "use_greedy": tf.placeholder(tf.bool, [])
        }

        self.transformer = txf.Transformer(
            num_layers=2, d_model=128, num_heads=8, dff=128,
            input_vocab_size=self.hp.vocab_size,
            target_vocab_size=self.hp.vocab_size)

    def get_decoded_idx_infer(self):
        decoded_idx, decoded_logits, p_gen, valid_len = self.transformer.decode(
            self.inputs["src"], training=False,
            max_tar_len=self.hp.n_tokens_per_action,
            sos_id=self.hp.sos_id,
            eos_id=self.hp.eos_id,
            use_greedy=self.inputs["use_greedy"],
            beam_size=self.inputs["beam_size"],
            temperature=self.inputs["temperature"])
        return decoded_idx, tf.squeeze(valid_len, axis=-1), decoded_logits, p_gen

    def get_q_actions(self):
        q_actions, p_gen, _, _ = self.transformer(
            self.inputs["src"], tar=self.inputs["action_idx"],
            training=True)
        return q_actions, p_gen

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_2Daction(
            q_actions, self.inputs["action_idx_out"], self.inputs["expected_q"],
            self.hp.vocab_size, self.inputs["action_len"],
            self.hp.max_action_len, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    def get_seq2seq_train_op(self, q_actions):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.inputs["action_idx_out"], logits=q_actions)
        action_len_mask = tf.sequence_mask(
            self.inputs["action_len"], self.hp.n_tokens_per_action)
        bare_loss = tf.reduce_mean(tf.boolean_mask(losses, action_len_mask))
        loss = (bare_loss +
                tf.add_n(self.transformer.decoder.final_layer.losses))
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return bare_loss, loss, train_op

    def get_best_2D_q(self, q_actions):
        action_idx = tf.argmax(q_actions, axis=-1, output_type=tf.int32)
        formatted_action_idx, _ = self.format_decoded_idx(action_idx)
        return formatted_action_idx

    def format_decoded_idx(self, decoded_idx):
        """
        Format a decoded idx w/ all padding id after the first </S>.
        :param decoded_idx:
        :return:
        """
        max_action_len = tf.shape(decoded_idx)[1]
        paddings = tf.constant([[0, 0], [0, 1]])
        # make sure that every row has at least one </S>
        padded_action_idx = tf.pad(
            decoded_idx[:, :-1], paddings, constant_values=self.hp.eos_id)

        def index1d(t):
            return tf.cast(
                tf.reduce_min(tf.where(tf.equal(t, self.hp.eos_id))), tf.int32)

        col_eos_idx = tf.map_fn(index1d, padded_action_idx, dtype=tf.int32)
        mask = tf.sequence_mask(
            col_eos_idx + 1, maxlen=max_action_len, dtype=tf.int32)
        final_action_idx = tf.multiply(padded_action_idx, mask)
        return final_action_idx, col_eos_idx

    def get_acc_impl(self, action_idx):
        true_idx = self.inputs["action_idx_out"]
        nnz = tf.count_nonzero(tf.reduce_sum(action_idx - true_idx, axis=1))
        total_cnt = tf.shape(action_idx)[0]
        acc = 1. - tf.cast(nnz, tf.float32) / tf.cast(total_cnt, tf.float32)
        return acc

    def get_acc(self, q_actions):
        action_idx = self.get_best_2D_q(q_actions)
        return self.get_acc_impl(action_idx)

    def get_acc_from_decoded_idx(self, decoded_idx):
        action_idx, _ = self.format_decoded_idx(decoded_idx)
        return self.get_acc_impl(action_idx)


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


def create_train_gen_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            initializer = tf.global_variables_initializer
            inputs = model.inputs
            q_actions, p_gen = model.get_q_actions()
            (decoded_idx, col_eos_idx, decoded_logits, p_gen_infer
             ) = model.get_decoded_idx_infer()
            acc_train = model.get_acc(q_actions)
            acc_infer = model.get_acc_from_decoded_idx(decoded_idx)
            loss, train_op, abs_loss = model.get_train_op(q_actions)
            loss_summary = tf.summary.scalar("loss", loss)
            acc_train_summary = tf.summary.scalar("acc_train", acc_train)
            acc_infer_summary = tf.summary.scalar("acc_infer", acc_infer)
            train_summary_op = tf.summary.merge([loss_summary])
            (bare_loss_seq2seq, loss_seq2seq, train_seq2seq_op
             ) = model.get_seq2seq_train_op(q_actions)
            loss_summary_2 = tf.summary.scalar("loss_seq2seq", loss_seq2seq)
            bare_loss_summary = tf.summary.scalar(
                "bare_loss_seq2seq", bare_loss_seq2seq)
            train_seq2seq_summary_op = tf.summary.merge(
                [loss_summary_2, bare_loss_summary,
                 acc_train_summary, acc_infer_summary])
    return TrainDQNGenModel(
        graph=graph, model=model, q_actions=q_actions,
        decoded_idx_infer=decoded_idx,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        train_op=train_op, action_idx_=inputs["action_idx"],
        action_idx_out_=inputs["action_idx_out"],
        action_len_=inputs["action_len"],
        b_weight_=inputs["b_weight"],
        temperature_=inputs["temperature"],
        expected_q_=inputs["expected_q"], loss=loss,
        abs_loss=abs_loss,
        p_gen=p_gen, p_gen_infer=p_gen_infer,
        train_summary_op=train_summary_op,
        loss_seq2seq=loss_seq2seq,
        train_seq2seq_op=train_seq2seq_op,
        train_seq2seq_summary_op=train_seq2seq_summary_op,
        beam_size_=inputs["beam_size"],
        use_greedy_=inputs["use_greedy"],
        col_eos_idx=col_eos_idx,
        decoded_logits_infer=decoded_logits,
        initializer=initializer)


def create_eval_gen_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp, is_infer=True)
            initializer = tf.global_variables_initializer
            inputs = model.inputs
            q_actions, p_gen = model.get_q_actions()
            (decoded_idx, col_eos_idx, decoded_logits, p_gen_infer
             ) = model.get_decoded_idx_infer()
    return EvalDQNGenModel(
        graph=graph, model=model,
        q_actions=q_actions,
        decoded_idx_infer=decoded_idx,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        action_idx_=inputs["action_idx"],
        temperature_=inputs["temperature"],
        p_gen=p_gen, p_gen_infer=p_gen_infer,
        beam_size_=inputs["beam_size"],
        use_greedy_=inputs["use_greedy"],
        col_eos_idx=col_eos_idx,
        decoded_logits_infer=decoded_logits,
        initializer=initializer)
