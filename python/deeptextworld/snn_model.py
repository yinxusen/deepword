import collections
import tensorflow as tf
from bert import modeling

import deeptextworld.dqn_func as dqn
from deeptextworld.dqn_model import BaseDQN, CNNEncoderDQN


class TrainSNNModel(
    collections.namedtuple(
        'TrainSNNModel',
        ('graph', 'model', 'pred','train_op', 'loss', 'train_summary_op',
         'src_', 'src_len_', 'src2_', 'src2_len_', 'labels_', 'initializer'))):
    pass


class EvalSNNModel(
    collections.namedtuple(
        'EvalSNNModel',
        ('graph', 'model', 'pred',
         'src_', 'src_len_', 'src2_', 'src2_len_', 'initializer'))):
    pass


class CNNEncoderSNN(object):
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
            "src2": tf.placeholder(tf.int32, [None, None]),
            "src2_len": tf.placeholder(tf.float32, [None]),
            "labels": tf.placeholder(tf.float32, [None])
        }

        self.filter_sizes = [3, 4, 5]
        self.num_filters = hp.num_conv_filters
        self.num_tokens = hp.num_tokens
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.5

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

    def get_pred(self):
        with tf.variable_scope("cnn-encoder", reuse=tf.AUTO_REUSE):
            h_state = dqn.encoder_cnn(
                self.inputs["src"], self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            new_h = dqn.decoder_dense_classification(h_state, 32)
            h_state2 = dqn.encoder_cnn(
                self.inputs["src2"], self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            new_h2 = dqn.decoder_dense_classification(h_state2, 32)

        diff_two_states = tf.abs(new_h - new_h2)
        pred = tf.squeeze(tf.layers.dense(
            diff_two_states, activation=tf.nn.sigmoid, units=1, use_bias=True))
        return pred

    def get_train_op(self, pred):
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
        pred = model.get_pred()
        loss, train_op = model.get_train_op(pred)
        loss_summary = tf.summary.scalar("loss", loss)
        train_summary_op = tf.summary.merge([loss_summary])
    return TrainSNNModel(
        graph=graph, model=model, pred=pred,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        src2_=inputs["src2"],
        src2_len_=inputs["src2_len"],
        labels_=inputs["labels"],
        train_op=train_op, loss=loss,
        train_summary_op=train_summary_op,
        initializer=initializer)


def create_eval_model(model_creator, hp):
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(hp, is_infer=True)
        initializer = tf.global_variables_initializer
        inputs = model.inputs
        pred = model.get_pred()
        # still need to put them here, otherwise the loaded model could not be trained
        _ = model.get_train_op(pred)
    return EvalSNNModel(
        graph=graph, model=model, pred=pred,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        src2_=inputs["src2"],
        src2_len_=inputs["src2_len"],
        initializer=initializer)
