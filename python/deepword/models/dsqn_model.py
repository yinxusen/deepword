import tensorflow as tf

import deepword.models.transformer as txf
import deepword.models.utils as dqn
from deepword.models.dqn_model import BaseDQN
from deepword.models.export_models import DSQNModel
from deepword.models.export_models import DSQNZorkModel


class CnnDSQN(BaseDQN):
    """
    DSQN that uses CNN as the trajectory encoder
    """
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(CnnDSQN, self).__init__(hp, src_embeddings, is_infer)
        self.filter_sizes = [3, 4, 5]
        self.num_filters = hp.num_conv_filters
        self.num_tokens = hp.num_tokens
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.5

        self.pos_embeddings = tf.get_variable(
            name="pos_embeddings", dtype=tf.float32,
            shape=[self.num_tokens, self.hp.embedding_size])

        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_seg": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(
                tf.int32, [None, self.hp.n_tokens_per_action]),
            "actions_repeats": tf.placeholder(tf.int32, [None]),
            "actions_len": tf.placeholder(tf.float32, [None]),
            "snn_src": tf.placeholder(tf.int32, [None, None]),
            "snn_src_len": tf.placeholder(tf.float32, [None]),
            "snn_src2": tf.placeholder(tf.int32, [None, None]),
            "snn_src2_len": tf.placeholder(tf.float32, [None]),
            "labels": tf.placeholder(tf.float32, [None])
        }

        self.wt = tf.layers.Dense(
            units=self.hp.hidden_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.wt_var = tf.layers.Dense(
            units=self.hp.hidden_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        h_state = self.get_h_state(self.inputs["src"])
        new_h = self.wt(h_state)
        new_h_var = self.wt_var(h_state)
        h_state_expanded = dqn.repeat(
            new_h + new_h_var, self.inputs["actions_repeats"])

        with tf.variable_scope("drrn-action-encoder", reuse=False):
            h_actions = dqn.encoder_lstm(
                self.inputs["actions"],
                self.inputs["actions_len"],
                self.src_embeddings,
                num_units=self.hp.hidden_state_size,
                num_layers=self.hp.lstm_num_layers)[-1].h

        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions, new_h

    def get_h_state(self, src):
        with tf.variable_scope("drrn-encoder", reuse=tf.AUTO_REUSE):
            h_state = dqn.encoder_cnn(
                src,
                self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
        return h_state

    def is_semantic_same(self):
        h_state = self.get_h_state(self.inputs["snn_src"])
        h_state2 = self.get_h_state(self.inputs["snn_src2"])
        new_h_var = self.wt_var(h_state)
        new_h_var2 = self.wt_var(h_state2)
        diff_two_states = tf.abs(new_h_var - new_h_var2)
        semantic_same = tf.squeeze(tf.layers.dense(
            diff_two_states, activation=None, units=1, use_bias=True,
            name="snn_dense"))
        return semantic_same, diff_two_states

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    def get_snn_train_op(self, semantic_same):
        labels = self.inputs["labels"]
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=semantic_same)
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

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
                semantic_same, h_states_diff = model.is_semantic_same()
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                snn_loss, snn_train_op = model.get_snn_train_op(semantic_same)
                (weighted_loss, merged_train_op, s1, s2
                 ) = model.get_merged_train_op(loss, snn_loss)
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
        return DSQNModel(
            graph=graph,
            q_actions=q_actions,
            semantic_same=semantic_same,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            actions_=inputs["actions"],
            actions_len_=inputs["actions_len"],
            snn_src_=inputs["snn_src"],
            snn_src_len_=inputs["snn_src_len"],
            snn_src2_=inputs["snn_src2"],
            snn_src2_len_=inputs["snn_src2_len"],
            actions_repeats_=inputs["actions_repeats"],
            src_seg_=inputs["src_seg"],
            labels_=inputs["labels"],
            b_weight_=inputs["b_weight"],
            abs_loss=abs_loss,
            train_op=train_op,
            action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"],
            loss=loss,
            snn_train_op=snn_train_op,
            weighted_loss=weighted_loss,
            snn_loss=snn_loss,
            merged_train_op=merged_train_op,
            train_summary_op=train_summary_op,
            snn_train_summary_op=snn_train_summary_op,
            weighted_train_summary_op=weighted_train_summary_op,
            h_states_diff=h_states_diff,
            h_state=new_h)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
                semantic_same, h_states_diff = model.is_semantic_same()
        return DSQNModel(
            graph=graph, q_actions=q_actions, semantic_same=semantic_same,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            actions_=inputs["actions"],
            actions_len_=inputs["actions_len"],
            snn_src_=inputs["snn_src"],
            snn_src_len_=inputs["snn_src_len"],
            snn_src2_=inputs["snn_src2"],
            snn_src2_len_=inputs["snn_src2_len"],
            actions_repeats_=inputs["actions_repeats"],
            src_seg_=inputs["src_seg"],
            labels_=inputs["labels"],
            b_weight_=inputs["b_weight"],
            abs_loss=None,
            train_op=None, action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"], loss=None,
            snn_train_op=None,
            weighted_loss=None,
            snn_loss=None,
            merged_train_op=None,
            train_summary_op=None,
            snn_train_summary_op=None,
            weighted_train_summary_op=None,
            h_states_diff=h_states_diff,
            h_state=new_h)


class CnnZorkDSQN(CnnDSQN):
    """
    DSQN for Zork
    """
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(CnnZorkDSQN, self).__init__(hp, src_embeddings, is_infer)

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action_v2(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.hp.n_actions, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
                semantic_same, h_states_diff = model.is_semantic_same()
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                snn_loss, snn_train_op = model.get_snn_train_op(semantic_same)
                (weighted_loss, merged_train_op, s1, s2
                 ) = model.get_merged_train_op(loss, snn_loss)
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
        return DSQNZorkModel(
            graph=graph,
            q_actions=q_actions,
            semantic_same=semantic_same,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            snn_src_=inputs["snn_src"],
            snn_src_len_=inputs["snn_src_len"],
            snn_src2_=inputs["snn_src2"],
            snn_src2_len_=inputs["snn_src2_len"],
            src_seg_=inputs["src_seg"],
            labels_=inputs["labels"],
            b_weight_=inputs["b_weight"],
            abs_loss=abs_loss,
            train_op=train_op,
            action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"],
            loss=loss,
            snn_train_op=snn_train_op,
            weighted_loss=weighted_loss,
            snn_loss=snn_loss,
            merged_train_op=merged_train_op,
            train_summary_op=train_summary_op,
            snn_train_summary_op=snn_train_summary_op,
            weighted_train_summary_op=weighted_train_summary_op,
            h_states_diff=h_states_diff,
            h_state=new_h)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
                semantic_same, h_states_diff = model.is_semantic_same()
        return DSQNZorkModel(
            graph=graph, q_actions=q_actions, semantic_same=semantic_same,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            snn_src_=inputs["snn_src"],
            snn_src_len_=inputs["snn_src_len"],
            snn_src2_=inputs["snn_src2"],
            snn_src2_len_=inputs["snn_src2_len"],
            src_seg_=inputs["src_seg"],
            labels_=inputs["labels"],
            b_weight_=inputs["b_weight"],
            abs_loss=None,
            train_op=None, action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"], loss=None,
            snn_train_op=None,
            weighted_loss=None,
            snn_loss=None,
            merged_train_op=None,
            train_summary_op=None,
            snn_train_summary_op=None,
            weighted_train_summary_op=None,
            h_states_diff=h_states_diff,
            h_state=new_h)


class TransformerDSQN(CnnDSQN):
    """
    DSQN that uses transformer as the trajectory encoder
    """
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        super(TransformerDSQN, self).__init__(hp, src_embeddings, is_infer)
        self.d_model = 128
        self.attn_encoder = txf.Encoder(
            num_layers=1, d_model=self.d_model, num_heads=8, dff=256,
            input_vocab_size=self.hp.vocab_size)

    def get_h_state(self, src):
        with tf.variable_scope("drrn-encoder", reuse=tf.AUTO_REUSE):
            padding_mask = txf.create_padding_mask(src)
            inner_state = self.attn_encoder(
                src, training=(not self.is_infer), mask=padding_mask,
                x_seg=None)
            pooled = tf.reduce_max(inner_state, axis=1)
            h_state = tf.reshape(pooled, [-1, self.d_model])
        return h_state
