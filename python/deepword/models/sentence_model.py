import bert.modeling as b_model
import tensorflow as tf

import deepword.models.utils as dqn
from deepword.hparams import conventions
from deepword.models.dqn_model import BaseDQN
from deepword.models.export_models import SentenceModel, VecDRRNModel


class BertSentence(BaseDQN):
    """
    Use SNN to encode sentences for additive features representation learning
    """
    def __init__(self, hp, is_infer=False):
        super(BertSentence, self).__init__(hp, is_infer)
        self.num_tokens = hp.num_tokens
        self.turns = hp.num_turns
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),

            "vec_src": tf.placeholder(
                tf.float32, [None, self.bert_config.hidden_size]),
            "vec_actions": tf.placeholder(
                tf.float32, [None, self.bert_config.hidden_size]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions_repeats": tf.placeholder(tf.int32, [None]),

            "target_master": tf.placeholder(
                tf.int32, [None, None, None]),
            "same_master": tf.placeholder(
                tf.int32, [None, None, None]),
            "diff_master": tf.placeholder(
                tf.int32, [None, None, None]),
            "target_action": tf.placeholder(
                tf.int32, [None, None, None]),
            "same_action": tf.placeholder(
                tf.int32, [None, None, None]),
            "diff_action": tf.placeholder(
                tf.int32, [None, None, None])
        }
        self.bert_init_ckpt_dir = conventions.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.bert_config = b_model.BertConfig.from_json_file(
            self.bert_config_file)
        self.bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

    def get_q_actions(self):
        new_h = tf.layers.dense(
            self.inputs["vec_src"], units=self.bert_config.hidden_size,
            use_bias=True)
        h_state_expanded = tf.repeat(
            new_h, self.inputs["actions_repeats"], axis=0)

        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, self.inputs["vec_actions"]), axis=-1)
        return q_actions, new_h

    def get_pretrained_embeddings(self):
        return self.get_raw_states(self.inputs["src"])

    def add_cls_token(self, src):
        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        # Note that selected tokens are 1, padded are 0
        src_masks = tf.cast(tf.math.not_equal(src_w_pad, 0), tf.int32)
        return src_w_pad, src_masks

    def get_raw_states(self, raw_src):
        src, src_masks = self.add_cls_token(raw_src)
        with tf.variable_scope("bert-state-encoder", reuse=tf.AUTO_REUSE):
            bert_model = b_model.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
            pooled = bert_model.get_pooled_output()
        return pooled

    def get_single_state(self, src, src_masks):
        """
        Encode one trajectory

        Args:
            src: [num_turns, num_tokens]
            src_masks: [num_turns, num_tokens]

        Returns:
            encoded state for the trajectory [hidden_state,]
        """
        with tf.variable_scope("bert-state-encoder", reuse=tf.AUTO_REUSE):
            bert_model = b_model.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
            pooled = bert_model.get_pooled_output()
            h_state = tf.reduce_sum(pooled, axis=0)
        return h_state

    def get_batch_states(self, raw_src):
        """
        Encode batch of actions

        Args:
            raw_src: [batch_size, num_turns, num_tokens_per_action]

        Returns:
            encoded batch actions [batch_size, hidden_state]
        """
        n_batch = tf.shape(raw_src)[0]
        n_turns = tf.shape(raw_src)[1]
        raw_src = tf.reshape(raw_src, [n_batch * n_turns, -1])
        src, src_masks = self.add_cls_token(raw_src)
        with tf.variable_scope("bert-state-encoder", reuse=tf.AUTO_REUSE):
            bert_model = b_model.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
            pooled = bert_model.get_pooled_output()
            pooled = tf.reshape(pooled, [n_batch, n_turns, -1])
            h_actions = tf.reduce_sum(pooled, axis=1)
        return h_actions

    def is_semantic_same(self):
        batch_size = tf.shape(self.inputs["target_master"])[0]

        combined_input = tf.concat(
            [self.inputs["target_master"],
             self.inputs["same_master"],
             self.inputs["diff_master"]], axis=0)

        inc_step = tf.constant(0)
        inc_hs = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=True)

        def _dec_cond(_step, _diff):
            return tf.less(_step, tf.shape(combined_input)[0])

        def _dec_next_step(_step, _hs):
            raw_src = combined_input[_step]
            src, src_masks = self.add_cls_token(raw_src)
            h_state = self.get_single_state(src, src_masks)
            _hs = _hs.write(_step, h_state)
            return _step + 1, _hs

        results = tf.while_loop(
            cond=_dec_cond,
            body=_dec_next_step,
            loop_vars=(inc_step, inc_hs))

        h_states = results[1].stack()

        combined_actions = tf.concat(
            [self.inputs["target_action"],
             self.inputs["same_action"],
             self.inputs["diff_action"]], axis=0)
        h_actions = self.get_batch_states(combined_actions)

        features = h_states + h_actions

        target_hs = features[:batch_size]
        same_hs = features[batch_size: batch_size * 2]
        diff_hs = features[-batch_size:]

        diff_hs = tf.concat(
            [tf.abs(target_hs - same_hs), tf.abs(target_hs - diff_hs)], axis=0)
        diff_hs = self.dropout(diff_hs, training=(not self.is_infer))

        semantic_same = tf.squeeze(tf.layers.dense(
            diff_hs, units=1, activation=None, use_bias=True,
            name="snn_dense"))

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return semantic_same

    def get_snn_train_op(self, semantic_same):
        batch_size = tf.shape(self.inputs["target_master"])[0]
        labels = tf.concat(
            [tf.zeros(batch_size), tf.ones(batch_size)], axis=0)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=semantic_same)
        loss = tf.reduce_mean(losses)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                semantic_same = model.is_semantic_same()
                loss, train_op = model.get_snn_train_op(semantic_same)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return SentenceModel(
            graph=graph,
            target_master_=inputs["target_master"],
            same_master_=inputs["same_master"],
            diff_master_=inputs["diff_master"],
            target_action_=inputs["target_action"],
            same_action_=inputs["same_action"],
            diff_action_=inputs["diff_action"],
            semantic_same=semantic_same,
            loss=loss,
            train_op=train_op,
            train_summary_op=train_summary_op)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                semantic_same = model.is_semantic_same()
        return SentenceModel(
            graph=graph,
            target_master_=inputs["target_master"],
            same_master_=inputs["same_master"],
            diff_master_=inputs["diff_master"],
            target_action_=inputs["target_action"],
            same_action_=inputs["same_action"],
            diff_action_=inputs["diff_action"],
            semantic_same=semantic_same,
            loss=None,
            train_op=None,
            train_summary_op=None)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                sentence_embeddings = model.get_pretrained_embeddings()
                q_actions, new_h = model.get_q_actions()
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return VecDRRNModel(
            graph=graph,
            q_actions=q_actions,
            src_=inputs["src"],
            sentence_embeddings=sentence_embeddings,
            vec_src_=inputs["vec_src"],
            vec_actions_=inputs["vec_actions"],
            actions_repeats_=inputs["actions_repeats"],
            b_weight_=inputs["b_weight"],
            h_state=new_h,
            abs_loss=abs_loss,
            train_op=train_op,
            action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"],
            loss=loss,
            train_summary_op=train_summary_op)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                sentence_embeddings = model.get_pretrained_embeddings()
                q_actions, new_h = model.get_q_actions()
        return VecDRRNModel(
            graph=graph,
            q_actions=q_actions,
            src_=inputs["src"],
            sentence_embeddings=sentence_embeddings,
            vec_src_=inputs["vec_src"],
            vec_actions_=inputs["vec_actions"],
            actions_repeats_=inputs["actions_repeats"],
            h_state=new_h,
            b_weight_=None,
            abs_loss=None,
            train_op=None,
            action_idx_=None,
            expected_q_=None,
            loss=None,
            train_summary_op=None)
