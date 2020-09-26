import bert.modeling as b_model
import tensorflow as tf

from deepword.hparams import conventions
from deepword.models.dqn_model import BaseDQN
from deepword.models.export_models import SentenceModel


class BertSentence(BaseDQN):
    """
    Use SNN to encode sentences for additive features representation learning
    """
    def __init__(self, hp, is_infer=False):
        super(BertSentence, self).__init__(hp, is_infer)
        self.num_tokens = hp.num_tokens
        self.turns = hp.num_turns
        self.inputs = {
            "src": tf.placeholder(
                tf.int32, [self.hp.batch_size * 2, None, None]),
            "src2": tf.placeholder(
                tf.int32, [self.hp.batch_size * 2, None, None]),
            "labels": tf.placeholder(tf.float32, [self.hp.batch_size * 2])
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
        raise NotImplementedError()

    def add_cls_token(self, src):
        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        # Note that selected tokens are 1, padded are 0
        src_masks = tf.cast(tf.math.not_equal(src_w_pad, 0), tf.int32)
        return src_w_pad, src_masks

    def get_h_state(self, src, src_masks):
        with tf.variable_scope("bert-state-encoder", reuse=tf.AUTO_REUSE):
            bert_model = b_model.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
            pooled = bert_model.get_pooled_output()
            h_state = tf.reduce_sum(pooled, axis=0)
        return h_state

    def is_semantic_same(self):
        inc_step = tf.constant(0)
        inc_diff = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=True)

        def _dec_cond(_step, _diff):
            return tf.less(_step, self.hp.batch_size * 2)

        def _dec_next_step(_step, _diff):
            raw_src = self.inputs["src"][_step]
            raw_src2 = self.inputs["src2"][_step]
            src, src_masks = self.add_cls_token(raw_src)
            src2, src2_masks = self.add_cls_token(raw_src2)
            h_state = self.get_h_state(src, src_masks)
            h_state2 = self.get_h_state(src2, src2_masks)
            diff_two_states = self.dropout(
                tf.abs(h_state - h_state2), training=(not self.is_infer))
            _diff = _diff.write(_step, diff_two_states)
            return _step + 1, _diff

        results = tf.while_loop(
            cond=_dec_cond,
            body=_dec_next_step,
            loop_vars=(inc_step, inc_diff))
        processed = results[1].stack()

        batch_diff_two_states = tf.stack(processed, axis=0)
        semantic_same = tf.squeeze(tf.layers.dense(
            batch_diff_two_states, units=1, activation=None, use_bias=True,
            name="snn_dense"))

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return semantic_same

    def get_train_op(self, semantic_same):
        labels = self.inputs["labels"]
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=semantic_same)
        loss = tf.reduce_mean(losses)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        return cls.get_eval_model(hp, device_placement)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                semantic_same = model.is_semantic_same()
                loss, train_op = model.get_train_op(semantic_same)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return SentenceModel(
            graph=graph,
            src_=inputs["src"],
            src2_=inputs["src2"],
            semantic_same=semantic_same,
            labels_=inputs["labels"],
            loss=loss,
            train_op=train_op,
            train_summary_op=train_summary_op)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                semantic_same = model.is_semantic_same()
        return SentenceModel(
            graph=graph,
            src_=inputs["src"],
            src2_=inputs["src2"],
            semantic_same=semantic_same,
            labels_=None,
            loss=None,
            train_op=None,
            train_summary_op=None)
