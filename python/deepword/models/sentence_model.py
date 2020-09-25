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
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.int32, [None]),
            "src2": tf.placeholder(tf.int32, [None, None]),
            "src2_len": tf.placeholder(tf.int32, [None]),
            "labels": tf.placeholder(tf.int32, [None])
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

    def add_cls_token(self, src, src_len):
        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        src_masks = tf.sequence_mask(
            src_len + 1, maxlen=self.num_tokens, dtype=tf.int32)
        return src_w_pad, src_masks

    def get_h_state(self, src, src_masks):
        batch_size = tf.shape(self.inputs["src_len"])[0] // self.turns
        with tf.variable_scope("bert-state-encoder", reuse=tf.AUTO_REUSE):
            bert_model = b_model.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
            pooled = bert_model.get_pooled_output()
            h_state = tf.reduce_sum(
                tf.reshape(
                    pooled, [-1, self.turns, self.bert_config.hidden_size]),
                axis=1)
        return h_state

    def is_semantic_same(self):
        src, src_masks = self.add_cls_token(
            self.inputs["src"], self.inputs["src_len"])
        src2, src2_masks = self.add_cls_token(
            self.inputs["src2"], self.inputs["src2_len"])

        h_state = self.get_h_state(src, src_masks)
        h_state2 = self.get_h_state(src2, src2_masks)
        diff_two_states = self.dropout(
            tf.abs(h_state - h_state2), training=(not self.is_infer))
        semantic_same = tf.squeeze(tf.layers.dense(
            diff_two_states, units=1, activation=None, use_bias=True,
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
            src_len_=inputs["src_len"],
            src2_=inputs["src2"],
            src2_len_=inputs["src2_len"],
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
            src_len_=inputs["src_len"],
            src2_=inputs["src2"],
            src2_len_=inputs["src2_len"],
            semantic_same=semantic_same,
            labels_=None,
            loss=None,
            train_op=None,
            train_summary_op=None)
