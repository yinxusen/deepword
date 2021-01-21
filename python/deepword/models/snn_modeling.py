import bert.modeling as b_model
import tensorflow as tf

from deepword.hparams import conventions
from deepword.log import Logging
from deepword.models.models import SNNModel


class BertSNN(Logging):
    """
    Use SNN to encode sentences for additive features representation learning
    """
    def __init__(self, hp, is_infer=False):
        super(BertSNN, self).__init__()
        self.is_infer = is_infer
        self.hp = hp
        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(self.hp.learning_rate)
        self.bert_init_ckpt_dir = conventions.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.bert_config = b_model.BertConfig.from_json_file(
            self.bert_config_file)
        self.bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers
        # bert language layer is index of one layer
        self.bert_language_layer = self.hp.bert_language_layer
        assert 0 <= self.bert_language_layer < self.hp.bert_num_hidden_layers, \
            "language layer doesn't match bert layers"
        self.bert_freeze_layers = set(self.hp.bert_freeze_layers.split(","))

        self.inputs = {
            "target_src": tf.placeholder(tf.int32, [None, None]),
            "same_src": tf.placeholder(tf.int32, [None, None]),
            "diff_src": tf.placeholder(tf.int32, [None, None]),
        }

    def add_cls_token(self, src):
        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        # Note that selected tokens are 1, padded are 0
        src_masks = tf.cast(tf.math.not_equal(src_w_pad, 0), tf.int32)
        return src_w_pad, src_masks

    def get_h_state(self, raw_src):
        src, src_masks = self.add_cls_token(raw_src)
        with tf.variable_scope("bert-state-encoder", reuse=tf.AUTO_REUSE):
            bert_model = b_model.BertModel(
                config=self.bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
            all_layers = bert_model.get_all_encoder_layers()
            snn_feature_output = all_layers[self.bert_language_layer]
            with tf.variable_scope("language_pooler"):
                first_token_tensor = tf.squeeze(
                    snn_feature_output[:, 0:1, :], axis=1)
                language_feature = tf.layers.dense(
                    first_token_tensor,
                    self.bert_config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(
                        stddev=self.bert_config.initializer_range))
        return language_feature

    def is_semantic_same(self):
        target_state = self.get_h_state(self.inputs["target_src"])
        same_state = self.get_h_state(self.inputs["same_src"])
        diff_state = self.get_h_state(self.inputs["diff_src"])

        diff_two_states = tf.concat(
            [tf.abs(target_state - same_state),
             tf.abs(target_state - diff_state)], axis=0)
        diff_two_states = self.dropout(
            diff_two_states, training=(not self.is_infer))
        semantic_same = tf.squeeze(tf.layers.dense(
            diff_two_states, activation=None, units=1, use_bias=True,
            name="snn_dense"))
        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})
        return semantic_same, diff_two_states

    def get_train_op(self, semantic_same):
        batch_size = tf.shape(self.inputs["target_src"])[0]
        labels = tf.concat(
            [tf.zeros(batch_size), tf.ones(batch_size)], axis=0)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=semantic_same)
        loss = tf.reduce_mean(losses)

        var_snn = tf.trainable_variables(scope="snn_dense")
        var_bert = tf.trainable_variables(scope="bert-state-encoder")
        allowed_var_bert = list(filter(
            lambda v: all([layer_name not in v.name.split("/")
                           for layer_name in self.bert_freeze_layers]),
            var_bert))

        # allow snn_dense to be frozen
        if "snn_dense" in self.bert_freeze_layers:
            trainable_vars = allowed_var_bert
        else:
            trainable_vars = var_snn + allowed_var_bert

        self.debug("trainable vars:\n{}\n".format(
            "\n".join([v.name for v in trainable_vars])))

        train_op = self.optimizer.minimize(
            loss, global_step=self.global_step,
            var_list=trainable_vars)

        return loss, train_op

    @classmethod
    def get_train_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp)
                inputs = model.inputs
                semantic_same, _ = model.is_semantic_same()
                loss, train_op = model.get_train_op(semantic_same)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return SNNModel(
            graph=graph,
            training=True,
            target_src_=inputs["target_src"],
            same_src_=inputs["same_src"],
            diff_src_=inputs["diff_src"],
            semantic_same=semantic_same,
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
                semantic_same, _ = model.is_semantic_same()
        return SNNModel(
            graph=graph,
            training=False,
            target_src_=inputs["target_src"],
            same_src_=inputs["same_src"],
            diff_src_=inputs["diff_src"],
            semantic_same=semantic_same,
            loss=None,
            train_op=None,
            train_summary_op=None)

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        return cls.get_eval_model(hp, device_placement)
