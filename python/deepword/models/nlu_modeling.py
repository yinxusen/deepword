import albert.modeling as ab_model
import bert.modeling as b_model
import tensorflow as tf

from deepword.hparams import conventions
from deepword.models.dqn_modeling import BaseDQN
from deepword.models.models import NLUModel


class BertNLU(BaseDQN):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode,
           has paddings, [CLS], and [SEP] prepared
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
          actions_mask: a 0-1 vector of size |actions|, using 0 to eliminate
                        some actions for a certain state.
        :param hp:
        :param is_infer:
        """
        super(BertNLU, self).__init__(hp, is_infer)
        self.num_tokens = hp.num_tokens
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.int32, [None]),
            "seg_tj_action": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "swag_labels": tf.placeholder(tf.int32, [None])
        }
        self.bert_init_ckpt_dir = conventions.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.bert_freeze_layers = set([
            int(x) for x in self.hp.bert_freeze_layers.split(",")])

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        seg_tj_action = self.inputs["seg_tj_action"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        bert_config = b_model.BertConfig.from_json_file(self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

        with tf.variable_scope("bert-state-encoder"):
            bert_model = b_model.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks,
                token_type_ids=seg_tj_action)
            pooled = bert_model.get_pooled_output()

        with tf.variable_scope("q-encoder"):
            output = self.dropout(pooled, training=(not self.is_infer))
            q_actions = tf.layers.dense(output, units=1, use_bias=True)[:, 0]

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return q_actions

    def get_classification_train_op(self, q_actions):
        """
        q_actions: [batch_size, 1]
        in this case, when we want to compute classification error, we need
        the batch_size = src batch size * num classes
        which means that number of classes for each src should be equal
        :param q_actions:
        :return:
        """
        swag_batch_size = tf.shape(self.inputs["swag_labels"])[0]
        q_actions = tf.reshape(q_actions, [swag_batch_size, -1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=q_actions, labels=self.inputs["swag_labels"])
        loss = tf.reduce_mean(losses)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op

    def get_train_op(self, q_actions):
        losses = tf.squared_difference(self.inputs["expected_q"], q_actions)
        loss = tf.reduce_mean(losses)

        var_q_encoder = tf.trainable_variables(scope="q-encoder")
        var_bert = tf.trainable_variables(scope="bert-state-encoder")
        allowed_var_bert = list(filter(
            lambda v: all([layer_name not in v.name.split("/")
                           for layer_name in self.bert_freeze_layers]),
            var_bert))
        trainable_vars = var_q_encoder + allowed_var_bert

        self.debug("trainable vars: {}".format(trainable_vars))

        train_op = self.optimizer.minimize(
            loss, global_step=self.global_step,
            var_list=trainable_vars)

        return loss, train_op

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return create_train_bert_nlu_model(cls, hp, device_placement)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        return create_eval_bert_nlu_model(cls, hp, device_placement)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        return cls.get_train_student_model(hp, device_placement)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        return cls.get_eval_student_model(hp, device_placement)


class AlbertNLU(BertNLU):
    def __init__(self, hp, is_infer=False):
        """
        inputs:
          src: source sentences to encode,
           has paddings, [CLS], and [SEP] prepared
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
          actions_mask: a 0-1 vector of size |actions|, using 0 to eliminate
                        some actions for a certain state.
        :param hp:
        :param is_infer:
        """
        super(AlbertNLU, self).__init__(hp, is_infer)
        self.bert_init_ckpt_dir = conventions.albert_ckpt_dir
        self.bert_config_file = "{}/albert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/model.ckpt-best".format(
            self.bert_init_ckpt_dir)

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        seg_tj_action = self.inputs["seg_tj_action"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.num_tokens, dtype=tf.int32)

        bert_config = ab_model.AlbertConfig.from_json_file(
            self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

        with tf.variable_scope("bert-state-encoder"):
            bert_model = ab_model.AlbertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks,
                token_type_ids=seg_tj_action)
            pooled = bert_model.get_pooled_output()
            output = self.dropout(pooled, training=(not self.is_infer))
            q_actions = tf.layers.dense(output, units=1, use_bias=True)[:, 0]

        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "bert-state-encoder/bert/"})

        return q_actions


def create_train_bert_nlu_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            inputs = model.inputs
            q_actions = model.get_q_actions()
            loss, train_op = model.get_train_op(q_actions)
            classification_loss, classification_train_op = \
                model.get_classification_train_op(q_actions)
            loss_summary = tf.summary.scalar("loss", loss)
            classification_loss_summary = tf.summary.scalar(
                "classification_loss", classification_loss)
            train_summary_op = tf.summary.merge([loss_summary])
            classification_train_summary_op = tf.summary.merge(
                [classification_loss_summary])
    return NLUModel(
        graph=graph,
        q_actions=q_actions,
        src_seg_=None,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        swag_labels_=inputs["swag_labels"],
        loss=loss,
        train_op=train_op,
        classification_loss=classification_loss,
        classification_train_op=classification_train_op,
        train_summary_op=train_summary_op,
        classification_train_summary_op=classification_train_summary_op,
        expected_q_=inputs["expected_q"],
        action_idx_=inputs["action_idx"],
        abs_loss=None,
        seg_tj_action_=inputs["seg_tj_action"],
        h_state=None,
        b_weight_=None)


def create_eval_bert_nlu_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp, is_infer=True)
            inputs = model.inputs
            q_actions = model.get_q_actions()
    return NLUModel(
        graph=graph,
        q_actions=q_actions,
        src_seg_=None,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        swag_labels_=None,
        loss=None,
        train_op=None,
        classification_loss=None,
        classification_train_op=None,
        train_summary_op=None,
        classification_train_summary_op=None,
        expected_q_=inputs["expected_q"],
        action_idx_=inputs["action_idx"],
        abs_loss=None,
        seg_tj_action_=inputs["seg_tj_action"],
        h_state=None,
        b_weight_=None)
