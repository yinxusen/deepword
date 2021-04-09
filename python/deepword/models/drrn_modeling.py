import bert.modeling as b_model
import tensorflow as tf

import deepword.models.transformer as txf
import deepword.models.utils as dqn
from deepword.hparams import conventions
from deepword.models.dqn_modeling import BaseDQN
from deepword.models.models import DRRNModel


class CnnDRRN(BaseDQN):
    def __init__(self, hp, src_embeddings=None, is_infer=False):
        """
        inputs:
          src: source sentences to encode
          src_len: length of source sentences
          action_idx: the action chose to run
          expected_q: E(q) computed from the iterative equation of DQN
          actions: all possible actions
          actions_len: length of actions
        :param hp:
        :param is_infer:
        """
        super(CnnDRRN, self).__init__(hp, src_embeddings, is_infer)
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
            "src_len": tf.placeholder(tf.int32, [None]),
            "action_idx": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "actions": tf.placeholder(
                tf.int32, [None, self.hp.n_tokens_per_action]),
            "actions_repeats": tf.placeholder(tf.int32, [None]),
            "actions_len": tf.placeholder(tf.int32, [None])
        }

    def get_q_actions(self):
        with tf.variable_scope("drrn-encoder", reuse=False):
            h_state = dqn.encoder_cnn(
                self.inputs["src"], self.inputs["src_len"],
                self.src_embeddings, self.pos_embeddings,
                self.filter_sizes, self.num_filters, self.hp.embedding_size,
                self.is_infer)
            new_h = tf.layers.dense(
                h_state, units=self.hp.hidden_state_size, use_bias=True)
            h_state_expanded = tf.repeat(
                new_h, self.inputs["actions_repeats"], axis=0)

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

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])
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
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return DRRNModel(
            graph=graph,
            training=True,
            q_actions=q_actions,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            actions_=inputs["actions"],
            actions_len_=inputs["actions_len"],
            actions_repeats_=inputs["actions_repeats"],
            b_weight_=inputs["b_weight"],
            h_state=new_h,
            abs_loss=abs_loss,
            train_op=train_op,
            action_idx_=inputs["action_idx"],
            expected_q_=inputs["expected_q"],
            loss=loss,
            train_summary_op=train_summary_op,
            src_seg_=None)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                q_actions, new_h = model.get_q_actions()
        return DRRNModel(
            graph=graph,
            training=False,
            q_actions=q_actions,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            actions_=inputs["actions"],
            actions_len_=inputs["actions_len"],
            actions_repeats_=inputs["actions_repeats"],
            h_state=new_h,
            b_weight_=None,
            abs_loss=None,
            train_op=None,
            action_idx_=None,
            expected_q_=None,
            loss=None,
            train_summary_op=None,
            src_seg_=None)

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_eval_student_model(cls, hp, device_placement):
        return cls.get_eval_model(hp, device_placement)


class TransformerDRRN(CnnDRRN):
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
        super(TransformerDRRN, self).__init__(hp, src_embeddings, is_infer)

    def get_q_actions(self):
        with tf.variable_scope("drrn-encoder", reuse=False):
            attn_encoder = txf.Encoder(
                num_layers=1, d_model=128, num_heads=8, dff=256,
                input_vocab_size=self.hp.vocab_size)
            padding_mask = txf.create_padding_mask(self.inputs["src"])
            inner_state = attn_encoder(
                self.inputs["src"],
                training=(not self.is_infer), mask=padding_mask, x_seg=None)
            pooled = tf.reduce_max(inner_state, axis=1)
            h_state = tf.reshape(pooled, [-1, 128])
            new_h = tf.layers.dense(
                h_state, units=self.hp.hidden_state_size, use_bias=True)
            h_state_expanded = tf.repeat(
                new_h, self.inputs["actions_repeats"], axis=0)

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


class BertDRRN(CnnDRRN):
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
        :param is_infer:
        """
        super(BertDRRN, self).__init__(hp, src_embeddings, is_infer)
        self.bert_init_ckpt_dir = conventions.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.h_state_size = self.hp.hidden_state_size
        self.wt = tf.layers.Dense(
            units=self.h_state_size, activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        src_masks = tf.sequence_mask(
            src_len, maxlen=self.hp.num_tokens, dtype=tf.int32)
        bert_config = b_model.BertConfig.from_json_file(self.bert_config_file)
        bert_config.num_hidden_layers = self.hp.bert_num_hidden_layers

        # padding the [CLS] in the beginning
        paddings = tf.constant([[0, 0], [1, 0]])
        src_w_pad = tf.pad(
            src, paddings=paddings, mode="CONSTANT",
            constant_values=self.hp.cls_val_id)
        src_masks_w_pad = tf.pad(
            src_masks, paddings=paddings, mode="CONSTANT",
            constant_values=1)

        with tf.variable_scope("tj-bert-encoder"):
            bert_model = b_model.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src_w_pad, input_mask=src_masks_w_pad)
        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "tj-bert-encoder/bert/"})
        h_state = self.wt(bert_model.pooled_output)
        with tf.variable_scope("drrn-action-encoder", reuse=False):
            h_actions = dqn.encoder_lstm(
                self.inputs["actions"],
                self.inputs["actions_len"],
                self.src_embeddings,
                num_units=self.hp.hidden_state_size,
                num_layers=self.hp.lstm_num_layers)[-1].h

        h_state_expanded = tf.repeat(
            h_state, self.inputs["actions_repeats"], axis=0)
        q_actions = tf.reduce_sum(
            tf.multiply(h_state_expanded, h_actions), axis=-1)
        return q_actions, h_state


class PseudoSeq2SeqDRRN(CnnDRRN):
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
        :param is_infer:
        """
        super(PseudoSeq2SeqDRRN, self).__init__(hp, src_embeddings, is_infer)
        self.bert_init_ckpt_dir = conventions.bert_ckpt_dir
        self.bert_config_file = "{}/bert_config.json".format(
            self.bert_init_ckpt_dir)
        self.bert_ckpt_file = "{}/bert_model.ckpt".format(
            self.bert_init_ckpt_dir)
        self.bert_decoder = txf.BertDecoder(
            num_layers=2, d_model=768, num_heads=12, dff=3072,
            tgt_vocab_size=self.hp.vocab_size, with_pointer=True)

    @classmethod
    def get_tgt_idx_pair(cls, tgt_matrix, tgt_len, sos_id, eos_id):
        """
        Create action index pair for seq2seq training.
        Given action index, e.g. [1, 2, 3, 4, pad, pad, pad, pad],
        with 0 as sos_id, and -1 as eos_id,
        we create training pair: [0, 1, 2, 3, 4, pad, pad, pad]
        as the input sentence, and [1, 2, 3, 4, -1, pad, pad, pad]
        as the output sentence.

        Notice that we remove the final pad to keep the action length unchanged.
        Notice 2. pad should be indexed as 0.

        Args:
            tgt_matrix: np array of action index of N * K, there are N
            actions, and each of them has a length of K (with paddings).
            tgt_len: length of each action (remove paddings).
            sos_id: int
            eos_id: int

        Returns
            action index as input, action index as output, new action len
        """
        n_row = tf.shape(tgt_matrix)[0]
        n_col = tf.shape(tgt_matrix)[1]
        tgt_in = tf.concat(
            [tf.fill([n_row, 1], sos_id), tgt_matrix[:, :-1]], axis=1)
        new_action_len = tf.reduce_min(
            [tgt_len + 1, tf.zeros_like(tgt_len) + n_col], axis=0)
        idx = tf.stack([tf.range(n_row), new_action_len - 1], axis=-1)
        mask = 1 - tf.scatter_nd(
            indices=idx,
            updates=tf.fill([n_row], 1),
            shape=tf.shape(tgt_matrix))
        eos_paddings = tf.scatter_nd(
            indices=idx,
            updates=tf.fill([n_row], eos_id),
            shape=tf.shape(tgt_matrix))
        tgt_out = tf.multiply(tgt_matrix, mask) + eos_paddings
        return tgt_in, tgt_out, new_action_len

    def get_q_actions(self):
        src = self.inputs["src"]
        src_len = self.inputs["src_len"]
        max_len = tf.reduce_max([tf.reduce_max(src_len), tf.shape(src)[1]])
        src_masks = tf.sequence_mask(
            src_len, maxlen=max_len, dtype=tf.int32)
        bert_config = b_model.BertConfig.from_json_file(self.bert_config_file)

        with tf.variable_scope("tj-bert-encoder"):
            bert_model = b_model.BertModel(
                config=bert_config, is_training=(not self.is_infer),
                input_ids=src, input_mask=src_masks)
        # initialize bert from checkpoint file
        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "tj-bert-encoder/bert/"})
        h_state = bert_model.get_sequence_output()

        with tf.variable_scope("drrn-action-encoder", reuse=False):
            h_state_expanded = tf.repeat(
                h_state, self.inputs["actions_repeats"], axis=0)
            tgt_in, tgt_out, tgt_len = self.get_tgt_idx_pair(
                self.inputs["actions"],
                self.inputs["actions_len"],
                self.hp.sos_id,
                self.hp.eos_id)
            inp = tf.repeat(
                self.inputs["src"], self.inputs["actions_repeats"], axis=0)
            enc_output = h_state_expanded
            dec_padding_mask = txf.create_padding_mask(inp)
            look_ahead_mask = txf.create_decode_masks(tgt_in)
            final_output, p_gen, _, _ = self.bert_decoder(
                tgt_in, inp, enc_output, not self.is_infer, look_ahead_mask,
                dec_padding_mask, copy_mask=None)

            batch_size = tf.shape(tgt_out)[0]
            tgt_seq_len = tf.shape(tgt_out)[1]
            idx_dim0 = tf.tile(tf.range(batch_size)[:, None], [1, tgt_seq_len])
            idx_dim1 = tf.tile(tf.range(tgt_seq_len)[None, :], [batch_size, 1])
            idx_dim2 = tgt_out
            idx_3d = tf.stack([idx_dim0, idx_dim1, idx_dim2], axis=-1)
            selected_logits = tf.gather_nd(final_output, idx_3d)

        tf.train.init_from_checkpoint(
            self.bert_ckpt_file,
            assignment_map={"bert/": "drrn-action-encoder/bert-decoder/bert/"})

        tgt_len_mask = tf.sequence_mask(
            tgt_len, maxlen=tgt_seq_len, dtype=tf.float32)
        q_actions = tf.reduce_mean(
            tf.multiply(selected_logits, tgt_len_mask), axis=-1)
        return q_actions, h_state

    def get_train_op(self, q_actions):
        loss, abs_loss = dqn.l2_loss_1d_action(
            q_actions, self.inputs["action_idx"], self.inputs["expected_q"],
            self.inputs["b_weight"])

        var_encoder = tf.trainable_variables(scope="tj-bert-encoder")
        var_action_encoder = tf.trainable_variables(scope="drrn-action-encoder")
        allowed_var_action_encoder = list(filter(
            lambda v: "bert-decoder" not in v.name, var_action_encoder))
        trainable_vars = var_encoder + allowed_var_action_encoder
        self.debug("trainable vars: {}".format(trainable_vars))

        train_op = self.optimizer.minimize(
            loss, global_step=self.global_step,
            var_list=trainable_vars)

        return loss, train_op, abs_loss
