import tensorflow as tf

from deeptextworld.models import transformer as txf
from deeptextworld.models.dqn_model import BaseDQN
from deeptextworld.models.export_models import GenDQNModel
from deeptextworld.models.utils import l2_loss_2d_action


class TransformerGenDQN(BaseDQN):
    def __init__(self, hp, is_infer=False):
        super(TransformerGenDQN, self).__init__(hp, is_infer)

        # redefine inputs, notice the shape of action_idx
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "src_seg": tf.placeholder(tf.int32, [None, None]),
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

    @classmethod
    def get_train_student_model(cls, hp, device_placement):
        return cls.get_train_model(hp, device_placement)

    @classmethod
    def get_train_model(cls, hp, device_placement):
        return create_train_gen_model(cls, hp, device_placement)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        return create_eval_gen_model(cls, hp, device_placement)

    def get_decoded_idx_infer(self):
        decoded_idx, decoded_logits, p_gen, valid_len = self.transformer.decode(
            self.inputs["src"], training=False,
            max_tar_len=self.hp.max_decoding_size,
            sos_id=self.hp.sos_id,
            eos_id=self.hp.eos_id,
            tj_master_mask=self.inputs["src_seg"],
            use_greedy=self.inputs["use_greedy"],
            beam_size=self.inputs["beam_size"],
            temperature=self.inputs["temperature"])
        return (
            decoded_idx, tf.squeeze(valid_len, axis=-1), decoded_logits, p_gen)

    def get_q_actions(self):
        q_actions, p_gen, _, _ = self.transformer(
            self.inputs["src"], tar=self.inputs["action_idx"],
            training=True, tj_master_mask=self.inputs["src_seg"])
        return q_actions, p_gen

    def get_train_op(self, q_actions):
        """
        b_weight under this function comes from experience replay pool, which
        is the abs difference between true q and estimated q values.
        """
        loss, abs_loss = l2_loss_2d_action(
            q_actions, self.inputs["action_idx_out"], self.inputs["expected_q"],
            self.hp.vocab_size, self.inputs["action_len"],
            self.hp.n_tokens_per_action, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss

    def get_seq2seq_train_op(self, q_actions):
        """
        b_weight under this function comes from softmax(q-values) from other
        DRRN agent.
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.inputs["action_idx_out"], logits=q_actions)
        losses = self.inputs["b_weight"][:, None] * losses
        action_len_mask = tf.sequence_mask(
            self.inputs["action_len"], self.hp.n_tokens_per_action)
        loss = tf.reduce_mean(tf.boolean_mask(losses, action_len_mask))
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op

    def get_best_2d_q(self, q_actions):
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
        action_idx = self.get_best_2d_q(q_actions)
        return self.get_acc_impl(action_idx)

    def get_acc_from_decoded_idx(self, decoded_idx):
        action_idx, _ = self.format_decoded_idx(decoded_idx)
        return self.get_acc_impl(action_idx)


def create_train_gen_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp)
            inputs = model.inputs
            q_actions, p_gen = model.get_q_actions()
            (decoded_idx, col_eos_idx, decoded_logits, p_gen_infer
             ) = model.get_decoded_idx_infer()
            acc_train = model.get_acc(q_actions)
            loss, train_op, abs_loss = model.get_train_op(q_actions)
            loss_summary = tf.summary.scalar("loss", loss)
            acc_train_summary = tf.summary.scalar("acc_train", acc_train)
            train_summary_op = tf.summary.merge([loss_summary])
            (loss_seq2seq, train_seq2seq_op
             ) = model.get_seq2seq_train_op(q_actions)
            loss_summary_2 = tf.summary.scalar("loss_seq2seq", loss_seq2seq)
            train_seq2seq_summary_op = tf.summary.merge(
                [loss_summary_2, acc_train_summary])
    return GenDQNModel(
        graph=graph, q_actions=q_actions,
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
        src_seg_=inputs["src_seg"], h_state=None)


def create_eval_gen_model(model_creator, hp, device_placement):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_placement):
            model = model_creator(hp, is_infer=True)
            inputs = model.inputs
            q_actions, p_gen = model.get_q_actions()
            (decoded_idx, col_eos_idx, decoded_logits, p_gen_infer
             ) = model.get_decoded_idx_infer()
    return GenDQNModel(
        graph=graph, q_actions=q_actions,
        decoded_idx_infer=decoded_idx,
        src_=inputs["src"],
        src_len_=inputs["src_len"],
        train_op=None, action_idx_=inputs["action_idx"],
        action_idx_out_=inputs["action_idx_out"],
        action_len_=inputs["action_len"],
        b_weight_=inputs["b_weight"],
        temperature_=inputs["temperature"],
        expected_q_=inputs["expected_q"], loss=None,
        abs_loss=None,
        p_gen=p_gen, p_gen_infer=p_gen_infer,
        train_summary_op=None,
        loss_seq2seq=None,
        train_seq2seq_op=None,
        train_seq2seq_summary_op=None,
        beam_size_=inputs["beam_size"],
        use_greedy_=inputs["use_greedy"],
        col_eos_idx=col_eos_idx,
        decoded_logits_infer=decoded_logits,
        src_seg_=inputs["src_seg"], h_state=None)
