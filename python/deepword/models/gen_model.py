import tensorflow as tf

from deepword.models import transformer as txf
from deepword.models.dqn_model import BaseDQN
from deepword.models.export_models import GenDQNModel
from deepword.models.utils import l2_loss_2d_action


class TransformerGenDQN(BaseDQN):
    def __init__(self, hp, is_infer=False):
        super(TransformerGenDQN, self).__init__(hp, is_infer)

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
                q_actions, p_gen = model.get_q_actions()
                (decoded_idx, p_gen_infer, col_eos_idx, decoded_logits
                 ) = model.decode()
                loss, train_op, abs_loss = model.get_train_op(q_actions)
                loss_summary = tf.summary.scalar("loss", loss)
                train_summary_op = tf.summary.merge([loss_summary])
        return GenDQNModel(
            graph=graph,
            q_actions=q_actions,
            decoded_idx_infer=decoded_idx,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            train_op=train_op,
            action_idx_=inputs["action_idx"],
            action_idx_out_=inputs["action_idx_out"],
            action_len_=inputs["action_len"],
            b_weight_=inputs["b_weight"],
            temperature_=inputs["temperature"],
            expected_q_=inputs["expected_q"],
            loss=loss,
            abs_loss=abs_loss,
            p_gen=p_gen,
            p_gen_infer=p_gen_infer,
            train_summary_op=train_summary_op,
            beam_size_=inputs["beam_size"],
            use_greedy_=inputs["use_greedy"],
            col_eos_idx=col_eos_idx,
            decoded_logits_infer=decoded_logits,
            src_seg_=None,
            h_state=None)

    @classmethod
    def get_eval_model(cls, hp, device_placement):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device_placement):
                model = cls(hp, is_infer=True)
                inputs = model.inputs
                q_actions, p_gen = model.get_q_actions()
                (decoded_idx, decoded_logits, p_gen_infer, col_eos_idx
                 ) = model.decode()
        return GenDQNModel(
            graph=graph,
            q_actions=q_actions,
            decoded_idx_infer=decoded_idx,
            src_=inputs["src"],
            src_len_=inputs["src_len"],
            train_op=None,
            action_idx_=inputs["action_idx"],
            action_idx_out_=inputs["action_idx_out"],
            action_len_=inputs["action_len"],
            b_weight_=inputs["b_weight"],
            temperature_=inputs["temperature"],
            expected_q_=inputs["expected_q"],
            loss=None,
            abs_loss=None,
            p_gen=p_gen,
            p_gen_infer=p_gen_infer,
            train_summary_op=None,
            beam_size_=inputs["beam_size"],
            use_greedy_=inputs["use_greedy"],
            col_eos_idx=col_eos_idx,
            decoded_logits_infer=decoded_logits,
            src_seg_=None,
            h_state=None)

    def decode(self):
        return self.transformer.decode(
            self.inputs["src"],
            training=False,
            max_tar_len=self.hp.max_decoding_size,
            sos_id=self.hp.sos_id,
            eos_id=self.hp.eos_id,
            padding_id=self.hp.padding_val_id,
            use_greedy=self.inputs["use_greedy"],
            beam_size=self.inputs["beam_size"],
            temperature=self.inputs["temperature"])

    def get_q_actions(self):
        q_actions, p_gen, _, _ = self.transformer(
            self.inputs["src"], tar=self.inputs["action_idx"],
            training=True)
        return q_actions, p_gen

    def get_train_op(self, q_actions):
        loss, abs_loss = l2_loss_2d_action(
            q_actions, self.inputs["action_idx_out"], self.inputs["expected_q"],
            self.hp.vocab_size, self.inputs["action_len"],
            self.hp.n_tokens_per_action, self.inputs["b_weight"])
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, abs_loss


class TransformerPGN(TransformerGenDQN):
    """
    TransformerPGN is similar with TransformerGenDQN, the only difference is
    the former uses cross entropy loss, while the latter uses MSE.
    Thus, TransformerPGN is not allowed training with the DQN framework.
    It can only be trained with supervised learning, e.g. imitation learning.
    """
    def __init__(self, hp, is_infer=False):
        super(TransformerPGN, self).__init__(hp, is_infer)

        # redefine inputs, notice the shape of action_idx
        # b_weight has two dimensions.
        # b_weight can be either [None, 1] to weigh each action for loss
        # or [None, None] to weigh each token of actions for loss
        self.inputs = {
            "src": tf.placeholder(tf.int32, [None, None]),
            "src_len": tf.placeholder(tf.float32, [None]),
            "src_seg": tf.placeholder(tf.int32, [None, None]),
            "action_idx": tf.placeholder(tf.int32, [None, None]),
            "action_idx_out": tf.placeholder(tf.int32, [None, None]),
            "expected_q": tf.placeholder(tf.float32, [None]),
            "action_len": tf.placeholder(tf.int32, [None]),
            "b_weight": tf.placeholder(tf.float32, [None, None]),
            "temperature": tf.placeholder(tf.float32, []),
            "beam_size": tf.placeholder(tf.int32, []),
            "use_greedy": tf.placeholder(tf.bool, [])
        }

    def get_train_op(self, q_actions):
        """
        b_weight could be
          1. per instance, i.e. [batch_size, 1]
          2. per token, i.e. [batch_size, n_tokens]
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.inputs["action_idx_out"], logits=q_actions)
        losses = self.inputs["b_weight"] * losses
        action_len_mask = tf.sequence_mask(
            self.inputs["action_len"], self.hp.n_tokens_per_action)
        loss = tf.reduce_mean(tf.boolean_mask(losses, action_len_mask))
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return loss, train_op, None
