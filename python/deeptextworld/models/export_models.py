from typing import Optional

import tensorflow as tf


class DQNModel(object):
    def __init__(
            self,
            graph: tf.Graph,
            q_actions: tf.Tensor,
            src_: tf.placeholder,
            src_len_: tf.placeholder,
            action_idx_: Optional[tf.placeholder],
            train_op: Optional[tf.Operation],
            loss: Optional[tf.Tensor],
            expected_q_: Optional[tf.placeholder],
            b_weight_: Optional[tf.placeholder],
            train_summary_op: Optional[tf.Operation],
            abs_loss: Optional[tf.Tensor],
            src_seg_: Optional[tf.placeholder],
            h_state: Optional[tf.Tensor]):
        self.graph = graph
        self.q_actions = q_actions
        self.src_ = src_
        self.src_len_ = src_len_
        self.action_idx_ = action_idx_
        self.train_op = train_op
        self.loss = loss
        self.expected_q_ = expected_q_
        self.b_weight_ = b_weight_
        self.train_summary_op = train_summary_op
        self.abs_loss = abs_loss
        self.src_seg_ = src_seg_
        self.h_state = h_state


class GenDQNModel(DQNModel):
    def __init__(
            self,
            graph: tf.Graph,
            q_actions: tf.Tensor,
            src_: tf.placeholder,
            src_len_: tf.placeholder,
            action_idx_: Optional[tf.placeholder],
            train_op: Optional[tf.Operation],
            loss: Optional[tf.Tensor],
            expected_q_: Optional[tf.placeholder],
            b_weight_: Optional[tf.placeholder],
            train_summary_op: Optional[tf.Operation],
            abs_loss: Optional[tf.Tensor],
            src_seg_: Optional[tf.placeholder],
            h_state: Optional[tf.Tensor],
            decoded_idx_infer: tf.Tensor,
            action_idx_out_: tf.placeholder,
            action_len_: tf.placeholder,
            temperature_: tf.placeholder,
            p_gen: tf.Tensor,
            p_gen_infer: tf.Tensor,
            beam_size_: tf.placeholder,
            use_greedy_: tf.placeholder,
            col_eos_idx: tf.Tensor,
            decoded_logits_infer: tf.Tensor,
            loss_seq2seq: Optional[tf.Tensor],
            train_seq2seq_summary_op: Optional[tf.Operation],
            train_seq2seq_op: Optional[tf.Operation]):
        super(GenDQNModel, self).__init__(
            graph,
            q_actions,
            src_,
            src_len_,
            action_idx_,
            train_op,
            loss,
            expected_q_,
            b_weight_,
            train_summary_op,
            abs_loss,
            src_seg_,
            h_state)
        self.decoded_idx_infer = decoded_idx_infer
        self.action_idx_out_ = action_idx_out_
        self.action_len_ = action_len_
        self.temperature_ = temperature_
        self.p_gen = p_gen
        self.p_gen_infer = p_gen_infer
        self.beam_size_ = beam_size_
        self.use_greedy_ = use_greedy_
        self.col_eos_idx = col_eos_idx
        self.decoded_logits_infer = decoded_logits_infer
        self.loss_seq2seq = loss_seq2seq
        self.train_seq2seq_summary_op = train_seq2seq_summary_op
        self.train_seq2seq_op = train_seq2seq_op


class DRRNModel(DQNModel):
    def __init__(
            self,
            graph: tf.Graph,
            q_actions: tf.Tensor,
            src_: tf.placeholder,
            src_len_: tf.placeholder,
            action_idx_: Optional[tf.placeholder],
            train_op: Optional[tf.Operation],
            loss: Optional[tf.Tensor],
            expected_q_: Optional[tf.placeholder],
            b_weight_: Optional[tf.placeholder],
            train_summary_op: Optional[tf.Operation],
            abs_loss: Optional[tf.Tensor],
            src_seg_: Optional[tf.placeholder],
            h_state: Optional[tf.Tensor],
            actions_: tf.placeholder,
            actions_len_: tf.placeholder,
            actions_repeats_: tf.placeholder):
        super(DRRNModel, self).__init__(
            graph,
            q_actions,
            src_,
            src_len_,
            action_idx_,
            train_op,
            loss,
            expected_q_,
            b_weight_,
            train_summary_op,
            abs_loss,
            src_seg_,
            h_state)
        self.actions_ = actions_
        self.actions_len_ = actions_len_
        self.actions_repeats_ = actions_repeats_


class CommonsenseModel(DQNModel):
    def __init__(
            self,
            graph: tf.Graph,
            q_actions: tf.Tensor,
            src_: tf.placeholder,
            src_len_: tf.placeholder,
            action_idx_: Optional[tf.placeholder],
            train_op: Optional[tf.Operation],
            loss: Optional[tf.Tensor],
            expected_q_: Optional[tf.placeholder],
            b_weight_: Optional[tf.placeholder],
            train_summary_op: Optional[tf.Operation],
            swag_train_summary_op: Optional[tf.Operation],
            abs_loss: Optional[tf.Tensor],
            src_seg_: Optional[tf.placeholder],
            h_state: Optional[tf.Tensor],
            seg_tj_action_: tf.placeholder,
            swag_labels_: Optional[tf.placeholder],
            swag_loss: Optional[tf.Tensor],
            swag_train_op: Optional[tf.Operation]):
        super(CommonsenseModel, self).__init__(
            graph,
            q_actions,
            src_,
            src_len_,
            action_idx_,
            train_op,
            loss,
            expected_q_,
            b_weight_,
            train_summary_op,
            abs_loss,
            src_seg_,
            h_state)
        self.seg_tj_action_ = seg_tj_action_
        self.swag_labels_ = swag_labels_
        self.swag_loss = swag_loss
        self.swag_train_op = swag_train_op
        self.swag_train_summary_op = swag_train_summary_op


class DSQNModel(DRRNModel):
    def __init__(
            self,
            graph: tf.Graph,
            q_actions: tf.Tensor,
            src_: tf.placeholder,
            src_len_: tf.placeholder,
            action_idx_: Optional[tf.placeholder],
            train_op: Optional[tf.Operation],
            loss: Optional[tf.Tensor],
            expected_q_: Optional[tf.placeholder],
            b_weight_: Optional[tf.placeholder],
            train_summary_op: Optional[tf.Operation],
            abs_loss: Optional[tf.Tensor],
            src_seg_: Optional[tf.placeholder],
            h_state: Optional[tf.Tensor],
            actions_: tf.placeholder,
            actions_len_: tf.placeholder,
            actions_repeats_: tf.placeholder,
            snn_train_summary_op: Optional[tf.Operation],
            weighted_train_summary_op: Optional[tf.Operation],
            semantic_same: tf.Tensor,
            snn_src_: Optional[tf.placeholder],
            snn_src_len_: Optional[tf.placeholder],
            snn_src2_: Optional[tf.placeholder],
            snn_src2_len_: Optional[tf.placeholder],
            labels_: Optional[tf.placeholder],
            snn_loss: Optional[tf.Tensor],
            weighted_loss: Optional[tf.Tensor],
            merged_train_op: Optional[tf.Operation],
            snn_train_op: Optional[tf.Operation],
            h_states_diff: Optional[tf.Tensor]):
        super(DSQNModel, self).__init__(
            graph,
            q_actions,
            src_,
            src_len_,
            action_idx_,
            train_op,
            loss,
            expected_q_,
            b_weight_,
            train_summary_op,
            abs_loss,
            src_seg_,
            h_state,
            actions_,
            actions_len_,
            actions_repeats_)

        self.snn_train_summary_op = snn_train_summary_op
        self.weighted_train_summary_op = weighted_train_summary_op
        self.semantic_same = semantic_same
        self.snn_src_ = snn_src_
        self.snn_src_len_ = snn_src_len_
        self.snn_src2_ = snn_src2_
        self.snn_src2_len_ = snn_src2_len_
        self.labels_ = labels_
        self.snn_loss = snn_loss
        self.weighted_loss = weighted_loss
        self.merged_train_op = merged_train_op
        self.snn_train_op = snn_train_op
        self.h_states_diff = h_states_diff
