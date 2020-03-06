from dataclasses import dataclass
from typing import Optional

import tensorflow as tf


@dataclass
class DQNModel:
    graph: tf.Graph
    q_actions: tf.Tensor
    src_: tf.placeholder
    src_len_: tf.placeholder
    action_idx_: Optional[tf.placeholder]
    train_op: Optional[tf.Operation]
    loss: Optional[tf.Tensor]
    expected_q_: Optional[tf.placeholder]
    b_weight_: Optional[tf.placeholder]
    train_summary_op: Optional[tf.Operation]
    abs_loss: Optional[tf.Tensor]
    src_seg_: Optional[tf.placeholder]
    h_state: Optional[tf.Tensor]


@dataclass
class GenDQNModel(DQNModel):
    decoded_idx_infer: tf.Tensor
    action_idx_out_: tf.placeholder
    action_len_: tf.placeholder
    temperature_: tf.placeholder
    p_gen: tf.Tensor
    p_gen_infer: tf.Tensor
    beam_size_: tf.placeholder
    use_greedy_: tf.placeholder
    col_eos_idx: tf.Tensor
    decoded_logits_infer: tf.Tensor
    loss_seq2seq: Optional[tf.Tensor]
    train_seq2seq_summary_op: Optional[tf.Operation]
    train_seq2seq_op: Optional[tf.Operation]


@dataclass
class DRRNModel(DQNModel):
    actions_: tf.placeholder
    actions_len_: tf.placeholder
    actions_repeats_: tf.placeholder


@dataclass
class CommonsenseModel(DQNModel):
    seg_tj_action_: tf.placeholder


@dataclass
class DSQNModel(DRRNModel):
    snn_train_summary_op: Optional[tf.Operation]
    weighted_train_summary_op: Optional[tf.Operation]
    semantic_same: tf.Tensor
    snn_src_: Optional[tf.placeholder]
    snn_src_len_: Optional[tf.placeholder]
    snn_src2_: Optional[tf.placeholder]
    snn_src2_len_: Optional[tf.placeholder]
    labels_: Optional[tf.placeholder]
    snn_loss: Optional[tf.Tensor]
    weighted_loss: Optional[tf.Tensor]
    merged_train_op: Optional[tf.Operation]
    snn_train_op: Optional[tf.Operation]
    h_states_diff: Optional[tf.Tensor]
