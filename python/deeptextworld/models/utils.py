import tensorflow as tf
import numpy as np


def repeat(data, repeats):
    """
    Repeat data with repeats. The function uses the same method as
    tf.repeat in tensorflow version >= 1.15.2
    For tensorflow version >= 1.15.2, you can use tf.repeat directly

    This function only works for 2D matrix, e.g.
    [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    repeats [1, 1, 2] times, then we get
    [[1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 4, 5]].

    :param data: 2D matrix, [batch_size, hidden_size]
    :param repeats: 1D int vector, [batch_size]
    :return: 2D matrix, [sum(repeats), hidden_size]
    """
    max_repeat = tf.reduce_max(repeats)
    data = data[:, None, :]
    data = tf.tile(data, [1, max_repeat, 1])
    mask = tf.sequence_mask(repeats)
    data = tf.boolean_mask(data, mask)
    return data


def l2_loss_1d_action(q_actions, action_idx, expected_q, b_weight):
    """
    l2 loss for 1D action space. only q values in q_actions
     selected by action_idx will be computed against expected_q
    e.g. "go east" would be one whole action.
    action_idx should have the same dimension as expected_q
    :param q_actions: q-values
    :param action_idx: placeholder, the action chose for the state,
           in a format of (tf.int32, [None])
    :param expected_q: placeholder, the expected reward gained from the step,
           in a format of (tf.float32, [None])
    :param b_weight: weights for each data point
    """
    predicted_q = tf.gather(q_actions, indices=action_idx)
    loss = tf.reduce_mean(b_weight * tf.square(expected_q - predicted_q))
    abs_loss = tf.abs(expected_q - predicted_q)
    return loss, abs_loss


def l2_loss_2d_action(
        q_actions, action_idx, expected_q,
        vocab_size, action_len, max_action_len, b_weight):
    """
    l2 loss for 2D action space.
    e.g. "go east" is an action composed by "go" and "east".
    :param q_actions: Q-matrix of a state for all action-components, e.g. tokens
    :param action_idx: placeholder, the action-components chose for the state,
           in a format of (tf.int32, [None, None])
    :param expected_q: placeholder, the expected reward gained from the step,
           in a format of (tf.float32, [None])
    :param vocab_size: number of action-components
    :param action_len: length of each action in a format of (tf.int32, [None])
    :param max_action_len: maximum length of action
    :param b_weight: weights for each data point
    """
    actions_idx_mask = tf.one_hot(indices=action_idx, depth=vocab_size)
    q_actions_mask = tf.sequence_mask(
        action_len, maxlen=max_action_len, dtype=tf.float32)
    q_val_by_idx = tf.multiply(q_actions, actions_idx_mask)
    q_val_by_valid_idx = tf.multiply(
        q_val_by_idx, tf.expand_dims(q_actions_mask, axis=-1))
    sum_q_by_idx = tf.reduce_sum(q_val_by_valid_idx, axis=[1, 2])
    predicted_q = tf.div(sum_q_by_idx, tf.cast(action_len, tf.float32))
    loss = tf.reduce_mean(b_weight * tf.square(expected_q - predicted_q))
    abs_loss = tf.abs(expected_q - predicted_q)
    return loss, abs_loss


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],
        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
