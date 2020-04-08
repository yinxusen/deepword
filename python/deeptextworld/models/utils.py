import tensorflow as tf
import numpy as np


def encoder_lstm(src, src_len, src_embeddings, num_units, num_layers):
    """
    encode state with LSTM
    :param src: placeholder, (tf.int32, [None, None])
    :param src_len: placeholder, (tf.float32, [None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param num_units: number of LSTM units
    :param num_layers: number of LSTM layers
    """
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(num_layers)])

    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    _, inner_states = tf.nn.dynamic_rnn(
        encoder_cell, src_emb, sequence_length=src_len,
        initial_state=None, dtype=tf.float32)
    return inner_states


def encoder_cnn_prepare_input_two_facets(src, src_embeddings, pos_embeddings):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [batch_size, src_len])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings: (tf.float32, [pos_emb_len, embedding_size])
    """
    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    pos_emb = tf.slice(pos_embeddings, [0, 0], [tf.shape(src_emb)[1], -1])
    src_pos_emb = src_emb + pos_emb
    src_emb_expanded = tf.stack([src_emb, src_pos_emb], axis=-1)
    return src_emb_expanded


def encoder_cnn_base(
        input_tensor, filter_sizes, num_filters, num_channels, embedding_size,
        is_infer=False):
    layer_outputs = []
    for i, fs in enumerate(filter_sizes):
        with tf.variable_scope("conv-block-%s" % fs):
            src_paddings = tf.constant([[0, 0], [fs - 1, 0], [0, 0], [0, 0]])
            src_w_pad = tf.pad(
                input_tensor, paddings=src_paddings, mode="CONSTANT")
            # Convolution Layer
            filter_shape = [fs, embedding_size, num_channels, num_filters]
            w = tf.get_variable(
                name="W",
                initializer=lambda: tf.truncated_normal(
                    filter_shape, stddev=0.1))
            b = tf.get_variable(
                name="b",
                initializer=lambda: tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(
                input=src_w_pad, filter=w, strides=[1, 1, 1, 1],
                padding="VALID", name="conv")
            # Apply nonlinearity
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
            dropout_h = tf.layers.dropout(
                inputs=h, rate=0.4,
                training=(not is_infer), name="dropout")
            layer_outputs.append(dropout_h)

    # Combine all the pooled features
    # Squeeze the 3rd dim that is the col of conv result
    inner_state = tf.squeeze(tf.concat(layer_outputs, axis=-1), axis=[2])
    return inner_state


def encoder_cnn_block(
        src, src_embeddings, pos_embeddings,
        filter_sizes, num_filters,
        embedding_size, is_infer=False):
    in_tn = encoder_cnn_prepare_input_two_facets(
        src, src_embeddings, pos_embeddings)
    return encoder_cnn_base(
        in_tn, filter_sizes, num_filters, num_channels=2,
        embedding_size=embedding_size, is_infer=is_infer)


def encoder_cnn(
        src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
        embedding_size, is_infer=False):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [None, None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings:
     position embedding, (tf.float32, [src_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    :param is_infer:
    """
    with tf.variable_scope("cnn_encoder"):
        h_cnn = encoder_cnn_block(
            src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
            embedding_size, is_infer)
        pooled = tf.reduce_max(h_cnn, axis=1)
        num_filters_total = num_filters * len(filter_sizes)
        inner_states = tf.reshape(pooled, [-1, num_filters_total])
    return inner_states


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
