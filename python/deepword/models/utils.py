import tensorflow as tf
import numpy as np


def encoder_lstm(src, src_len, src_embeddings, num_units, num_layers):
    """
    encode state with LSTM

    Args:
        src: placeholder, (tf.int32, [None, None])
        src_len: placeholder, (tf.float32, [None])
        src_embeddings: (tf.float32, [vocab_size, embedding_size])
        num_units: number of LSTM units
        num_layers: number of LSTM layers

    Returns:
        inner states (c, h)
    """
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(num_layers)])

    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    _, inner_states = tf.nn.dynamic_rnn(
        encoder_cell, src_emb, sequence_length=src_len,
        initial_state=None, dtype=tf.float32)
    return inner_states


def encoder_cnn_base(
        input_tensor, filter_sizes, num_filters, num_channels, embedding_size,
        is_infer=False, activation="tanh"):
    """
    We pad input_tensor in the head for each string to generate equal-size
    output. E.g.

    go north forest path this is a path ...
    given conv-filter size 3, it will be padded in the head with two tokens
    <S> <S> go north forest path this is a path ... OR
    [PAD] [PAD] go north forest path this is a path ...

    the type of padding values doesn't matter only if it is a special token, and
    be identical for each model.

    We use constant value 0 here, so make sure index-0 is a special token
    that can be used to pad in your vocabulary.

    Args:
        input_tensor: (tf.float32,
          [batch_size, seq_len, embedding_size, num_channels])
        filter_sizes: list of ints, e.g. [3, 4, 5]
        num_filters: number of filters for each filter size
        num_channels: 1 or 2, depending on the input tensor
        embedding_size: word embedding size
        is_infer: training or infer
        activation: choose from "tanh" or "relu". Notice that if choose relu,
          make sure adding an extra dense layer, otherwise the output is all
          non-negative values.

    Returns:
        a vector as the inner state
    """
    layer_outputs = []
    for i, fs in enumerate(filter_sizes):
        with tf.variable_scope("conv-block-%s" % fs):
            src_paddings = tf.constant([[0, 0], [fs - 1, 0], [0, 0], [0, 0]])
            src_w_pad = tf.pad(
                input_tensor, paddings=src_paddings,
                mode="CONSTANT", constant_values=0)
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
            if activation == "tanh":
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
            elif activation == "relu":
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            else:
                raise ValueError("Unknown activation: {}".format(activation))
            dropout_h = tf.layers.dropout(
                inputs=h, rate=0.4,
                training=(not is_infer), name="dropout")
            layer_outputs.append(dropout_h)

    # Combine all the pooled features
    # Squeeze the 3rd dim that is the col of conv result
    inner_state = tf.squeeze(tf.concat(layer_outputs, axis=-1), axis=[2])
    return inner_state


def encoder_cnn(
        src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
        embedding_size, is_infer=False, num_channels=2, activation="tanh"):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification

    Args:
        src: placeholder, (tf.int32, [batch_size, seq_len])
        src_embeddings: (tf.float32, [vocab_size, embedding_size])
        pos_embeddings: (tf.float32, [max_position_size, embedding_size])
        filter_sizes: list of ints, e.g. [3, 4, 5]
        num_filters: number of filters of each filter_size
        embedding_size: embedding size
        is_infer: training or inference
        num_channels: 1 or 2.
        activation: tanh (default) or relu

    Returns:
        a vector as the inner state
    """
    with tf.variable_scope("cnn_encoder"):
        src_emb = tf.nn.embedding_lookup(src_embeddings, src)
        pos_emb = tf.slice(pos_embeddings, [0, 0], [tf.shape(src_emb)[1], -1])
        src_pos_emb = src_emb + pos_emb
        if num_channels == 1:
            in_tn = tf.expand_dims(src_pos_emb, axis=-1)  # channel dimension
        elif num_channels == 2:
            in_tn = tf.stack([src_emb, src_pos_emb], axis=-1)
        else:
            raise ValueError(
                "num_channels: 1 or 2. {} received".format(num_channels))
        h_cnn = encoder_cnn_base(
            in_tn, filter_sizes, num_filters, num_channels=num_channels,
            embedding_size=embedding_size, is_infer=is_infer,
            activation=activation)
        pooled = tf.reduce_max(h_cnn, axis=1)
        num_filters_total = num_filters * len(filter_sizes)
        inner_states = tf.reshape(pooled, [-1, num_filters_total])
    return inner_states


def l2_loss_1d_action(q_actions, action_idx, expected_q, b_weight):
    """
    l2 loss for 1D action space. only q values in q_actions
     selected by action_idx will be computed against expected_q
    e.g. "go east" would be one whole action.
    action_idx should have the same dimension as expected_q

    Args:
        q_actions: q-values
        action_idx: placeholder, the action chose for the state,
          in a format of (tf.int32, [None])
        expected_q: placeholder, the expected reward gained from the step,
          in a format of (tf.float32, [None])
        b_weight: weights for each data point

    Returns:
        l2 loss and l1 loss
    """
    predicted_q = tf.gather(q_actions, indices=action_idx)
    loss = tf.reduce_mean(b_weight * tf.square(expected_q - predicted_q))
    abs_loss = tf.abs(expected_q - predicted_q)
    return loss, abs_loss


def l2_loss_1d_action_v2(
        q_actions, action_idx, expected_q, n_actions, b_weight):
    """
    l2 loss for 1D action space.
    e.g. "go east" would be one whole action.

        q_actions: Q-vector of a state for all actions
        action_idx: placeholder, the action chose for the state,
          in a format of (tf.int32, [None])
        expected_q: placeholder, the expected reward gained from the step,
          in a format of (tf.float32, [None])
        n_actions: number of total actions
        b_weight: weights for each data point

    Returns:
        l2 loss and l1 loss
    """
    actions_mask = tf.one_hot(indices=action_idx, depth=n_actions)
    predicted_q = tf.reduce_sum(
        tf.multiply(q_actions, actions_mask), axis=1)
    loss = tf.reduce_mean(b_weight * tf.square(expected_q - predicted_q))
    abs_loss = tf.abs(expected_q - predicted_q)
    return loss, abs_loss


def l2_loss_2d_action(
        q_actions, action_idx, expected_q,
        vocab_size, action_len, max_action_len, b_weight):
    """
    l2 loss for 2D action space.
    e.g. "go east" is an action composed by "go" and "east".

    Args:
        q_actions: Q-matrix of a state for all action-components, e.g. tokens
        action_idx: placeholder, the action-components chose for the state,
          in a format of (tf.int32, [None, None])
        expected_q: placeholder, the expected reward gained from the step,
          in a format of (tf.float32, [None])
        vocab_size: number of action-components
        action_len: length of each action in a format of (tf.int32, [None])
        max_action_len: maximum length of action
        b_weight: weights for each data point

    Returns:
        l2 loss and l1 loss
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


def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    Create position embeddings with sin/cos, not need to train

    Args:
        position: maximum position size
        d_model: embedding size

    Returns:
        position embeddings in shape (1, position, d_model)
    """
    angle_rads = _get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],
        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
