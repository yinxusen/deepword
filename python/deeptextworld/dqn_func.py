import tensorflow as tf
import numpy as np


def l2_loss_1Daction(q_actions, action_idx, expected_q, n_actions, b_weight):
    """
    l2 loss for 1D action space.
    e.g. "go east" would be one whole action.
    :param q_actions: Q-vector of a state for all actions
    :param action_idx: placeholder, the action chose for the state,
           in a format of (tf.int32, [None])
    :param expected_q: placeholder, the expected reward gained from the step,
           in a format of (tf.float32, [None])
    :param n_actions: number of total actions
    """
    actions_mask = tf.one_hot(indices=action_idx, depth=n_actions)
    predicted_q = tf.reduce_sum(tf.multiply(q_actions, actions_mask),
                                axis=1)
    loss = tf.reduce_mean(b_weight * tf.square(expected_q - predicted_q))
    abs_loss = tf.abs(expected_q - predicted_q)
    return loss, abs_loss


def l1_loss_1Daction(q_actions, action_idx, expected_q, n_actions, b_weight):
    """
    l2 loss for 1D action space.
    e.g. "go east" would be one whole action.
    :param q_actions: Q-vector of a state for all actions
    :param action_idx: placeholder, the action chose for the state,
           in a format of (tf.int32, [None])
    :param expected_q: placeholder, the expected reward gained from the step,
           in a format of (tf.float32, [None])
    :param n_actions: number of total actions
    """
    actions_mask = tf.one_hot(indices=action_idx, depth=n_actions)
    predicted_q = tf.reduce_sum(tf.multiply(q_actions, actions_mask),
                                axis=1)
    abs_loss = tf.abs(expected_q - predicted_q)
    loss = tf.reduce_mean(b_weight * abs_loss)
    return loss, abs_loss


def l2_loss_2Daction(
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


def encoder_cnn_prepare_input(src, src_embeddings, pos_embeddings):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [batch_size, src_len])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings: (tf.float32, [pos_emb_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    :return inner_state: (tf.float32, [batch_size, max_src_len, len(filter_sizes) * num_filters])
    """
    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    pos_emb = tf.slice(pos_embeddings, [0, 0], [tf.shape(src_emb)[1], -1])
    src_pos_emb = src_emb + pos_emb
    src_emb_expanded = tf.expand_dims(src_pos_emb, axis=-1)  # channel dimension
    return src_emb_expanded


def encoder_cnn_prepare_input_two_facets(src, src_embeddings, pos_embeddings):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [batch_size, src_len])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings: (tf.float32, [pos_emb_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    :return inner_state: (tf.float32, [batch_size, max_src_len, len(filter_sizes) * num_filters])
    """
    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    pos_emb = tf.slice(pos_embeddings, [0, 0], [tf.shape(src_emb)[1], -1])
    src_pos_emb = src_emb + pos_emb
    src_emb_expanded = tf.stack([src_emb, src_pos_emb], axis=-1)
    return src_emb_expanded


def encoder_cnn_prepare_input_three_facets(
        src, src_seg, src_embeddings, pos_embeddings, seg_embeddings):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [batch_size, src_len])
    :param src_seg: placeholder, (tf.int32, [batch_size, src_len])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_embeddings: (tf.float32, [pos_emb_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    :return inner_state: (tf.float32, [batch_size, max_src_len, len(filter_sizes) * num_filters])
    """
    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    pos_emb = tf.slice(pos_embeddings, [0, 0], [tf.shape(src_emb)[1], -1])
    seg_emb = tf.nn.embedding_lookup(seg_embeddings, src_seg)
    src_emb_expanded = tf.stack(
        [src_emb, src_emb + pos_emb, src_emb + seg_emb], axis=-1)
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
    :param pos_emb: position embedding, (tf.float32, [src_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    """
    with tf.variable_scope("cnn_encoder"):
        h_cnn = encoder_cnn_block(
            src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
            embedding_size, is_infer)
        pooled = tf.reduce_max(h_cnn, axis=1)
        num_filters_total = num_filters * len(filter_sizes)
        inner_states = tf.reshape(pooled, [-1, num_filters_total])
    return inner_states


def encoder_cnn_block_3(
        src, src_seg, src_embeddings, pos_embeddings, seg_embeddings,
        filter_sizes, num_filters,
        embedding_size, is_infer=False):
    in_tn = encoder_cnn_prepare_input_three_facets(
        src, src_seg, src_embeddings, pos_embeddings, seg_embeddings)
    return encoder_cnn_base(
        in_tn, filter_sizes, num_filters, num_channels=3,
        embedding_size=embedding_size, is_infer=is_infer)


def encoder_cnn_3(
        src, src_seg, src_embeddings, pos_embeddings, seg_embeddings,
        filter_sizes, num_filters,
        embedding_size, is_infer=False):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [None, None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_emb: position embedding, (tf.float32, [src_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    """
    with tf.variable_scope("cnn_encoder"):
        h_cnn = encoder_cnn_block_3(
            src, src_seg, src_embeddings, pos_embeddings, seg_embeddings,
            filter_sizes, num_filters,
            embedding_size, is_infer)
        pooled = tf.reduce_max(h_cnn, axis=1)
        num_filters_total = num_filters * len(filter_sizes)
        inner_states = tf.reshape(pooled, [-1, num_filters_total])
    return inner_states


def encoder_attn_cnn(
        src, src_embeddings, pos_embeddings, max_src_len,
        filter_sizes, num_filters, embedding_size, is_infer=False):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [None, None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_emb: position embedding, (tf.float32, [src_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    """
    with tf.variable_scope("cnn_encoder"):
        h_cnn = encoder_cnn_block(
            src, src_embeddings, pos_embeddings, filter_sizes, num_filters,
            embedding_size, is_infer)
        pool_shape = [max_src_len, len(filter_sizes) * num_filters]
        w_pool = tf.get_variable(
            "w_pool",
            initializer=lambda: tf.truncated_normal(pool_shape, stddev=0.1),
            dtype=tf.float32)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w_pool)
        pooled = tf.reduce_sum(
            tf.multiply(h_cnn, tf.expand_dims(w_pool, axis=0)), axis=1)
        num_filters_total = num_filters * len(filter_sizes)
        inner_states = tf.reshape(pooled, [-1, num_filters_total])
    return inner_states


def encoder_cnn_multilayers(
        src, src_embeddings, pos_embeddings, num_layers, filter_size, embedding_size):
    """
    encode state with CNN, refer to
    Convolutional Neural Networks for Sentence Classification
    :param src: placeholder, (tf.int32, [None, None])
    :param src_embeddings: (tf.float32, [vocab_size, embedding_size])
    :param pos_emb: position embedding, (tf.float32, [src_len, embedding_size])
    :param filter_sizes: list of ints, e.g. [3, 4, 5]
    :param num_filters: number of filters of each filter_size
    :param embedding_size: embedding size
    """
    in_tn = encoder_cnn_prepare_input(src, src_embeddings, pos_embeddings)
    out_tns = []

    with tf.variable_scope("cnn_encoder_multilayers"):
        for layer in range(num_layers):
            with tf.variable_scope("cnn_encoder_layer_{}".format(layer)):
                h_cnn = encoder_cnn_base(
                    in_tn,
                    filter_sizes=[filter_size],
                    num_filters=embedding_size,
                    num_channels=2,
                    embedding_size=embedding_size)
                out_tns.append(h_cnn)
                in_tn = tf.expand_dims(h_cnn, axis=-1)

    return out_tns[-1]


def encoder_cnn_multichannels(
        src, src_len, src_embeddings, filter_sizes, num_filters,
        embedding_size, num_channels):
    """

    :param src: ('int32', [None, None, None])
    :param src_len: ('float', [None, None])
    :param src_embeddings:
    :param filter_sizes:
    :param num_filters:
    :param embedding_size:
    :return:
    """
    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    src_emb_trans = tf.transpose(src_emb, perm=[0, 2, 3, 1])  # for NHWC

    max_src_len = tf.shape(src)[1]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size,
                            num_channels, num_filters]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                            name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(  # depthwise_conv2d is too slow
                src_emb_trans,
                w,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Max-pooling over the outputs
            src_mask = tf.sequence_mask(src_len - filter_size + 1,
                                        maxlen=max_src_len)
            h = tf.multiply(h, src_mask)
            pooled = tf.reduce_max(h, axis=1)
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_features_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, axis=3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_features_total])
    inner_states = h_pool_flat
    return inner_states


def decoder_dense_classification(inner_states, n_actions):
    """

    :param inner_states:
    :param n_actions:
    :return:
    """
    q_actions = tf.layers.dense(inner_states, units=n_actions, use_bias=True)
    return q_actions


def decoder_fix_len_lstm(
        inner_state, n_actions, tgt_embeddings, num_units, num_layers,
        sos_id, eos_id, max_action_len=10):
    """

    :param inner_state:
    :param n_actions:
    :param tgt_embeddings:
    :param num_units:
    :param num_layers:
    :param sos_id:
    :return:
    """
    batch_size = tf.shape(inner_state[-1].c)[0]
    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units) for _ in range(num_layers)])

    projection_layer = tf.layers.Dense(units=n_actions, use_bias=True)
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=tgt_embeddings,
        start_tokens=tf.fill([batch_size], sos_id),
        end_token=eos_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, inner_state,
        output_layer=projection_layer)
    outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, output_time_major=False, impute_finished=True,
        swap_memory=True, maximum_iterations=max_action_len)
    q_actions = outputs.rnn_output
    return q_actions


def decoder_fix_len_cnn(
        inner_state, tgt_embeddings, pos_embeddings, n_tokens, embedding_size,
        filter_sizes, num_filters, sos_id, max_action_len=10):

    with tf.variable_scope("cnn_decoder", reuse=tf.AUTO_REUSE):
        projection_layer = tf.layers.Dense(
            units=n_tokens, use_bias=True, name="dense_tokens")

        def decode_one(x, z):
            # get state of last token
            h = encoder_cnn_block(
                src=x, src_embeddings=tgt_embeddings,
                pos_embeddings=pos_embeddings,
                filter_sizes=filter_sizes, num_filters=num_filters,
                embedding_size=embedding_size)[:, -1, :]
            # compute attention weight
            a = tf.nn.softmax(tf.reduce_sum(
                tf.multiply(tf.expand_dims(h, axis=1), z), axis=-1))
            # compute attention vector
            c = tf.reduce_sum(tf.multiply(a[:, :, tf.newaxis], z), axis=1)
            # compute token distribution
            y = projection_layer(h + c)
            return y

        def cond(i, out_ta, in_tn):
            return tf.less(i, max_action_len)

        def body(i, out_ta, in_tn):
            token_readout = decode_one(in_tn, inner_state)
            token_idx = tf.argmax(token_readout, axis=1, output_type=tf.int32)
            out_ta = out_ta.write(i, token_readout)
            in_tn = tf.concat([in_tn, token_idx[:, tf.newaxis]], axis=1)
            i = tf.add(i, 1)
            return [i, out_ta, in_tn]

        start_i = tf.constant(0, dtype=tf.int32)
        batch_size = tf.shape(inner_state)[0]
        batch_sos = tf.tile([sos_id], [batch_size])
        input_tn = batch_sos[:, tf.newaxis]
        output_ta = tf.TensorArray(dtype=tf.float32, size=max_action_len)
        _, output_ta, _ = tf.while_loop(
            cond, body,
            loop_vars=[start_i, output_ta, input_tn],
            shape_invariants=[start_i.get_shape(), tf.TensorShape(None),
                              tf.TensorShape([None, None])])
        q_actions = tf.transpose(output_ta.stack(), perm=[1, 0, 2])
    return q_actions


def decoder_fix_len_cnn_multilayers(
        inner_state, tgt_embeddings, pos_embeddings, n_tokens, embedding_size,
        num_layers, filter_size, sos_id, max_action_len=10):

    with tf.variable_scope("cnn_decoder", reuse=tf.AUTO_REUSE):
        projection_layer = tf.layers.Dense(
            units=n_tokens, use_bias=True, name="dense_tokens")

        def decode_one(x, z):
            # get state of last token
            h = encoder_cnn_multilayers(
                src=x, src_embeddings=tgt_embeddings,
                pos_embeddings=pos_embeddings,
                num_layers=num_layers,
                filter_size=filter_size,
                embedding_size=embedding_size)[:, -1, :]
            # compute attention weight
            a = tf.nn.softmax(tf.reduce_sum(
                tf.multiply(tf.expand_dims(h, axis=1), z), axis=-1))
            # compute attention vector
            c = tf.reduce_sum(tf.multiply(a[:, :, tf.newaxis], z), axis=1)
            # compute token distribution
            y = projection_layer(h + c)
            return y

        def cond(i, out_ta, in_tn):
            return tf.less(i, max_action_len)

        def body(i, out_ta, in_tn):
            token_readout = decode_one(in_tn, inner_state)
            token_idx = tf.argmax(token_readout, axis=1, output_type=tf.int32)
            out_ta = out_ta.write(i, token_readout)
            in_tn = tf.concat([in_tn, token_idx[:, tf.newaxis]], axis=1)
            i = tf.add(i, 1)
            return [i, out_ta, in_tn]

        start_i = tf.constant(0, dtype=tf.int32)
        batch_size = tf.shape(inner_state)[0]
        batch_sos = tf.tile([sos_id], [batch_size])
        input_tn = batch_sos[:, tf.newaxis]
        output_ta = tf.TensorArray(dtype=tf.float32, size=max_action_len)
        _, output_ta, _ = tf.while_loop(
            cond, body,
            loop_vars=[start_i, output_ta, input_tn],
            shape_invariants=[start_i.get_shape(), tf.TensorShape(None),
                              tf.TensorShape([None, None])])
        q_actions = tf.transpose(output_ta.stack(), perm=[1, 0, 2])
    return q_actions


def get_best_1Daction(q_actions_t, actions, mask=None):
    """
    :param q_actions_t: a q-vector of a state computed from TF at step t
    :param actions: action list
    """
    action_idx, q_val = get_best_1D_q(q_actions_t, mask)
    action = actions[action_idx]
    return action_idx, q_val, action


def get_best_1D_q(q_actions_t, mask=None):
    if mask is not None:
        inv_mask = np.logical_not(mask)
        min_q_val = np.min(q_actions_t)
        q_actions_t = q_actions_t * mask + inv_mask * min_q_val
    action_idx = np.argmax(q_actions_t)
    q_val = q_actions_t[action_idx]
    return action_idx, q_val


def choose_from_multinomial(dist):
    norm = np.sum(dist) * 1.0
    if norm <= 0:
        return np.random.randint(low=0, high=len(dist))
    else:
        rnd = np.random.random()
        dist /= norm
        agg_sum = 0
        for i in range(len(dist)):
            agg_sum += dist[i]
            if rnd <= agg_sum:
                return i
        return len(dist) - 1


def choose_from_n_multinominal(dists):
    """
    sample index according to multinominal distributions.
    :param dists: multinomial distributions in shape (n, m), which means there
                  are n distributions, each of them has m dimension.
    """
    n, m = dists.shape
    norm = np.expand_dims(np.sum(dists, axis=1), axis=1)
    rnd = np.random.random(size=n)
    normalized_dists = dists * 1. / norm
    agg_sum = np.zeros(n)
    chosen_idx = np.full(n, -1, dtype=np.int32)
    for i in range(m):
        agg_sum += normalized_dists[:, i]
        chosen_idx[(rnd < agg_sum) & (chosen_idx == -1)] = i
    chosen_idx[chosen_idx == -1] = m - 1
    return chosen_idx


def get_random_1Daction(actions, mask=1):
    """
    :param actions: action list
    :param mask: mask for the action list. 1 means OK to choose, 0 means NO.
           could be either an integer, or a numpy array the same size with
           actions.
    """
    rnd_dist = np.random.random(len(actions)) * mask
    action_idx = choose_from_multinomial(rnd_dist)
    action = actions[action_idx]
    return action_idx, action


def get_random_1Daction_fairly(actions, mask):
    """
    :param actions: action list
    :param mask: mask for the action list. 1 means OK to choose, 0 means NO.
           could be either an integer, or a numpy array the same size with
           actions.
    """
    actions2idx = dict(map(lambda x: (x[1], x[0]), enumerate(actions)))

    val_actions = list(
        map(lambda x: x[1], filter(lambda x: x[0], zip(mask, actions))))
    verbs = list(map(lambda a: a.split()[0], val_actions))

    uniq_verbs = list(set(verbs))
    print("unique verbs: {}".format(uniq_verbs))
    rnd_dist_verbs = np.random.random(len(uniq_verbs))
    verb_idx = choose_from_multinomial(rnd_dist_verbs)

    chosen_actions = list(
        filter(lambda a: a.split()[0] == uniq_verbs[verb_idx], val_actions))
    rnd_dist = np.random.random(len(chosen_actions))
    action_idx_in_chosen_actions = choose_from_multinomial(rnd_dist)

    action = chosen_actions[action_idx_in_chosen_actions]
    action_idx = actions2idx[action]
    return action_idx, action


def get_sampled_1Daction(q_actions_t, actions):
    """notice that the q_actions_t should be changed to probabilities first"""
    action_idx = choose_from_multinomial(q_actions_t)
    q_val = q_actions_t[action_idx]
    action = actions[action_idx]
    return action_idx, q_val, action


def get_best_2Daction(q_actions_t, tgt_tokens, eos_id):
    """
    create action string according to action token index.
    if q_actions_t[0] == eos_id, then return empty action string.
    :param q_actions_t: a q-matrix of a state computed from TF at step t
    :param tgt_tokens: target token list
    :param eos_id: end-of-sentence
    """
    action_idx, q_val, valid_len = get_best_2D_q(q_actions_t, eos_id)
    action = " ".join([tgt_tokens[a] for a in action_idx[:valid_len-1]])
    return action_idx, q_val, action


def get_best_2D_q(q_actions_t, eos_id) -> (list, float):
    """
    </S> also counts for an action, which is the empty action
    the last token should be </S>
    if it's not </S> according to the argmax, then force set it to be </S>.
    Q val for a whole action is the average of all Q val of valid tokens.
    :param q_actions_t: a q-matrix of a state computed from TF at step t
    :param eos_id: end-of-sentence
    """
    action_idx = np.argmax(q_actions_t, axis=1)
    valid_len = 0
    for a in action_idx:
        valid_len += 1
        if a == eos_id:
            break
    padded_action_idx = np.zeros_like(action_idx)
    padded_action_idx[:valid_len] = action_idx[:valid_len]
    # make sure the last token is eos no matter what
    padded_action_idx[valid_len-1] = eos_id
    q_val = np.mean(
        q_actions_t[range(valid_len), padded_action_idx[:valid_len]])
    return padded_action_idx, q_val, valid_len


def get_best_2D_q_v2(q_actions_t, eos_id) -> (list, float):
    """
    </S> also counts for an action, which is the empty action
    the last token should be </S>
    if it's not </S> according to the argmax, then force set it to be </S>.
    Q val for a whole action is the Q val of the last token.
    :param q_actions_t: a q-matrix of a state computed from TF at step t
    :param eos_id: end-of-sentence
    """
    action_idx = np.argmax(q_actions_t, axis=1)
    valid_len = 0
    for a in action_idx:
        valid_len += 1
        if a == eos_id:
            break
    padded_action_idx = np.zeros_like(action_idx)
    padded_action_idx[:valid_len] = action_idx[:valid_len]
    # make sure the last token is eos no matter what
    padded_action_idx[valid_len-1] = eos_id
    q_val = q_actions_t[range(valid_len), padded_action_idx[valid_len-1]]
    return padded_action_idx, q_val, valid_len


def get_best_2D_q_v3(q_actions_t, eos_id) -> (list, float):
    """
    </S> also counts for an action, which is the empty action
    the last token should be </S>
    if it's not </S> according to the argmax, then force set it to be </S>.
    Q val for a whole action is the maximum Q val of all valid tokens.
    :param q_actions_t: a q-matrix of a state computed from TF at step t
    :param eos_id: end-of-sentence
    """
    action_idx = np.argmax(q_actions_t, axis=1)
    valid_len = 0
    for a in action_idx:
        valid_len += 1
        if a == eos_id:
            break
    padded_action_idx = np.zeros_like(action_idx)
    padded_action_idx[:valid_len] = action_idx[:valid_len]
    # make sure the last token is eos no matter what
    padded_action_idx[valid_len-1] = eos_id
    q_val = np.max(
        q_actions_t[range(valid_len), padded_action_idx[:valid_len]])
    return padded_action_idx, q_val, valid_len


def get_random_2Daction_from_1Daction(actions, tgt_token2idx):
    """
    get a random 2D action by sampling a 1D action first, then decode the
    token indexes.
    :param actions: 1D action list
    :param tgt_token2idx: a token to index dictionary
    """
    action_idx = np.random.randint(0, len(actions))
    action = actions[action_idx]
    action_token_idx = [tgt_token2idx[t] for t in action.split()]
    return action_token_idx, action


def get_sampled_2Daction(q_actions_t, tgt_tokens):
    n, m = q_actions_t.shape
    action_idx = choose_from_n_multinominal(q_actions_t)
    q_val = np.mean(q_actions_t[range(n), action_idx])
    action = " ".join(tgt_tokens[a] for a in action_idx)
    return action_idx, q_val, action


def test():
    src = np.asarray([[1,2], [2,2], [2,1], [1,2], [1,2]], dtype=np.int32)
    src_embeddings = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.4, 0.6]], dtype=np.float32)
    pos_embeddings = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.4, 0.6], [0.1, 0.4, 0.6], [0.1, 0.4, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.4, 0.6], [0.1, 0.4, 0.6], [0.1, 0.4, 0.6]], dtype=np.float32)
    filter_size = 3
    filter_sizes = [1, 2]
    n_tokens = 3
    num_filters = 2
    embedding_size = 3
    sos_id = 1
    inner_state = encoder_cnn(
        src, src_embeddings, pos_embeddings, filter_sizes, num_filters, embedding_size, is_infer=False)

    # q_actions = decoder_fix_len_cnn_multilayers(
    #     inner_state, src_embeddings, pos_embeddings, n_tokens,
    #     embedding_size, 3, filter_size, sos_id)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run(inner_state)
    print(res)


if __name__ == '__main__':
    test()
