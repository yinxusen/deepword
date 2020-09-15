"""
Copied from https://www.tensorflow.org/beta/tutorials/text/transformer
decode function added by Xusen Yin
"""

import tensorflow as tf

from deepword.models.utils import positional_encoding


def create_padding_mask(seq):
    """
    Padding value should be 0.
    This mask contains one dimension for num_heads, i.e.
    (batch_size, <broadcast to num_heads>, <broadcast to seq_len_q>, seq_len_k)

    Args:
        seq: (batch_size, seq_len_k)

    Returns:
        padding mask, paddings is set to True, others are False
        shape: (batch_size, 1, 1, seq_len_k)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int):
    """
    create look ahead mask for decoding

    At every decoding step i, only t_0, ..., t_i can be accessed by the model,
    while t_{i+1}, ..., t_n should be masked out.

    Args:
        size: decoding output size

    Returns:
        look ahead mask, True means masked out.

    Examples:
        >>> create_look_ahead_mask(3)
        array([[0., 1., 1.],
               [0., 0., 1.],
               [0., 0., 0.]], dtype=float32)
    """
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_decode_masks(tar):
    """
    Create masking for decoding

    This masking combines the look ahead mask and target sentence padding mask.

    1. We create look ahead mask for each sentence;
    2. We combine the sentence padding mask with the look ahead mask, e.g. when
       the look ahead mask says "0" for a token, while the sentence padding mask
       says "1" for the same token because of the token is a padding, then the
       final mask for this token is "1".

    Args:
        tar: target sentence, shape: (batch_size, seq_len_k)

    Returns:
        a combined mask of look ahead mask and padding mask, shape: (batch_size,
        1, seq_len_k, seq_len_k)

    Examples:
        >>> tar_src = [[1,2,3,4,0,0], [1,3,0,0,0,0]]
        >>> create_decode_masks(tar_src)
        array([[[[0., 1., 1., 1., 1., 1.],
                 [0., 0., 1., 1., 1., 1.],
                 [0., 0., 0., 1., 1., 1.],
                 [0., 0., 0., 0., 1., 1.],
                 [0., 0., 0., 0., 1., 1.],
                 [0., 0., 0., 0., 1., 1.]]],

              [[[0., 1., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1., 1.]]]], dtype=float32)
    """
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Notice that mask must have the same dimensions as q, k, v.
      e.g. if q, k, v are (batch_size, num_heads, seq_len, depth), then the mask
      should be also (batch_size, num_heads, seq_len, depth).
      However, if q, k, v are (batch_size, seq_len, depth), then the mask should
      also not contain num_heads.

    Returns:
      output (a.k.a. context vectors), scaled_attention_logits
    """
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)
    return output, scaled_attention_logits


def point_wise_feed_forward_network(d_model, dff):
    """
    Two dense layers, one with activation, the second without activation.

    Args:
        d_model: model size
        dff: intermediate size

    Returns:
        FFN(x)
    """
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
         (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # (batch_size, num_heads, seq_len_q, depth)
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        (scaled_attention, scaled_attention_logits
         ) = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, seq_len_k)
        attn_logits = tf.reduce_sum(scaled_attention_logits, axis=1)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attn_logits


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layer_norm1(x + attn_output)

        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layer_norm2(out1 + ffn_output)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layer_norm1(attn1 + x)

        # attention weights from decoder output to encoder outputs
        # attn_logits: (batch_size, target_seq_len, enc_output_seq_len)
        # attn2: (batch_size, target_seq_len, d_model)
        attn2, attn_logits = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layer_norm2(attn2 + out1)

        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layer_norm3(ffn_output + out2)

        return out3, attn_logits


class Encoder(tf.keras.layers.Layer):
    def __init__(
            self, num_layers, d_model, num_heads, dff, input_vocab_size,
            dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        self.seg_embeddings = tf.stack(
            [tf.zeros(self.d_model), tf.ones(self.d_model)],
            name="seg_embeddings")

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None, mask=None, x_seg=None):
        if mask is None:
            mask = create_padding_mask(x)

        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        if x_seg is not None:
            x += tf.nn.embedding_lookup(self.seg_embeddings, x_seg)

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        # (batch_size, input_seq_len, d_model)
        return x


def _get_sparse_idx_tar_for_copy(
        batch_size: int, src_seq_len: int, target_seq_len: int):
    # (target_seq_len)
    # [0, 1, 2, ..., target_seq_len-1]
    idx_tar = tf.range(0, limit=target_seq_len)
    idx_tar = tf.expand_dims(idx_tar, axis=-1)
    # (target_seq_len, src_seq_len)
    idx_tar = tf.tile(idx_tar, [1, src_seq_len])
    idx_tar = tf.expand_dims(idx_tar, axis=0)
    # (batch_size, target_seq_len, src_seq_len)
    # e.g. (2, 3, 2), we have
    # [[[0, 0],
    #   [1, 1],
    #   [2, 2]],
    #  [[0, 0],
    #   [1, 1],
    #   [2, 2]]]
    idx_tar = tf.tile(idx_tar, [batch_size, 1, 1])
    return idx_tar


def _get_sparse_idx_src_for_copy(src, target_seq_len: int):
    # e.g. enc_x: [[a, a], [b, c]], we have
    # [[[a, a],
    #   [a, a],
    #   [a, a]],
    #  [[b, c],
    #   [b, c],
    #   [b, c]]]
    idx_src = tf.expand_dims(src, axis=1)
    idx_src = tf.tile(idx_src, [1, target_seq_len, 1])
    return idx_src


def get_sparse_idx_for_copy(src, target_seq_len: int):
    """
    Create sparse index from source sentence for copying into decoder using
    the `tf.scatter_nd` method.

    Considering the following source sentence: "a, b, a, c"; turn it into
    indices: [0, 1, 0, 2], and they have attention weights
    attn = [a0, a1, a2, a3].

    Now we want to decode a sentence with 3 tokens, for each generated token, we
    want to collect attention weights from the source sentence, and mix with
    the logits to generate the current token.

    I.e. for decoded sentence position i, we have
    logits(i) = [0.1, 0.2, 0.3, 0.5] for all possible tokens a, b, c, d.
    Then we want to sum the attention weights of two-0s, one-1, and one-2 into
    the logits(i) according to a generation weight p(i), i.e. total logits =
    logits(i) * p(i) + [a0 + a2, a1, a3, 0] * (1 - p(i)).

    The goal is to create a dense vector of vocabulary size, and copy attention
    weights from source sentence to the dense vector.

    We create a inverse index to do so. For target token i, we need to collect
    [(0, a0), (1, a1), (0, a2), (2, a3)] to construct the vector.

    Args:
        src: source sentence
        target_seq_len: target sequence len

    Returns:
        sparse index to construct attention weight matrix for a batch

    Examples:
        >>> get_sparse_idx_for_copy(src=[[0, 1, 0, 2]], target_seq_len=3)
        array([[[[0, 0],
                 [0, 1],
                 [0, 0],
                 [0, 2]],

                [[1, 0],
                 [1, 1],
                 [1, 0],
                 [1, 2]],

                [[2, 0],
                 [2, 1],
                 [2, 0],
                 [2, 2]]]], dtype=int32)
        shape: (1, 3, 4, 2)  # batch_size, target sentence len, source sentence
        len, 2D matrix indices
    """
    batch_size = tf.shape(src)[0]
    src_seq_len = tf.shape(src)[1]
    idx_src = _get_sparse_idx_src_for_copy(src, target_seq_len)
    idx_tar = _get_sparse_idx_tar_for_copy(
        batch_size, src_seq_len, target_seq_len)
    idx_tar_src = tf.stack([idx_tar, idx_src], axis=3)
    return idx_tar_src


class Decoder(tf.keras.layers.Layer):
    def __init__(
            self, num_layers, d_model, num_heads, dff, tgt_vocab_size,
            dropout_rate=0.1, with_pointer=False):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.tgt_vocab_size = tgt_vocab_size
        self.with_pointer = with_pointer

        self.embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = positional_encoding(tgt_vocab_size, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.logit_gen_layer = tf.keras.layers.Dense(units=1, use_bias=True)
        self.final_layer = tf.keras.layers.Dense(
            units=tgt_vocab_size, use_bias=True)

    def call(
            self, x, enc_x, enc_output, training,
            look_ahead_mask, padding_mask, copy_mask=None):
        """
        decode with pointer

        Args:
            x: decoder input
            enc_x: encoder input
            enc_output: encoder encoded result
            training: is training or inference
            look_ahead_mask: combined look ahead mask with padding mask
            padding_mask: padding mask for source sentence
            copy_mask: dense vector size |V| to mark all tokens that
             skip copying with 1; otherwise, 0.

        Returns:
            total logits, probability of generation, gen logits, copy logits
        """
        seq_len = tf.shape(x)[1]
        attention_logits = []

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        raw_x = x
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn_logits = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask)
            attention_logits.append(attn_logits)

        x = self.dropout(x, training=training)
        gen_logits = self.final_layer(x)
        gen_logits = gen_logits - tf.reduce_logsumexp(
            gen_logits, axis=-1, keepdims=True)

        idx_tar_src = get_sparse_idx_for_copy(enc_x, target_seq_len=seq_len)
        # (batch_size, target_seq_len, src_seq_len)
        attn_weights = tf.nn.softmax(attention_logits[-1])

        # [batch_size, target_seq_len, tgt_vocab_size]
        copy_logits = tf.log(tf.map_fn(
            fn=lambda y: tf.scatter_nd(
                y[0], y[1], [seq_len, self.tgt_vocab_size]),
            elems=(idx_tar_src, attn_weights), dtype=tf.float32) + 1e-10)
        if copy_mask is not None:
            copy_logits += (copy_mask * -1e9)[None, None, :]
        copy_logits = copy_logits - tf.reduce_logsumexp(
            copy_logits, axis=-1, keepdims=True)

        # the combined features is different with LSTM-PGN
        # LSTM-PGN uses three features, decoder input, decoder state, and
        # context vectors. but for transformer, the decoder state and context
        # vectors are highly correlated, so we use one of them.
        combined_features = tf.concat([x, raw_x], axis=-1)
        combined_features = self.dropout(combined_features, training=training)
        # (batch_size, dec_t, 1)
        logit_gen = self.logit_gen_layer(combined_features)
        # normalized logit of gen
        n_logit_gen = -tf.reduce_logsumexp(
            tf.concat([tf.zeros_like(logit_gen), -logit_gen], axis=-1),
            axis=-1, keepdims=True)
        n_logit_copy = -logit_gen + n_logit_gen

        if self.with_pointer:
            total_logits = tf.reduce_logsumexp(
                tf.stack(
                    [n_logit_gen + gen_logits, n_logit_copy + copy_logits],
                    axis=-1),
                axis=-1)
        else:
            total_logits = gen_logits

        return total_logits, tf.exp(n_logit_gen), gen_logits, copy_logits


def token_logit_masking(token_id: int, vocab_size: int):
    """
    Generate logits to choose the token_id. e.g. with vocab_size = 10,
    token_id = 0, we have
    [  0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]
    plus this mask with normal logits, only token_id=0 can be chose
    """
    assert 0 <= token_id < vocab_size
    mask = tf.concat(
        [tf.fill([token_id], -1e9),
         tf.constant([0.], dtype=tf.float32),
         tf.fill([vocab_size - 1 - token_id], -1e9)],
        axis=0)
    return mask


def categorical_without_replacement(logits, k: int):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/\
    9260#issuecomment-437875125
    also cite here:
    @misc{vieira2014gumbel,
        title = {Gumbel-max trick and weighted reservoir sampling},
        author = {Tim Vieira},
        url = {http://timvieira.github.io/blog/post/2014/08/01/\
        gumbel-max-trick-and-weighted-reservoir-sampling/},
        year = {2014}
    }
    Notice that the logits represent unnormalized log probabilities,
    in the citation above, there is no need to normalized them first to add
    the Gumbel random variant, which surprises me! since I thought it should
    be `logits - tf.reduce_logsumexp(logits) + z`
    """
    z = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, k)
    return indices


def categorical_with_replacement(logits, k: int):
    return tf.random.categorical(logits, num_samples=k, dtype=tf.int32)


def nucleus_renormalization(logits, p=0.95):
    """
    Refer to [Holtzman et al., 2020] for nucleus sampling

    Args:
        logits: last-dimension logits of vocabulary V;
         2D array, [batch, V] or [batch*beam, V]
        p: the cumulative probability bound, default 0.95;

    Returns:
        normalized nucleus logits
    """
    batch_size = tf.shape(logits)[0]
    vocab_size = tf.shape(logits)[1]

    # sort logits
    normalized_logits = logits - tf.reduce_logsumexp(
        logits, axis=-1, keep_dims=True)
    sorted_logits_idx = tf.argsort(
        normalized_logits, axis=-1, direction="DESCENDING")
    idx_dim0 = tf.tile(tf.range(batch_size)[:, None], [1, vocab_size])
    idx_2d = tf.stack([idx_dim0, sorted_logits_idx], axis=-1)
    sorted_logits = tf.gather_nd(normalized_logits, idx_2d)

    # turn sorted logits into mask, true: need to remove, false: nucleus
    sorted_ps = tf.exp(sorted_logits)
    cum_sum_ps = tf.math.cumsum(sorted_ps, axis=-1)
    ps_mask = tf.greater(cum_sum_ps, p)
    # make sure at least the first vocab is valid
    ps_mask = tf.concat(
        [tf.fill([batch_size, 1], False), ps_mask[:, 1:]], axis=-1)

    # turn mask back into original index before sorting
    original_idx = tf.tile(tf.range(vocab_size)[None, :], [batch_size, 1])
    sorted_idx_idx = tf.argsort(
        sorted_logits_idx, axis=-1, direction="ASCENDING")
    revert_idx_2d = tf.stack([idx_dim0, sorted_idx_idx], axis=-1)
    original_idx = tf.gather_nd(original_idx, revert_idx_2d)
    mask_idx_2d = tf.stack([idx_dim0, original_idx], axis=-1)
    original_mask = tf.gather_nd(ps_mask, mask_idx_2d)

    # create and normalize nucleus
    nucleus = tf.where(
        original_mask, x=tf.zeros_like(normalized_logits) + -1e9,
        y=normalized_logits)
    normalized_nucleus = nucleus - tf.reduce_logsumexp(
        nucleus, axis=-1, keep_dims=True)
    return normalized_nucleus


def _dec_beam_search(
        logits, inc_sum_logits, inc_valid_len, batch_size, beam_tgt_len,
        beam_size):
    """
    perform one-step beam search, given token logits, return selected ids
    accumulated logits after the selection.
    Choose top K from p(y_1, ..., y_{i-1}, y_i | X) at each step.
    """
    cond_logits = logits + inc_sum_logits
    cond_ppl = tf.reshape(
        tf.div(cond_logits, tf.dtypes.cast(inc_valid_len, dtype=tf.float32)),
        [batch_size, -1])
    _, predicted_id = tf.math.top_k(
        input=cond_ppl[:, :beam_tgt_len], k=beam_size)
    idx_dim0 = tf.tile(tf.range(batch_size)[:, None], [1, beam_size])
    idx_2d = tf.stack([idx_dim0, predicted_id], axis=-1)
    inc_sum_logits = tf.reshape(
        tf.gather_nd(tf.reshape(cond_logits, [batch_size, -1]), idx_2d),
        [batch_size * beam_size, 1])
    return predicted_id, inc_sum_logits


def _dec_nucleus_sampling(
        logits, inc_sum_logits, batch_size, beam_tgt_len, beam_size,
        temperature):
    """
    perform one-step nucleus sampling, given token logits, return selected ids
    and accumulated logits after the selection.
    Sampling ~ Nucleus( p(y_i | y_1, ..., y_{i-1}, X) )

    Note: this is not real beam search, this function equals to sampling
    a sentence *beam_size* times.
    Sampling with replacement at each step. Note: It is possible to select
    tokens that are filtered out if using sampling w/o replacement.
    E.g. when beam_size = 10, while the best 9 tokens have already large
    enough to build the nucleus.
    """

    # logits: [batch_size * beam_size, vocab_size]
    vocab_size = tf.shape(logits)[1]
    nucleus_logits = nucleus_renormalization(logits / temperature)
    # sample one token in each beam
    # since we don't do actual beam search
    # the predicted_id should add cumulative beam_id * vocab_size
    # to match other beam-generated predicted_id.
    predicted_id = categorical_with_replacement(logits=nucleus_logits, k=1)
    predicted_id = tf.reshape(predicted_id, [batch_size, beam_size])
    cum_beam_id = tf.range(beam_size)[None, :] * vocab_size
    predicted_id = predicted_id + cum_beam_id

    cond_logits = nucleus_logits + inc_sum_logits
    idx_dim0 = tf.tile(tf.range(batch_size)[:, None], [1, beam_size])
    idx_2d = tf.stack([idx_dim0, predicted_id], axis=-1)
    inc_sum_logits = tf.reshape(
        tf.gather_nd(tf.reshape(cond_logits, [batch_size, -1]), idx_2d),
        [batch_size * beam_size, 1])
    return predicted_id, inc_sum_logits


def _dec_sampling(
        logits, inc_sum_logits, batch_size, beam_tgt_len, beam_size,
        temperature):
    """
    perform one-step sampling, given token logits, return selected ids
    and accumulated logits after the selection.
    Sampling ~ p(y_i | y_1, ..., y_{i-1}, X)
    Sampling without replacement at each step
    """
    logits = tf.div(logits, temperature)
    logits = logits - tf.reduce_logsumexp(logits, axis=-1, keep_dims=True)
    predicted_id = categorical_without_replacement(
        logits=tf.reshape(logits, [batch_size, -1])[:, :beam_tgt_len],
        k=beam_size)
    cond_logits = logits + inc_sum_logits
    idx_dim0 = tf.tile(tf.range(batch_size)[:, None], [1, beam_size])
    idx_2d = tf.stack([idx_dim0, predicted_id], axis=-1)
    inc_sum_logits = tf.reshape(
        tf.gather_nd(tf.reshape(cond_logits, [batch_size, -1]), idx_2d),
        [batch_size * beam_size, 1])
    return predicted_id, inc_sum_logits


def decode_next_step(
        decoder, time,
        enc_x, enc_output, training,
        dec_padding_mask, copy_mask,
        batch_size, tgt_vocab_size, eos_id, padding_id,
        beam_size, use_greedy, temperature,
        inc_tar, inc_continue, inc_valid_len, inc_p_gen,
        inc_sum_logits):
    """
    decode one step with beam search
    given inc_tar as the current decoded target sequence
    (batch_size * beam_size), first decode one step with decoder to get
    decoded_logits.
    then mask the decoded_logits:
      1) if continue to decode (i.e. eos never reached) and current time
         reach the max_tar_len, then only EOS is allowed to choose;
      2) if not continue to decode, only PAD is allowed to choose;
      3) default, we don't mask the decoded_logits.
    After get predicted_id, either by sampling method or greedy method,
    we compute 1) beam_id and 2) token_id from predicted_id.
    beam_id indicates which beam to choose, token_id indicates under that
    beam, which token to choose.

    for loop variables, inc_tar, inc_continue, inc_logits, inc_valid_len,
    and inc_p_gen, we first select rows according to beam_id, then pad
    the token_id related info to the end. e.g. given beam_size = 2,
    batch_size = 2, we have inc_tar:

    [[[1, 2, 3],
      [2, 3, 4]],  # --> this beam row will be deleted
     [[9, 8, 7],
      [8, 7, 6]]]
    if beam_id = [[0, 0], [0, 1]], then we choose [1, 2, 3] twice, and
    [9, 8, 7] once, and [8, 7, 6] once, then make the inc_tar to be
    [[[1, 2, 3],
      [1, 2, 3]],
     [[9, 8, 7],
      [8, 7, 6]]]
    then pad new token_id to the end.
    """

    combined_mask = create_decode_masks(inc_tar)
    # decoded_logits:
    # (batch_size * beam_size, target_seq_len, tgt_vocab_size)
    # p_gen: (batch_size * beam_size, target_seq_len, 1)
    decoded_logits, p_gen, _, _ = decoder(
        inc_tar, enc_x, enc_output, training, combined_mask,
        dec_padding_mask, copy_mask)

    # (batch_size * beam_size, tgt_vocab_size)
    curr_logits = decoded_logits[:, -1, :]
    # TODO: here we don't need to normalize the logits, for both beam-search
    #   or not beam-search. Because previously in decoder it was already
    #   normalized due to pointer-generator.
    #   otherwise, it needs to be normalized for beam-search, since we'll group
    #   logits inside a beam together (i.e. the |BE| * |V|) to select from.
    curr_p_gen = p_gen[:, -1, :]

    padding_logit_mask = token_logit_masking(
        token_id=padding_id, vocab_size=decoder.tgt_vocab_size)
    masked_logits = (
        tf.multiply(curr_logits, tf.cast(inc_continue, dtype=tf.float32)) +
        tf.multiply(padding_logit_mask[None, :],
                    1. - tf.cast(inc_continue, dtype=tf.float32)))

    # for the first token decoding, the beam size is 1.
    beam_tgt_len = tf.cond(
        tf.equal(time, 1),
        true_fn=lambda: tgt_vocab_size,
        false_fn=lambda: tgt_vocab_size * beam_size)

    predicted_id, inc_sum_logits = tf.cond(
        use_greedy,
        true_fn=lambda: _dec_beam_search(
            masked_logits, inc_sum_logits, inc_valid_len, batch_size,
            beam_tgt_len, beam_size),
        false_fn=lambda: _dec_nucleus_sampling(
            masked_logits, inc_sum_logits, batch_size, beam_tgt_len,
            beam_size, temperature))

    # (batch_size, beam_size)
    beam_id = predicted_id // tgt_vocab_size
    # (batch_size * beam_size, 1)
    token_id = tf.reshape(
        predicted_id % tgt_vocab_size, (batch_size * beam_size, 1))
    # (batch_size * beam_size, )
    gather_beam_idx = tf.reshape(
        tf.range(batch_size)[:, None] * beam_size + beam_id,
        (batch_size * beam_size, ))

    # create inc tensors according to which beam to choose
    inc_tar_beam = tf.gather(inc_tar, gather_beam_idx)
    inc_tar = tf.concat([inc_tar_beam, token_id], axis=-1)

    inc_continue_beam = tf.gather(inc_continue, gather_beam_idx)
    inc_continue = tf.math.logical_and(
        tf.math.not_equal(token_id, eos_id), inc_continue_beam)
    inc_valid_len_beam = tf.gather(inc_valid_len, gather_beam_idx)
    inc_valid_len = inc_valid_len_beam + tf.dtypes.cast(
        inc_continue, dtype=tf.int32)
    inc_p_gen_beam = tf.gather(inc_p_gen, gather_beam_idx)
    inc_p_gen = tf.concat([inc_p_gen_beam, curr_p_gen], axis=-1)
    return (
        time + 1, inc_tar, inc_continue, inc_valid_len, inc_p_gen,
        inc_sum_logits)


def sequential_decoding(
        decoder, copy_mask, enc_x, enc_output, training, max_tar_len,
        sos_id, eos_id, padding_id,
        use_greedy=True, beam_size=1, temperature=1.):

    batch_size = tf.shape(enc_x)[0]
    src_seq_len = tf.shape(enc_x)[1]
    tgt_vocab_size = decoder.tgt_vocab_size

    inc_time = tf.constant(1)
    inc_tar = tf.fill([batch_size * beam_size, 1], sos_id)
    inc_continue = tf.fill([batch_size * beam_size, 1], True)
    inc_valid_len = tf.fill([batch_size * beam_size, 1], 1)
    inc_p_gen = tf.fill([batch_size * beam_size, 1], 0.)
    inc_sum_logits = tf.fill([batch_size * beam_size, 1], 0.)

    # repeat enc_output and inp w.r.t. beam size
    # (batch_size * beam_size, inp_seq_len, d_model)
    enc_output = tf.reshape(
        tf.tile(enc_output[:, None, :, :], (1, beam_size, 1, 1)),
        (batch_size * beam_size, src_seq_len, -1))
    enc_x = tf.reshape(
        tf.tile(enc_x[:, None, :], (1, beam_size, 1)),
        (batch_size * beam_size, -1))
    dec_padding_mask = create_padding_mask(enc_x)

    def _dec_next_step(
            _time, _tar, _continue, _valid_len, _p_gen, _sum_logits):
        return decode_next_step(
            decoder,
            _time,
            enc_x, enc_output, training,
            dec_padding_mask, copy_mask,
            batch_size, tgt_vocab_size, eos_id, padding_id,
            beam_size, use_greedy, temperature,
            _tar, _continue, _valid_len, _p_gen, _sum_logits)

    def _dec_cond(
            _time, _tar, _continue, _valid_len, _p_gen, _sum_logits):
        return tf.logical_and(
            tf.less_equal(_time, max_tar_len), tf.reduce_any(_continue))

    results = tf.while_loop(
        cond=_dec_cond,
        body=_dec_next_step,
        loop_vars=(
            inc_time,
            inc_tar,
            inc_continue,
            inc_valid_len,
            inc_p_gen,
            inc_sum_logits),
        shape_invariants=(
            inc_time.get_shape(),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, 1]),
            tf.TensorShape([None, 1]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, 1])
        ))

    tar = results[1][:, 1:]
    # valid_len includes the final EOS
    valid_len = tf.squeeze(results[3], axis=-1)
    p_gen = results[4][:, 1:]
    sum_logits = results[5]
    return tar, p_gen, valid_len, sum_logits


class Transformer(tf.keras.Model):
    def __init__(
            self, num_layers, d_model, num_heads, dff, input_vocab_size,
            target_vocab_size, dropout_rate=0.1, with_pointer=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size,
            dropout_rate, with_pointer)

    def call(self, inp, tar, training, copy_mask=None):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = enc_padding_mask
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask, x_seg=None)
        look_ahead_mask = create_decode_masks(tar)
        final_output, p_gen, gen_logits, copy_logits = self.decoder(
            tar, inp, enc_output, training, look_ahead_mask, dec_padding_mask,
            copy_mask)
        return final_output, p_gen, gen_logits, copy_logits

    def decode(
            self, enc_x, training, max_tar_len, sos_id, eos_id, padding_id,
            use_greedy=True, beam_size=1, temperature=1., copy_mask=None):
        # ======= encoding input sentences =======
        enc_padding_mask = create_padding_mask(enc_x)
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(enc_x, training, enc_padding_mask, x_seg=None)
        # ======= end of encoding ======
        return sequential_decoding(
            self.decoder, copy_mask, enc_x, enc_output, training,
            max_tar_len, sos_id, eos_id, padding_id,
            use_greedy, beam_size, temperature)


if __name__ == '__main__':
    def test():
        txf = Transformer(
            num_layers=1, d_model=4, num_heads=2, dff=4,
            input_vocab_size=10, target_vocab_size=10, dropout_rate=0.1)
        inp = tf.constant([[1, 1, 2, 3, 5, 8], [8, 7, 6, 3, 5, 1]])
        res2, res_logits2, p_gen2, inc_valid_len2 = txf.decode(
            inp, training=False, max_tar_len=2, sos_id=0,
            use_greedy=tf.constant(True), beam_size=5, eos_id=9)
        res3, res_logits3, p_gen3, inc_valid_len3 = txf.decode(
            inp, training=False, max_tar_len=3, sos_id=0,
            use_greedy=tf.constant(True), beam_size=5, eos_id=9)
        res4, res_logits4, p_gen4, inc_valid_len4 = txf.decode(
            inp, training=False, max_tar_len=4, sos_id=0,
            use_greedy=tf.constant(True), beam_size=5, eos_id=9)
        res5, res_logits5, p_gen5, inc_valid_len5 = txf.decode(
            inp, training=False, max_tar_len=5, sos_id=0,
            use_greedy=tf.constant(True), beam_size=5, eos_id=9)

        # res2, res_logits2, p_gen2, inc_valid_len2 = txf.decode(
        #     inp, training=False, max_tar_len=10, sos_id=0,
        #     use_greedy=tf.constant(True), beam_size=1, eos_id=9)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(
            [res2, res_logits2, p_gen2, inc_valid_len2,
             res3, res_logits3, p_gen3, inc_valid_len3,
             res4, res_logits4, p_gen4, inc_valid_len4,
             res5, res_logits5, p_gen5, inc_valid_len5
             ])

        print(res[0])
        print(res[1])
        print(res[3])

        print(res[4])
        print(res[5])
        print(res[7])

        print(res[8])
        print(res[9])
        print(res[11])

        print(res[12])
        print(res[13])
        print(res[15])
        # print(res_t)
        # print(inc_valid_len_t)
        # # print(res_logits_t)
        # print(np.sum(res_logits_t, axis=-1))
        # print(res2_t)
        # print(inc_valid_len2_t)
        # # print(res_logits2_t)
        # print(np.sum(res_logits2_t, axis=-1))
    test()
