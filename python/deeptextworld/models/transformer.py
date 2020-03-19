"""
Copied from https://www.tensorflow.org/beta/tutorials/text/transformer
"""

import tensorflow as tf
import numpy as np

from deeptextworld.models.utils import positional_encoding


def create_padding_mask(seq):
    """
    Padding value should be 0.
    :param seq:
    :return:
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # (seq_len, seq_len)
    return mask


def create_decode_masks(tar):
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
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
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        if x_seg is not None:
            x += tf.nn.embedding_lookup(self.seg_embeddings, x_seg)
        if mask is None:
            mask = create_padding_mask(x)

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        # (batch_size, input_seq_len, d_model)
        return x


def get_sparse_idx_tar_for_copy(batch_size, src_seq_len, target_seq_len):
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


def get_sparse_idx_src_for_copy(src, target_seq_len):
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


def get_sparse_idx_for_copy(src, target_seq_len):
    batch_size = tf.shape(src)[0]
    src_seq_len = tf.shape(src)[1]
    idx_src = get_sparse_idx_src_for_copy(src, target_seq_len)
    idx_tar = get_sparse_idx_tar_for_copy(
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
            tgt_vocab_size, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(
            self, x, enc_x, enc_output, training,
            look_ahead_mask, padding_mask, tj_master_mask=None):
        """
        decode with pointer
        :param x: decoder input
        :param enc_x: encoder input
        :param enc_output: encoder encoded result
        :param training:
        :param look_ahead_mask:
        :param padding_mask:
        :param tj_master_mask: select master from tj
        :return:
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

        gen_logits = self.final_layer(x)
        gen_logits = gen_logits - tf.reduce_logsumexp(
            gen_logits, axis=-1, keepdims=True)

        idx_tar_src = get_sparse_idx_for_copy(enc_x, target_seq_len=seq_len)
        # (batch_size, target_seq_len, src_seq_len)
        attn_weights = tf.nn.softmax(attention_logits[-1])
        if tj_master_mask is not None:
            attn_weights = tf.multiply(
                attn_weights,
                tf.cast(tf.expand_dims(tj_master_mask, axis=1), tf.float32))

        # [batch_size, target_seq_len, tgt_vocab_size]
        copy_logits = tf.log(tf.map_fn(
            fn=lambda y: tf.scatter_nd(
                y[0], y[1], [seq_len, self.tgt_vocab_size]),
            elems=(idx_tar_src, attn_weights), dtype=tf.float32) + 1e-10)
        copy_logits = copy_logits - tf.reduce_logsumexp(
            copy_logits, axis=-1, keepdims=True)

        # the combined features is different with LSTM-PGN
        # LSTM-PGN uses three features, decoder input, decoder state, and
        # context vectors. but for transformer, the decoder state and context
        # vectors are highly correlated, so we use one of them.
        combined_features = tf.concat([x, raw_x], axis=-1)
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

    def call(self, inp, tar, training, tj_master_mask=None):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = enc_padding_mask
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask, x_seg=None)
        look_ahead_mask = create_decode_masks(tar)
        final_output, p_gen, gen_logits, copy_logits = self.decoder(
            tar, inp, enc_output, training, look_ahead_mask, dec_padding_mask,
            tj_master_mask=tj_master_mask)
        return final_output, p_gen, gen_logits, copy_logits

    @classmethod
    def categorical_without_replacement(cls, logits, k):
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

    @classmethod
    def categorical_with_replacement(cls, logits, k):
        return tf.random.categorical(logits, num_samples=k, dtype=tf.int32)

    def dec_step(
            self, time,
            enc_x, enc_output, training, tj_master_mask,
            dec_padding_mask, continue_masking, halt_masking,
            batch_size, max_tar_len, tgt_vocab_size, eos_id,
            beam_size, use_greedy, temperature,
            inc_tar, inc_continue, inc_logits, inc_valid_len, inc_p_gen):

        combined_mask = create_decode_masks(inc_tar)
        decoded_logits, p_gen, _, _ = self.decoder(
            inc_tar, enc_x, enc_output, training, combined_mask,
            dec_padding_mask, tj_master_mask=tj_master_mask)
        masking = tf.map_fn(
            lambda c: tf.cond(
                c,
                true_fn=lambda: continue_masking,
                false_fn=lambda: halt_masking),
            tf.squeeze(inc_continue, axis=-1), dtype=tf.float32)
        # (batch_size * beam_size, tgt_vocab_size)
        curr_logits = decoded_logits[:, -1, :]
        curr_p_gen = p_gen[:, -1]
        # mask current logits according to stop seq decoding or not
        masked_logits = tf.reshape(
            tf.where(tf.squeeze(inc_continue, axis=-1), curr_logits, masking),
            (batch_size, -1))
        # for the first token decoding, the beam size is 1.
        beam_tgt_len = tgt_vocab_size * (1 if time == 1 else beam_size)

        # notice that when use greedy, choose the tokens that maximizes
        # \sum p(t_i), while for not using greedy, choose the tokens ~ p(t)
        # predicted_id: (batch_size, beam_size)
        predicted_id = tf.cond(
            use_greedy,
            true_fn=lambda: tf.math.top_k(
                input=masked_logits[:, :beam_tgt_len],
                k=beam_size)[1],
            false_fn=lambda: self.categorical_without_replacement(
                logits=masked_logits[:, :beam_tgt_len] / temperature,
                k=beam_size))

        # (batch_size, beam_size)
        beam_id = predicted_id // tgt_vocab_size
        token_id = tf.reshape(
            predicted_id % tgt_vocab_size, (batch_size * beam_size, 1))
        # (batch_size * beam_size, 1)
        gather_beam_idx = tf.reshape(
            tf.range(batch_size)[:, None] * beam_size + beam_id,
            (batch_size * beam_size, -1))
        # create inc tensors according to which beam to choose
        inc_tar_beam = tf.gather_nd(inc_tar, gather_beam_idx)
        inc_tar = tf.concat([inc_tar_beam, token_id], axis=-1)
        inc_continue_beam = tf.gather_nd(inc_continue, gather_beam_idx)
        curr_continue = tf.math.not_equal(token_id, eos_id)
        inc_continue = tf.concat([inc_continue_beam, curr_continue], axis=-1)
        inc_continue = tf.reduce_all(inc_continue, axis=-1, keepdims=True)
        inc_valid_len_beam = tf.gather_nd(inc_valid_len, gather_beam_idx)
        inc_valid_len = tf.concat(
            [inc_valid_len_beam,
             tf.dtypes.cast(inc_continue, dtype=tf.int32)], axis=-1)
        inc_valid_len = tf.reduce_sum(inc_valid_len, axis=-1, keepdims=True)
        inc_p_gen_beam = tf.gather(inc_p_gen, tf.squeeze(gather_beam_idx, axis=-1))
        inc_p_gen = tf.concat([inc_p_gen_beam, curr_p_gen], axis=-1)
        print(inc_p_gen.shape)
        # (batch_size * beam_size, 2)
        gather_token_idx = tf.concat([gather_beam_idx, token_id], axis=-1)
        print(gather_token_idx.shape)
        selected_logits = tf.gather_nd(
            curr_logits, gather_token_idx)[:, None]
        print(selected_logits.shape)
        inc_logits_beam = tf.gather(inc_logits, tf.squeeze(gather_beam_idx, axis=-1))
        print(inc_logits_beam.shape)
        inc_logits = tf.concat([inc_logits_beam, selected_logits], axis=-1)
        print(inc_logits.shape)
        return (
            time + 1, inc_tar, inc_continue, inc_logits, inc_valid_len,
            inc_p_gen)

    def decode(
            self, enc_x, training, max_tar_len, sos_id, eos_id,
            tj_master_mask=None, use_greedy=True,
            beam_size=1, temperature=1.):
        # ======= encoding input sentences =======
        enc_padding_mask = create_padding_mask(enc_x)
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(enc_x, training, enc_padding_mask, x_seg=None)
        # ======= end of encoding ======

        batch_size = tf.shape(enc_x)[0]
        src_seq_len = tf.shape(enc_x)[1]
        tgt_vocab_size = self.decoder.tgt_vocab_size
        init_time = tf.constant(1)
        inc_tar = tf.fill([batch_size * beam_size, 1], sos_id)
        inc_continue = tf.fill([batch_size * beam_size, 1], True)
        inc_logits = tf.fill([batch_size * beam_size, 1], 0.)
        inc_valid_len = tf.fill([batch_size * beam_size, 1], 1)
        inc_p_gen = tf.fill([batch_size * beam_size, 1], 0.)

        # repeat enc_output and inp w.r.t. beam size
        # (batch_size * beam_size, inp_seq_len, d_model)
        enc_output = tf.reshape(
            tf.tile(enc_output[:, None, :, :], (1, beam_size, 1, 1)),
            (batch_size * beam_size, src_seq_len, -1))
        enc_x = tf.reshape(
            tf.tile(enc_x[:, None, :], (1, beam_size, 1)),
            (batch_size * beam_size, -1))
        dec_padding_mask = create_padding_mask(enc_x)

        # whenever a decoded seq reaches to </S>, stop fan-out the paths with
        # new target tokens by making the 0th token 0, and others -inf.
        halt_masking = tf.concat(
            [tf.constant([0.], dtype=tf.float32),
             tf.fill([tgt_vocab_size - 1], -np.inf)],
            axis=0)
        continue_masking = tf.ones_like(halt_masking)

        def dec_step_with(
                time, target_seq, is_continue, logits, valid_len, p_gen):
            return self.dec_step(
                time,
                enc_x, enc_output, training, tj_master_mask,
                dec_padding_mask, continue_masking, halt_masking,
                batch_size, max_tar_len, tgt_vocab_size, eos_id,
                beam_size, use_greedy, temperature,
                target_seq, is_continue, logits, valid_len, p_gen)

        def dec_cond(time, target_seq, is_continue, logits, valid_len, p_gen):
            return tf.logical_and(
                time <= max_tar_len, tf.reduce_any(is_continue))

        results = tf.while_loop(
            cond=dec_cond,
            body=dec_step_with,
            loop_vars=(
                init_time,
                inc_tar,
                inc_continue,
                inc_logits,
                inc_valid_len,
                inc_p_gen),
            shape_invariants=(
                init_time.get_shape(),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None, None])))

        decoded_idx = results[1][:, 1:]
        decoded_logits = results[3][:, 1:]
        decoded_valid_len = results[4]
        decoded_p_gen = results[5][:, 1:]
        return decoded_idx, decoded_logits, decoded_p_gen, decoded_valid_len


if __name__ == '__main__':
    def test():
        txf = Transformer(
            num_layers=1, d_model=4, num_heads=2, dff=4,
            input_vocab_size=10, target_vocab_size=10, dropout_rate=0.1)
        inp = tf.constant([[1, 1, 2, 3, 5, 8], [8, 7, 6, 3, 5, 1]])
        res, res_logits, p_gen, inc_valid_len = txf.decode(
            inp, training=False, max_tar_len=10, sos_id=0,
            use_greedy=tf.constant(False), beam_size=5, eos_id=9)

        res2, res_logits2, p_gen2, inc_valid_len2 = txf.decode(
            inp, training=False, max_tar_len=10, sos_id=0,
            use_greedy=tf.constant(False), beam_size=1, eos_id=9)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        (res_t, res_logits_t, p_gen_t, inc_valid_len_t,
         res2_t, res_logits2_t, p_gen2_t, inc_valid_len2_t) = sess.run(
            [res, res_logits, p_gen, inc_valid_len,
             res2, res_logits2, p_gen2, inc_valid_len2])
        print(res_t)
        print(res_logits_t)
        print(np.sum(res_logits_t, axis=-1))
        print(res2_t)
        print(res_logits2_t)
        print(np.sum(res_logits2_t, axis=-1))


    test()
