"""
Copied from https://www.tensorflow.org/beta/tutorials/text/transformer
"""

import tensorflow as tf
import numpy as np


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


def create_padding_mask(seq):
    """
    Padding value should be 0.
    :param seq:
    :return:
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


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

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, scaled_attention_logits


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
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # (batch_size, num_heads, seq_len_q, depth)
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        (scaled_attention, scaled_attention_logits
         ) = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, seq_len_k)
        attn_logits = tf.reduce_sum(scaled_attention_logits, axis=1)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attn_logits


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, _ = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # attention weights from decoder output to encoder outputs
        # attn_logits: (batch_size, target_seq_len, enc_output_seq_len)
        attn2, attn_logits = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_logits


class Encoder(tf.keras.layers.Layer):
    def __init__(
            self, num_layers, d_model, num_heads, dff, input_vocab_size,
            rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        self.seg_embeddings = tf.stack(
            [tf.zeros(self.d_model), tf.ones(self.d_model)],
            name="seg_embeddings")

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, x_seg, training, mask):
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

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(
            self, num_layers, d_model, num_heads, dff, tgt_vocab_size,
            rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.tgt_vocab_size = tgt_vocab_size

        self.embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = positional_encoding(tgt_vocab_size, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.logit_gen_layer = tf.keras.layers.Dense(units=1, use_bias=True)
        self.final_layer = tf.keras.layers.Dense(
            tgt_vocab_size, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(
            self, x, enc_x, enc_output, training,
            look_ahead_mask, padding_mask, with_pointer=False):
        """
        decode one token
        :param x: decoder input
        :param enc_x: encoder input
        :param enc_output: encoder encoded result
        :param training:
        :param look_ahead_mask:
        :param padding_mask:
        :param with_pointer:
        :return:
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        before_dec = x

        x = self.dropout(x, training=training)

        # move first layer out of loop since we need attn_logits
        x, attn_logits = self.dec_layers[0](
            x, enc_output, training, look_ahead_mask, padding_mask)
        for i in range(1, self.num_layers):
            x, attn_logits = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask)

        """
        :param enc_inp: encoder input, batch_size * max_action_len
        :param dec_inp_emb: decoder input (a.k.a. target), batch_size * max_action_len * d_model
        :param attn_distribution: attention distribution from encoded states to decoded states,
            batch_size * max_action_len * max_action_len
        :param context_vector: context vector by weighted summing of attention distribution,
            batch_size * max_action_len * d_model
        :param dec_state: output from decoder, batch_size * max_action_len * d_model
        :param dec_out_probability: decoder output probability for vocab sampling,
            batch_size * max_action_len * tgt_vocab_size
        :return:
        """
        gen_logits = self.final_layer(x)
        gen_logits = gen_logits - tf.reduce_logsumexp(gen_logits, axis=-1)

        attn_weights = tf.nn.softmax(attn_logits)
        batch_size = tf.shape(attn_logits)[0]
        dec_t = tf.shape(attn_logits)[1]
        attn_len = tf.shape(attn_logits)[2]

        dec = tf.range(0, limit=dec_t)  # [dec]
        dec = tf.expand_dims(dec, axis=-1)  # [dec, 1]
        dec = tf.tile(dec, [1, attn_len])  # [dec, atten_len]
        dec = tf.expand_dims(dec, axis=0)  # [1, dec, atten_len]
        dec = tf.tile(dec, [batch_size, 1, 1])  # [batch_size, dec, atten_len]

        enc_x = tf.expand_dims(enc_x, axis=1)  # [batch_size, 1, atten_len]
        enc_x = tf.tile(enc_x, [1, dec_t, 1])  # [batch_size, dec, atten_len]
        enc_x = tf.stack([dec, enc_x], axis=3)  # [batch_size, dec, atten_len, 2]

        # [batch_size, dec, tgt_vocab_size]
        copy_logits = tf.log(tf.map_fn(
            fn=lambda y: tf.scatter_nd(
                y[0], y[1], [dec_t, self.tgt_vocab_size]),
            elems=(enc_x, attn_weights), dtype=tf.float32) + 1e-10)
        copy_logits = copy_logits - tf.reduce_logsumexp(copy_logits, axis=-1)

        combined_features = tf.concat([x, before_dec, attn_logits], axis=-1)
        logit_gen = self.logit_gen_layer(combined_features)
        # normalized logit of gen
        n_logit_gen = -tf.reduce_logsumexp([0, -logit_gen])
        n_logit_copy = -logit_gen + n_logit_gen

        if with_pointer:
            total_logits = tf.reduce_logsumexp(
                [n_logit_gen + gen_logits, n_logit_copy + copy_logits])
        else:
            total_logits = gen_logits

        return total_logits, tf.exp(n_logit_gen), gen_logits, copy_logits


class Transformer(tf.keras.Model):
    def __init__(
            self, num_layers, d_model, num_heads, dff, input_vocab_size,
            target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, rate)

    def call(self, inp, tar, training):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = enc_padding_mask
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, None, training, enc_padding_mask)
        look_ahead_mask = create_decode_masks(tar)
        final_output, p_gen, gen_logits, copy_logits = self.decoder(
            tar, inp, enc_output, training, look_ahead_mask,
            dec_padding_mask, with_pointer=True)
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

    def decode(
            self, inp, training, max_tar_len, sos_id, eos_id,
            use_greedy=True, beam_size=1, temperature=1.):
        """
        decode indexes given input sentences.
        :param inp: input sentence (batch_size, seq_len) for the encoder
        :param training: bool, training or inference
        :param max_tar_len: maximum target length, int32
        :param sos_id: <S> id
        :param eos_id: </S> id
        :param use_greedy: tf.bool, use greedy or sampling to decode per token
        :param beam_size: tf.int, for beam search
        :param temperature: tf.float, to control sampling randomness
        :return:
          decoded indexes, (batch_size * beam_size, max_tar_len) int32
          decoded logits, (batch_size * beam_size, max_tar_len) float32
          probability of generation (copy otherwise),
            (batch_size * beam_size, max_tar_len) float32
        """
        # ======= encoding input sentences =======
        enc_padding_mask = create_padding_mask(inp)
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, None, training, enc_padding_mask)
        # ======= end of encoding ======

        # ======== decoding output sentences ========
        tgt_vocab_size = self.decoder.tgt_vocab_size
        batch_size = tf.shape(inp)[0]
        inp_seq_len = tf.shape(inp)[1]

        # recurring tensors for
        # target sequence, target logits, continue generation or not,
        # and valid length for each output, based on the position of </S>.
        inc_tar = tf.fill([batch_size * beam_size, 1], sos_id)
        inc_logits = tf.fill([batch_size * beam_size, 1], 0.)
        inc_continue = tf.fill([batch_size * beam_size, 1], True)
        inc_valid_len = tf.fill([batch_size * beam_size, 1], 1)

        # repeat enc_output and inp w.r.t. beam size
        # (batch_size * beam_size, inp_seq_len, d_model)
        enc_output = tf.reshape(
            tf.tile(enc_output[:, None, :, :], (1, beam_size, 1, 1)),
            (batch_size * beam_size, inp_seq_len, -1))
        inp = tf.reshape(
            tf.tile(inp[:, None, :], (1, beam_size, 1)),
            (batch_size * beam_size, -1))
        dec_padding_mask = create_padding_mask(inp)

        # whenever a decoded seq reaches to </S>, stop fan-out the paths with
        # new target tokens by making the 0th token 0, and others -inf.
        halt_masking = tf.concat(
            [tf.constant([0.], dtype=tf.float32),
             tf.fill([tgt_vocab_size - 1], -np.inf)],
            axis=0)
        continue_masking = tf.ones_like(halt_masking)

        p_gen = []
        for i in range(1, max_tar_len+1):
            combined_mask = create_decode_masks(inc_tar)
            final_prob, p_gen_latest, _, _ = self.decoder(
                inc_tar, inp, enc_output, training, combined_mask,
                dec_padding_mask, with_pointer=True)
            masking = tf.map_fn(
                lambda c: tf.cond(
                    c,
                    true_fn=lambda: continue_masking,
                    false_fn=lambda: halt_masking),
                tf.squeeze(inc_continue, axis=-1), dtype=tf.float32)
            # (batch_size * beam_size, tgt_vocab_size)
            curr_logits = final_prob[:, -1, :]
            # mask current logits according to stop seq decoding or not
            masked_logits = tf.reshape(tf.where(
                tf.squeeze(inc_continue, axis=-1), curr_logits, masking),
                (batch_size, -1))
            # How to remove the same w/ same weights?
            p_gen.append(p_gen_latest[:, -1])
            # for the first token decoding, the beam size is 1.
            beam_tgt_len = tgt_vocab_size * (1 if i == 1 else beam_size)
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
                predicted_id % tgt_vocab_size,
                (batch_size * beam_size, -1))
            # (batch_size * beam_size, 1)
            gather_beam_idx = tf.reshape(
                tf.range(batch_size)[:, None] * beam_size + beam_id,
                (batch_size * beam_size, -1))
            # create inc tensors according to which beam to choose
            inc_tar_beam = tf.gather_nd(inc_tar, gather_beam_idx)
            inc_tar = tf.concat([inc_tar_beam, token_id], axis=-1)
            inc_continue_beam = tf.gather_nd(inc_continue, gather_beam_idx)
            inc_continue = tf.math.logical_and(
                tf.math.not_equal(token_id, eos_id), inc_continue_beam)
            inc_valid_len_beam = tf.gather_nd(inc_valid_len, gather_beam_idx)
            inc_valid_len = inc_valid_len_beam + tf.dtypes.cast(
                inc_continue, dtype=tf.int32)

            # (batch_size * beam_size, 2)
            gather_token_idx = tf.concat([gather_beam_idx, token_id], axis=-1)
            selected_logits = tf.gather_nd(
                curr_logits, gather_token_idx)[:, None]
            inc_logits_beam = tf.gather_nd(inc_logits, gather_beam_idx)
            inc_logits = tf.concat([inc_logits_beam, selected_logits], axis=-1)
            # end for
        # ======= end of decoding ======
        p_gen = tf.concat(p_gen, axis=1)
        decoded_idx = inc_tar[:, 1:]
        decoded_logits = inc_logits[:, 1:]
        return decoded_idx, decoded_logits, p_gen, inc_valid_len


if __name__ == '__main__':
    def test():
        txf = Transformer(
            num_layers=1, d_model=4, num_heads=2, dff=4,
            input_vocab_size=10, target_vocab_size=10, rate=0.1)
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
