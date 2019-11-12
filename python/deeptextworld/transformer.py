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
      output (a.k.a. context vectors), attention_weights
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
    return output, attention_weights


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

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # (batch_size, seq_len_q, seq_len_k)
        attention_weights = tf.reduce_sum(attention_weights, axis=1)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


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

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block2


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

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.p_gen_dense = tf.keras.layers.Dense(
            units=1, use_bias=True, activation=tf.sigmoid)
        self.final_layer = tf.keras.layers.Dense(tgt_vocab_size)

    def call(self, x, enc_x, enc_output, training, look_ahead_mask, padding_mask, with_pointer=False):
        seq_len = tf.shape(x)[1]
        attention_weights = []

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        before_dec = x

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, vanilla_attn = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights.append(vanilla_attn)

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
        total_prob = self.final_layer(x)

        if with_pointer:
            vanilla_attention = attention_weights[-1]

            batch_size = tf.shape(vanilla_attention)[0]
            dec_t = tf.shape(vanilla_attention)[1]
            attn_len = tf.shape(vanilla_attention)[2]

            dec = tf.range(0, limit=dec_t)  # [dec]
            dec = tf.expand_dims(dec, axis=-1)  # [dec, 1]
            dec = tf.tile(dec, [1, attn_len])  # [dec, atten_len]
            dec = tf.expand_dims(dec, axis=0)  # [1, dec, atten_len]
            dec = tf.tile(dec, [batch_size, 1, 1])  # [batch_size, dec, atten_len]

            enc_x = tf.expand_dims(enc_x, axis=1)  # [batch_size, 1, atten_len]
            enc_x = tf.tile(enc_x, [1, dec_t, 1])  # [batch_size, dec, atten_len]
            enc_x = tf.stack([dec, enc_x], axis=3)

            copy_output = tf.map_fn(
                fn=lambda y: tf.scatter_nd(
                    y[0], y[1], [dec_t, self.tgt_vocab_size]),
                elems=(enc_x, vanilla_attention), dtype=tf.float32)

            # vanilla_attention_expanded = tf.expand_dims(
            #     vanilla_attention, axis=3)
            # inp_idx = tf.expand_dims(
            #     tf.one_hot(indices=enc_x, depth=self.tgt_vocab_size), axis=1)
            # copy_output = tf.reduce_sum(
            #     tf.multiply(vanilla_attention_expanded, inp_idx), axis=2)
            context_vectors = tf.matmul(vanilla_attention, enc_output)
            combined_features = tf.concat(
                [x, before_dec, context_vectors], axis=-1)
            p_gen = self.p_gen_dense(combined_features)

            total_prob = p_gen * total_prob + (1 - p_gen) * copy_output
        # x.shape == (batch_size, target_seq_len, tgt_vocab_size)
        return total_prob


class Transformer(tf.keras.Model):
    def __init__(
            self, num_layers, d_model, num_heads, dff, input_vocab_size,
            target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.tgt_vocab_size = target_vocab_size

    def call(
            self, inp, tar, training, max_tar_len, sos_id, eos_id, temperature):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = enc_padding_mask
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, None, training, enc_padding_mask)

        if training:
            look_ahead_mask = create_decode_masks(tar)
            final_output = self.decoder(
                tar, inp, enc_output, training, look_ahead_mask, dec_padding_mask,
                with_pointer=True)
        else:
            batch_size = tf.shape(inp)[0]
            inc_tar = tf.fill([batch_size, 1], sos_id)
            last_predictions = []
            for i in range(max_tar_len):
                combined_mask = create_decode_masks(inc_tar)
                final_prob = self.decoder(
                    inc_tar, inp, enc_output, training, combined_mask,
                    dec_padding_mask, with_pointer=True)
                predictions = final_prob[:, -1:, :]
                last_predictions.append(predictions)
                predicted_id = tf.multinomial(
                    predictions[:, 0, :] / temperature,
                    1, output_dtype=tf.int32)
                # predicted_id = tf.cast(
                #     tf.argmax(predictions, axis=-1), tf.int32)
                # concatentate the predicted_id to the output which is given to the decoder
                # as its input.
                inc_tar = tf.concat([inc_tar, predicted_id], axis=-1)
                # return the result if the predicted_id is equal to the end token
                if predicted_id == eos_id:
                    break
            final_output = tf.concat(last_predictions, axis=1)
            src_paddings = tf.constant(
                [[0, 0], [0, max_tar_len-len(last_predictions)], [0, 0]])
            final_output = tf.pad(
                final_output, paddings=src_paddings, mode="CONSTANT")
        return final_output


if __name__ == '__main__':
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform(
        (1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out_res, attn_res = sess.run([out, attn])
    print(out_res)
    print(attn_res)
