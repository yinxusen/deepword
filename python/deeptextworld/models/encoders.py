import tensorflow as tf
from deeptextworld.models.utils import positional_encoding
from deeptextworld.models.transformer import Encoder, create_padding_mask


class CnnEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, filter_sizes, num_filters, rate=0.4):
        super(CnnEncoderLayer, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.d_model = len(filter_sizes) * num_filters

        self.dropout = tf.keras.layers.Dropout(rate)

        self.conv_filters = []
        for i, fs in enumerate(self.filter_sizes):
            conv = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[fs, self.d_model],
                padding="VALID",
                strides=[1, 1],
                activation=tf.tanh,
                use_bias=True,
                data_format="channel_last")
            self.conv_filters.append(conv)

    def call(self, x, training=None):
        layer_outputs = []
        for fs, conv in zip(self.filter_sizes, self.conv_filters):
            src_paddings = tf.constant(
                [[0, 0], [fs - 1, 0], [0, 0], [0, 0]])
            src_w_pad = tf.pad(
                x, paddings=src_paddings, mode="CONSTANT")
            layer_outputs.append(conv(src_w_pad))
        layer_outputs = tf.squeeze(tf.concat(layer_outputs, axis=-1), axis=[2])
        layer_outputs = self.dropout(layer_outputs, training=training)
        return layer_outputs


class CnnEncoder(tf.keras.layers.Layer):
    def __init__(
            self, filter_sizes, num_filters, num_layers, input_vocab_size,
            rate=0.4):
        super(CnnEncoder, self).__init__()

        self.d_model = len(filter_sizes) * num_filters
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, self.d_model)
        self.seg_embeddings = tf.keras.layers.Embedding(
            input_dim=2, output_dim=self.d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        self.seg_embeddings = tf.stack(
            [tf.zeros(self.d_model), tf.ones(self.d_model)],
            name="seg_embeddings")

        self.enc_layers = [
            CnnEncoderLayer(filter_sizes, num_filters, rate)
            for _ in range(num_layers)]

    def call(self, x, x_seg=None, training=None):
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        if x_seg is not None:
            x += tf.nn.embedding_lookup(self.seg_embeddings, x_seg)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        pooled = tf.reduce_max(x, axis=1)
        return x, pooled


class LstmEncoder(tf.keras.layers.Layer):
    def __init__(self, num_units, num_layers, input_vocab_size, d_model):
        super(LstmEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, self.d_model)

        self.enc_layers = [
            tf.keras.layers.LSTM(
                units=num_units, activation=tf.tanh,
                return_sequences=True, return_state=True)
            for _ in range(num_layers)]

    def call(self, x, training=None):
        # index-0 is paddings
        mask = tf.math.equal(x, 0)
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        state = None
        for i in range(self.num_layers):
            output, x, state = self.enc_layers[i](
                x, mask=mask, training=training)
        return x, state


class TxEncoder(tf.keras.layers.Layer):
    def __init__(
            self, num_layers, d_model, num_heads, dff, input_vocab_size,
            rate=0.1):
        super(TxEncoder, self).__init__()

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, rate)

    def call(self, x, x_seg=None, training=None):
        enc_padding_mask = create_padding_mask(x)
        x = self.encoder(
            x, x_seg=x_seg, training=training, mask=enc_padding_mask)
        pooled = tf.reduce_max(x, axis=1)
        return x, pooled
