import tensorflow as tf


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
        return state
