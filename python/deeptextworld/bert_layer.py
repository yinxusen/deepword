import tensorflow as tf
import tensorflow_hub as hub


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.bert_module = None
        self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    def build(self, input_shape):
        self.bert_module = hub.Module(
            self.bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert_module.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if
                          not "/cls/" in var.name]

        # Select how many layers to fine tune
        if self.n_fine_tune_layers <= 0:
            trainable_vars = []
        else:
            trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert_module.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert_module(
            inputs=bert_inputs, signature="tokens", as_dict=True)
        result = result["sequence_output"]
        return result

    # TODO: wrong output shape
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_size
