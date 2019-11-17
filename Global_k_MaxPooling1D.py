from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf

class Global_k_MaxPooling1D(Layer):
    """ Original version was by Alexander Backus https://github.com/keras-team/keras/issues/373
        I added the last line to generate GlobalMaxPooling1D-like tensor.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[1] * self.k))
    def call(self, inputs):
        inputs = tf.transpose(inputs, [0, 2, 1]) # To be sorted along the sequence (time) dimension
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=True, name=None)[0]
        top_k = tf.transpose(top_k, [0, 2, 1]) # To generate GlbalMaxPooling1D-like tensor  
        return Flatten()(top_k)
