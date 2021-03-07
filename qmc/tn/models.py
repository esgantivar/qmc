import tensorflow as tf
from .layers import MPSLayer

class MPSClassifier(tf.keras.Model):

    def __init__(self, n_sites=28 ** 2, d_bond=2, n_output=10, dim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mps = MPSLayer(n_sites=n_sites, d_bond=d_bond, n_output=n_output, dim=dim)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        return self.softmax(self.mps(inputs))

    def get_config(self):
        return super(MPSClassifier, self).get_config()