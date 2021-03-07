import sys
sys.path.insert(0, "../")

import numpy as np
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler
from sklearn.datasets import make_blobs, make_moons, make_circles
from typeguard import typechecked
from qmc.tf.layers import QFeatureMapRFF


class QFeatureMapRFFTN(tf.keras.layers.Layer):
    @typechecked
    def __init__(self, input_dim: int, dim: int, gamma: float, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state
    
    def build(self, input_shape):
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state
        )
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=True,
            name='rff_weights'
        )
        self.offset = tf.Variable(
            initial_value=rbf_sampler.random_offset_,
            dtype=tf.float32,
            trainable=True,
            name='offset'
        )
        self.built = True
    
    def call(self, inputs):
        vals = tf.matmul(inputs, self.rff_weights) + self.offset
        cos_vals = tf.math.cos(vals) * tf.sqrt(2. / self.dim)
        sin_vals = tf.math.sin(vals) * tf.sqrt(2. / self.dim)
        cos_norms = tf.linalg.norm(cos_vals, axis=1)
        sin_norms = tf.linalg.norm(sin_vals, axis=1)
        cos_psi = cos_vals / tf.expand_dims(cos_norms, axis=-1)
        sin_psi = cos_vals / tf.expand_dims(sin_norms, axis=-1)
        return tf.stack([cos_psi, sin_psi], axis=2)
    
    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] )
x_train = x_train / 255.

fm = QFeatureMapRFFTN(input_dim=28**2, dim=20, gamma=0.0001)
# fm.build((1, 28**2))
print(fm([x_train[0]]))