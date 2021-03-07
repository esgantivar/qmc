import tensorflow as tf
import numpy as np

class MPSLayer(tf.keras.layers.Layer):
    def __init__(self, n_sites, d_bond, n_output, dim=2, dtype=tf.float32):
        super(MPSLayer, self).__init__()
        if n_sites % 2:
            raise NotImplementedError("Number of sites should be even but is "
                                      "{}.".format(n_sites))

        self.n_half = n_sites // 2
        self.dim = dim
        self.left = tf.Variable(self._initializer(self.n_half, self.dim, d_bond),
                                dtype=dtype, trainable=True)
        self.right = tf.Variable(self._initializer(self.n_half, self.dim, d_bond),
                                 dtype=dtype, trainable=True)
        self.middle = tf.Variable(self._initializer(1, n_output, d_bond)[:, 0, :, :],
                                  dtype=dtype, trainable=True)

    @staticmethod
    def _initializer(n_sites, d_phys, d_bond):
        w = np.stack(d_phys * n_sites * [np.eye(d_bond)])
        w = w.reshape((d_phys, n_sites, d_bond, d_bond))
        return w + np.random.normal(0, 1e-2, size=w.shape)

    def call(self, inputs, **kwargs):
        with tf.name_scope('call'):
            left = tf.einsum("slij,bls->lbij", self.left, inputs[:, :self.n_half])
            right = tf.einsum("slij,bls->lbij", self.right, inputs[:, self.n_half:])
            left = self.reduction(left)
            right = self.reduction(right)
        return tf.einsum("bij,cjk,bki->bc", left, self.middle, right)

    @staticmethod
    def reduction(tensor):
        with tf.name_scope('reduction'):
            size = int(tensor.shape[0])
            while size > 1:
                half_size = size // 2
                nice_size = 2 * half_size
                leftover = tensor[nice_size:]
                tensor = tf.matmul(tensor[0:nice_size:2], tensor[1:nice_size:2])
                tensor = tf.concat([tensor, leftover], axis=0)
                size = half_size + int(size % 2 == 1)
        return tensor[0]

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape