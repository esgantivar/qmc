import tensorflow as tf
import tensornetwork as tn
import numpy as np


class QMeasureDensityMPS(tf.keras.layers.Layer):
    def __init__(self, n_sites, d_bond, d_phys, **kwargs):
        super(QMeasureDensityMPS, self).__init__(**kwargs)
        self.single_rank = n_sites // 2
        self.n_sites = n_sites
        self.d_bond = d_bond
        self.d_phys = d_phys

        '''
         |   |   |   |
        LB- -L- -R- -RB 
             |   |    
        '''
        self.left_b = tf.Variable(self._boundary(self.d_phys, self.d_bond), name=f'mps_left_b', trainable=True,
                                  dtype=tf.float32)
        self.left = tf.Variable(self._middle(self.d_phys, self.single_rank - 1, self.d_bond), name=f'mps_left',
                                trainable=True, dtype=tf.float32)
        self.right = tf.Variable(self._middle(self.d_phys, self.single_rank - 1, self.d_bond), name=f'mps_right',
                                 trainable=True, dtype=tf.float32)
        self.right_b = tf.Variable(self._boundary(self.d_phys, self.d_bond), name=f'mps_right_b', trainable=True,
                                   dtype=tf.float32)

    def call(self, inputs, **kwargs):
        def f(input_vec, left_b, left, right, right_b):
            lb_n = tn.Node(left_b)
            ln = tn.Node(left)
            rb_n = tn.Node(right_b)
            rn = tn.Node(right)

            ilb = tn.Node(input_vec[0])
            il = tn.Node(input_vec[1:self.single_rank])
            ir = tn.Node(input_vec[self.single_rank:self.n_sites - 1])
            irb = tn.Node(input_vec[self.n_sites - 1])

            # Create TN
            ilb[0] ^ lb_n[1]
            lb_n[0] ^ ln[1]
            ln[0] ^ il[0]
            ln[3] ^ il[1]

            irb[0] ^ rb_n[1]
            rb_n[0] ^ rn[1]
            rn[0] ^ ir[0]
            rn[3] ^ ir[1]

            ln[2] ^ rn[2]

            # Contract
            ans = tn.contractors.greedy(tn.reachable(ln))
            return ans.tensor

        # Norm Phys Dimension
        self.left_b[2].assign(self.left_b[2] / tf.linalg.norm(self.left_b[2]))
        self.left[3].assign(self.left[3] / tf.linalg.norm(self.left[3]))
        self.right[3].assign(self.right[3] / tf.linalg.norm(self.right[3]))
        self.right_b[2].assign(self.right_b[2] / tf.linalg.norm(self.right_b[2]))

        result = tf.vectorized_map(
            lambda vec: f(vec, self.left_b, self.left, self.right, self.right_b),
            inputs
        )
        return result

    @staticmethod
    def _middle(d_phys, n_sites, bond):
        return np.random.normal(0, 1e-2, size=(n_sites, bond, bond, d_phys))

    @staticmethod
    def _boundary(d_phys, bond):
        return np.random.normal(0, 1e-2, size=(bond, d_phys))
