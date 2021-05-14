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
        mps = [
            self._boundary(self.d_phys, self.d_bond),
            self._middle(self.d_phys, self.single_rank - 1, self.d_bond),
            self._boundary(self.d_phys, self.d_bond),
            self._middle(self.d_phys, self.single_rank - 1, self.d_bond),
        ]

        self.mps_var = [tf.Variable(node, name=f'mps{i}', trainable=True, dtype=tf.float32) for (i, node) in
                        enumerate(mps)]

    def call(self, inputs, **kwargs):
        def f(input_vec, mps_var):
            l_1n = tn.Node(mps_var[0])
            ln = tn.Node(mps_var[1])
            r_1n = tn.Node(mps_var[2])
            rn = tn.Node(mps_var[3])

            il_1 = tn.Node(input_vec[0])
            il = tn.Node(input_vec[1:self.single_rank])
            ir = tn.Node(input_vec[self.single_rank:self.n_sites - 1])
            ir_1 = tn.Node(input_vec[self.n_sites - 1])

            # Create TN
            il_1[0] ^ l_1n[1]
            l_1n[0] ^ ln[1]
            ln[0] ^ il[0]
            ln[3] ^ il[1]

            ir_1[0] ^ r_1n[1]
            r_1n[0] ^ rn[1]
            rn[0] ^ ir[0]
            rn[3] ^ ir[1]

            ln[2] ^ rn[2]

            # Contract
            ans = tn.contractors.greedy(tn.reachable(ln))
            return ans.tensor

        result = tf.vectorized_map(
            lambda vec: f(vec, self.mps_var),
            inputs
        )
        return result

    @staticmethod
    def _middle(d_phys, n_sites, bond):
        return np.random.normal(0, 1e-2, size=(n_sites, bond, bond, d_phys))

    @staticmethod
    def _boundary(d_phys, bond):
        return np.random.normal(0, 1e-2, size=(bond, d_phys))
