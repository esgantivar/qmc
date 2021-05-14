import tensorflow as tf

from qmc.tf.tn.layers import QMeasureDensityMPS


class DMKDClassifierTNSGD(tf.keras.Model):
    def __init__(self, num_classes=10, n_sites=28 ** 2, d_bond=8, d_phys=2, **kwargs):
        super(DMKDClassifierTNSGD, self).__init__()
        self.d_bond = d_bond
        self.n_sites = n_sites
        self.d_phys = d_phys
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensityMPS(n_sites=n_sites, d_bond=d_bond, d_phys=self.d_phys))

    def call(self, inputs, **kwargs):
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](inputs))
        posteriors = tf.stack(probs, axis=-1)
        return tf.nn.softmax(posteriors)
