import torch
import torch.nn.functional as F
from .layers import QMeasureDensity, QFeatureMapRFF, QMeasureDensityEig

class DMKDClassifier(torch.nn.Module):
    def __init__(self, fm_x, dim_x, num_classes=2):
        super(DMKDClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensity(dim_x))
        self.num_samples = torch.tensor(
            torch.zeros((num_classes, ))
        )
    
    def call_train(self, x, y):
        psi = self.fm_x(x) # shape (bs, dim_x)
        rho = self.cp([psi, tf.math.conj(psi)]) # shape (bs, dim_x, dim_x)
        ohy = tf.keras.backend.one_hot(y, self.num_classes)
        ohy = tf.reshape(ohy, (-1, self.num_classes))
        num_samples = tf.squeeze(tf.reduce_sum(ohy, axis=0))
        ohy = tf.expand_dims(ohy, axis=-1) 
        ohy = tf.expand_dims(ohy, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos = tf.cast(ohy, tf.complex64) * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos = tf.reduce_sum(rhos, axis=0) # shape (num_classes, dim_x, dim_x)
        self.num_samples.assign_add(num_samples)
        return rhos

    def forward(self, inputs):
        '''
            1: fit
            2: train_step
            3: call
        '''
        # fit
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign(self.qmd[i].weights[0] / self.num_samples[i])
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = torch.stack(probs, dim=-1)
        posteriors = (posteriors / torch.squeeze(posteriors.sum(), dim=-1))
        return posteriors



class DMKDClassifierSGD(torch.nn.Module):
    def __init__(self, input_dim, dim_x, num_classes, num_eig=0, gamma=1, random_state=None):
        super(DMKDClassifierSGD, self).__init__()
        self.fm_x = QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, 
            gamma=gamma, 
            random_state=random_state
        )
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensityEig(dim_x, num_eig))
        self.gamma = gamma
        self.random_state = random_state
    
    def forward(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = torch.stack(probs, dim=-1)
        posteriors = (posteriors / torch.squeeze(posteriors.sum(),dim=-1))
        return (F.softmax(posteriors)).to(torch.float)
    
    def predict(self, inputs):
        y_pred = self.forward(inputs)

