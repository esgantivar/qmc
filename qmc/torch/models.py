import torch
from .layers import QMeasureDensity, QFeatureMapRFF, QMeasureDensityEig, QMeasureDensityTNEig
from .utils_layers import CrossProduct


class DMKDClassifier(torch.nn.Module):
    def __init__(self, fm_x, dim_x, num_classes=2):
        super(DMKDClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensity(dim_x))
        self.cp = CrossProduct()
        self.num_samples = torch.zeros((num_classes,))

    def calc_batch_train(self, x, y):
        psi = self.fm_x(x)  # shape (bs, dim_x)
        rho = self.cp([psi, torch.conj(psi)])  # shape (bs, dim_x, dim_x)
        ohy = torch.reshape(y, (-1, self.num_classes))
        num_samples = torch.squeeze(torch.sum(ohy, dim=0))
        ohy = torch.unsqueeze(ohy, dim=-1)
        ohy = torch.unsqueeze(ohy, dim=-1)  # shape (bs, num_classes, 1, 1)
        rhos = ohy * torch.unsqueeze(rho, dim=1)  # shape (bs, num_classes, dim_x, dim_x)
        rhos = torch.sum(rhos, dim=0)  # shape (num_classes, dim_x, dim_x)
        self.num_samples += num_samples
        return rhos

    def calc_train(self, x, y):
        rhos = self.calc_batch_train(x, y)
        for i in range(self.num_classes):
            self.qmd[i].rho += rhos[i]
        for i in range(self.num_classes):
            self.qmd[i].rho /= self.num_samples[i]

    def forward(self, inputs):
        # Quantum feature mapping
        psi_x = self.fm_x(inputs)
        probs = []
        # Prediction operator
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = torch.stack(probs, dim=-1)
        posteriors = (posteriors / torch.unsqueeze(torch.sum(posteriors, dim=-1), dim=-1))
        # Partial trace => argmax over posteriors
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
        posteriors = (posteriors / torch.unsqueeze(torch.sum(posteriors, dim=-1), dim=-1))
        return posteriors

    def predict(self, inputs):
        y_pred = self.forward(inputs)


class DMKDClassifierTNSGD(DMKDClassifierSGD):
    def __init__(self, input_dim, dim_x, num_classes, num_eig=0, gamma=1, random_state=None):
        super(DMKDClassifierTNSGD, self).__init__(input_dim, dim_x, num_classes, num_eig, gamma, random_state)
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
            self.qmd.append(QMeasureDensityTNEig(dim_x, num_eig))
        self.gamma = gamma
        self.random_state = random_state