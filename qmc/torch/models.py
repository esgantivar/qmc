import torch
from .layers import QMeasureDensity

class DMKDClassifier(torch.nn.Module):
    def __init__(self, fm_x, dim_x, num_classes=2):
        super(DMKDClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensity(dim_x))

    def forward(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = torch.stack(probs, dim=-1)
        posteriors = (posteriors / torch.squeeze(posteriors.sum(), dim=-1))
        return posteriors