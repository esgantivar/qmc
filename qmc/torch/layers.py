import torch
import numpy as np
import torch.nn.functional as F
from sklearn.kernel_approximation import RBFSampler

class QFeatureMapRFF(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim: int = 100,
        gamma: float = 1,
        random_state=None
    ):
        super(QFeatureMapRFF, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state
        self._build()
    
    def _build(self):
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = torch.nn.Parameter(
            torch.tensor(
                rbf_sampler.random_weights_,
                dtype=torch.float32
            )
        )
        self.offset = torch.nn.Parameter(
            torch.tensor(
                rbf_sampler.random_offset_,
                dtype=torch.float32
            )
        )
        self.built = True
    
    def forward(self, input_data):
        vals = torch.matmul(input_data, self.rff_weights) + self.offset
        vals = torch.cos(vals)
        vals = vals * torch.sqrt(torch.tensor(2. / self.dim, dtype=torch.float32))
        norms = torch.linalg.norm(vals, dim=1)
        psi = vals / torch.unsqueeze(norms, dim=-1)
        return psi


class QMeasureDensity(torch.nn.Module):
    def __init__(
            self,
            dim_x: int
    ):
        self.dim_x = dim_x
        super(QMeasureDensity, self).__init__()
        self._build()

    def _build(self):
        self.rho = rho = torch.nn.Parameter(torch.zeros(self.dim_x, self.dim_x))
        self.built = True

    def forward(self, inputs):
        oper = torch.einsum(
            '...i,...j->...ij',
            inputs,
            torch.conj(inputs)
        ) # shape (b, nx, nx)
        rho_res = torch.einsum(
            '...ik, km, ...mi -> ...',
            oper, 
            self.rho, 
            oper
        )  # shape (b, nx, ny, nx, ny)
        return rho_res

class QMeasureDensityEig(torch.nn.Module):
    def __init__(self, dim_x: int, num_eig: int = 0):
        super(QMeasureDensityEig, self).__init__()
        self.dim_x = dim_x
        if num_eig < 1:
            num_eig = dim_x
        self.num_eig = num_eig
        self._build()
    
    def _build(self):
        self.eig_vec = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.dim_x, self.num_eig)
            )
        )
        self.eig_val = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.empty((self.num_eig, ))
            )
        )

    def forward(self, inputs):
        norms = torch.squeeze(torch.linalg.norm(self.eig_vec, dim=0), dim=0)
        eig_vec = self.eig_vec / norms
        eig_val = F.relu(self.eig_val)
        eig_val = eig_val / eig_val.sum()
        rho_h = torch.matmul(eig_vec, torch.diag(torch.sqrt(eig_val)))
        rho_h = torch.matmul(torch.conj(inputs), rho_h)
        rho_res = torch.einsum(
            '...i, ...i -> ...',
            rho_h, torch.conj(rho_h)
        ) # shape (b,)
        return rho_res