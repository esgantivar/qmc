import torch
import unittest

from layers import QFeatureMapRFF
from utils import get_moons

class TestRFFLayer(unittest.TestCase):
    def test_rff(self):
        (X, y), (x_train, y_train), (x_test, y_test) = get_moons()
        dim_x = 150
        rff = QFeatureMapRFF(
            input_dim=2,
            dim=dim_x,
            gamma=20,
            random_state=17
        )
        x_rff = rff(torch.tensor(x_train, dtype=torch.float))
        self.assertEqual(x_rff.shape[1], dim_x)
        norms = torch.linalg.norm(x_rff, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones((len(x_rff),))))


if __name__ == '__main__':
    unittest.main()