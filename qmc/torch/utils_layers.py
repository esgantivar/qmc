import torch

class CrossProduct(torch.nn.Module):
    def __init__(self):
        super(CrossProduct, self).__init__()
        self.built = False
    
    def _build(self, inputs):
        idx1 = 'abcdefghij'
        idx2 = 'klmnopqrst'
        self.eins_eq = ('...' + idx1[:len(inputs[0].shape) - 1] + ',' +
                        '...' + idx2[:len(inputs[1].shape) - 1] + '->' +
                        '...' + idx1[:len(inputs[0].shape) - 1] +
                        idx2[:len(inputs[1].shape) - 1])
        self.built = True 

    def forward(self, inputs):
        if len(inputs) != 2:
            raise ValueError("A CrossProduct layer should be called on exactly 2 inputs")
        if not self.built:
            self._build(inputs)
        return torch.einsum(
            self.eins_eq,
            inputs[0],
            inputs[1]
        )
