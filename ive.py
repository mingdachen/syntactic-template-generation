
import torch
import numpy as np
import scipy.special
from numbers import Number


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, v, z):

        assert isinstance(v, Number), 'v must be a scalar'

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.detach().cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=np.double)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=np.double)
        else:   # v > 0
            output = scipy.special.ive(v, z_cpu, dtype=np.double)
#         else:
#             print(v, type(v), np.isclose(v, 0))
#             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        return torch.tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1].double()
        return None, (grad_output.double() *
                      (ive(self.v - 1, z) - ive(self.v, z) *
                       (self.v + z) / z)).float()


class Ive(torch.nn.Module):

    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


ive = IveFunction.apply
