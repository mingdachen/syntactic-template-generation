import math
import torch
import warnings

import numpy as np

from ive import ive
from scipy import special as sp


class HypersphericalUniform(torch.nn.Module):
    @property
    def dim(self):
        return self._dim

    def __init__(self, dim):
        super(HypersphericalUniform, self).__init__()
        self._dim = dim

    def entropy(self):
        return self.__log_surface_area()

    def __log_surface_area(self):
        output = math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - \
            sp.loggamma((self._dim + 1) / 2).real
        return output


class VonMisesFisher(torch.nn.Module):

    @property
    def mean(self):
        return self.loc * (ive(self.__m / 2, self.scale) /
                           ive(self.__m / 2 - 1, self.scale))

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale):
        super(VonMisesFisher, self).__init__()
        self.loc = loc
        self.scale = scale
        self.__m = int(loc.size()[-1])
        self.__e1 = torch.tensor([1.] + [0] * (loc.size()[-1] - 1),
                                 device=loc.device)

    def rsample(self, shape=[]):

        w = self.__sample_w3(shape=shape) \
            if self.__m == 3 else self.__sample_w_rej(shape=shape)

        v = np.random.normal(0, 1, size=shape + list(self.loc.size()))
        v = np.swapaxes(v, 0, -1)[1:]
        v = np.swapaxes(v, 0, -1)

        v = torch.from_numpy(v).float().to(self.loc.device)

        v = v / v.norm(dim=-1, keepdim=True)

        x = torch.cat((w, torch.sqrt(1 - (w ** 2)) * v), -1)
        z = self.__householder_rotation(x)

        return z

    def __sample_w3(self, shape):
        shape = shape + list(self.scale.size())
        u = torch.from_numpy(
            np.random.uniform(low=0, high=1, size=shape)).float()\
            .to(self.loc.device)

        self.__w = torch.stack(
            [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0)
        self.__w = VonMisesFisher.logsumexp(self.__w, dim=0)
        self.__w = 1 + self.__w / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(torch.max(
            torch.zeros_like(self.scale), self.scale - 10)[0],
            torch.ones_like(self.scale))[0]
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        return self.__w

    def __while_loop(self, b, a, d, shape):

        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.size()))) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b), \
            torch.zeros_like(b), (torch.ones_like(b) == 1)

        shape = shape + list(self.scale.size())
        max_try = 50 * np.array(shape).prod()
        n_try = 0  # will give up after given patiance
        while bool_mask.sum().item() != 0:
            if n_try > max_try:
                warnings.warn("Maximum iterations for rejection sampling exceeded!")
                break
            e_ = torch.from_numpy(
                np.random.beta((self.__m - 1) / 2,
                               (self.__m - 1) / 2,
                               size=shape[:-1]).reshape(shape)).float()\
                .to(self.loc.device)
            u = torch.from_numpy(np.random.uniform(0, 1, size=shape)).float()\
                .to(self.loc.device)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * t.log() - t + d) > torch.log(u)
            reject = (1 - accept.long()).byte()

            accept_mask = (bool_mask * accept).detach()
            if accept_mask.sum().item():
                w[accept_mask] = w_[accept_mask]
                e[accept_mask] = e_[accept_mask]

                bool_mask[accept_mask] = reject[accept_mask]
            n_try += 1

        return e, w

    def __householder_rotation(self, x):
        u = (self.__e1 - self.loc)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        output = - self.scale.double() * \
            ive(self.__m / 2, self.scale) / \
            ive((self.__m / 2) - 1, self.scale)

        return output.view(*(output.size()[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.size()[:-1]))

    def _log_normalization(self):
        output = - ((self.__m / 2 - 1) * torch.log(self.scale.double()) -
                    (self.__m / 2) * math.log(2 * math.pi) -
                    (self.scale.double() + torch.log(
                        ive(self.__m / 2 - 1, self.scale))))

        return output.view(*(output.size()[:-1]))

    @staticmethod
    def logsumexp(inputs, dim=None, keepdim=False):
        """Numerically stable logsumexp.

        Args:
            inputs: A Variable with any shape.
            dim: An integer.
            keepdim: A boolean.

        Returns:
            Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
        """
        # For a 1-D array x (any array along a single dimension),
        # log sum exp(x) = s + log sum exp(x - s)
        # with s = max(x) being a common choice.
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = torch.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def kl_div(self):
        # output = self.scale.double() * \
        #     ive(self.__m / 2, self.scale) / \
        #     ive((self.__m / 2) - 1, self.scale) + \
        #     ((self.__m / 2 - 1) * torch.log(self.scale.double()) -
        #      (self.__m / 2) * math.log(2 * math.pi) -
        #      torch.log(ive(self.__m / 2 - 1, self.scale))) + \
        #     self.__m / 2 * math.log(math.pi) + math.log(2) - \
        #     sp.loggamma(self.__m / 2).real
        # return output.float()
        return - self.entropy().float() + \
            HypersphericalUniform(self.__m - 1).entropy()
