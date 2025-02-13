import torch
from math import cos, sin, pi


class PoissonBinomial:
    """ based on pip package 'poisson-binomial', with import of fft fixed """

    def __init__(self, prob_array):
        if isinstance(prob_array, torch.Tensor):
            pass
        elif isinstance(prob_array, list) or isinstance(prob_array, tuple):
            prob_array = torch.stack(prob_array)
        else:
            raise TypeError

        self.p = prob_array
        self.pmf = self.get_poisson_binomial()
        self.cdf = torch.cumsum(self.pmf, dim=0)

    def x_or_less(self, x):
        return self.cdf[x]

    def x_or_more(self, x):
        return 1 - self.cdf[x] + self.pmf[x]

    def get_poisson_binomial(self):
        """This version of the poisson_binomial is implemented
        from the fast fourier transform method described in
        'On computing the distribution function for the
        Poisson binomial distribution' by Yili Hong 2013."""

        def x(w, l):
            if l == 0:
                return torch.complex(torch.tensor(1.0), torch.tensor(0.0))
            else:
                wl = w * l
                real = 1 + self.p * (cos(wl) - 1)
                imag = self.p * sin(wl)
                mod = torch.sqrt(imag ** 2 + real ** 2)
                arg = torch.atan2(imag, real)
                d = torch.exp(torch.sum(torch.log(mod)))
                arg_sum = torch.sum(arg)
                a = d * torch.cos(arg_sum)
                b = d * torch.sin(arg_sum)
                return torch.complex(a, b)

        n = self.p.size(0)
        w = 2 * pi / (1 + n)

        xs = [x(w, i) for i in range((n + 1) // 2 + 1)]
        for i in range((n + 1) // 2 + 1, n + 1):
            c = xs[n + 1 - i]
            xs.append(c.conj())

        xs = torch.stack(xs)
        return torch.real(torch.fft.fft(xs)) / (n + 1)
