import numpy as np
from math import atan2, sqrt, log, cos, sin, exp, pi
from scipy.fft import fft


class PoissonBinomial:
    """ based on pip package 'poisson-binomial', with import of fft fixed """

    def __init__(self, prob_array):
        self.p = np.array(prob_array)
        self.pmf = self.get_poisson_binomial()
        self.cdf = np.cumsum(self.pmf)

    def x_or_less(self, x):
        return self.cdf[x]

    def x_or_more(self, x):
        return 1 - self.cdf[x] + self.pmf[x]

    def get_poisson_binomial(self):

        """This version of the poisson_binomial is implemented
        from the fast fourier transform method described in
        'On computing the distribution function for the
        Poisson binomial distribution'by Yili Hong 2013."""

        real = np.vectorize(lambda x: x.real)

        def x(w, l):
            v_atan2 = np.vectorize(atan2)
            v_sqrt = np.vectorize(sqrt)
            v_log = np.vectorize(log)

            if l == 0:
                return complex(1, 0)
            else:

                wl = w * l
                real = 1 + self.p * (cos(wl) - 1)
                imag = self.p * sin(wl)
                mod = v_sqrt(imag ** 2 + real ** 2)
                arg = v_atan2(imag, real)
                d = exp((v_log(mod)).sum())
                arg_sum = arg.sum()
                a = d * cos(arg_sum)
                b = d * sin(arg_sum)
                return complex(a, b)

        n = self.p.size
        w = 2 * pi / (1 + n)

        xs = [x(w, i) for i in range((n + 1) // 2 + 1)]
        for i in range((n + 1) // 2 + 1, n + 1):
            c = xs[n + 1 - i]
            xs.append(c.conjugate())

        return real(fft(xs)) / (n + 1)


if __name__ == '__main__':
    import random
    from performative_game.predicted_population.torch_poisson_binomial import PoissonBinomial as TorchPoissonBinomial

    fail = False
    for _ in range(100):
        n_trials = 50
        probs = [random.random() for _ in range(n_trials)]
        torch_poissonb = TorchPoissonBinomial(probs)
        np_poissonb = PoissonBinomial(probs)
        for i in range(n_trials):
            if abs(torch_poissonb.pmf[i].item() - np_poissonb.pmf[i]) > 1e-7:
                fail = True
                break
        if fail:
            break
    print(fail)
