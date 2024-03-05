import deepinv as dinv
import numpy as np
import torch

class PMCPnPIterator(torch.nn.Module):
    def __init__(self, gamma, sigma, alpha=1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.noise_std = np.sqrt(2 * gamma)
        self.sigma = sigma
        self.iter = 0

    def forward(self, x, y, physics, likelihood, prior):
        self.iter += 1
        noise = torch.randn_like(x) * self.noise_std
        lhood = - likelihood.grad(x, y, physics)
        lprior = - prior(x - (self.gamma * lhood), self.sigma) * self.alpha
        return x + self.gamma * (lhood + lprior) + noise
        
class PMCPnP(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigma, gamma, alpha, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(0, 1), verbose=True,
    ):
        # generate an iterator
        iterator = PMCPnPIterator(gamma=gamma, sigma=sigma, alpha=alpha)
        # set the params of the base class
        super().__init__(
            iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, burnin_ratio=burnin_ratio, clip=clip, verbose=verbose,
        )
        
        
class PMCPnPAnnealingIterator(torch.nn.Module):
    def __init__(self, gamma, sigmas, alphas=torch.ones(1000), max_iter=1e3):
        super().__init__()
        self.gamma = gamma
        self.alphas = alphas
        self.noise_std = np.sqrt(2 * gamma)
        self.sigmas = sigmas
        self.iter = 0
        self.max_iter = max_iter

    def forward(self, x, y, physics, likelihood, prior):
        noise = torch.randn_like(x) * self.noise_std
        lhood = - likelihood.grad(x, y, physics)
        lprior = - prior(x - (self.gamma * lhood), self.sigmas[self.iter]) * self.alphas[self.iter]
        self.iter = (self.iter + 1) % self.max_iter
        return x + self.gamma * (lhood + lprior) + noise
        
class PMCPnPAnnealing(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigmas, gamma, alphas, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(-1, 2), verbose=True,
    ):
        # generate an iterator
        iterator = PMCPnPAnnealingIterator(gamma=gamma, sigmas=sigmas, alphas=alphas, max_iter=max_iter)
        # set the params of the base class
        super().__init__(
            iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, burnin_ratio=burnin_ratio, clip=clip, verbose=verbose,
        )
        