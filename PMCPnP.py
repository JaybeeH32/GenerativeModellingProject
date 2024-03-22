import deepinv as dinv
import numpy as np
import torch

class PMCPnPIterator(torch.nn.Module):
    def __init__(self, gamma, sigma, alpha=1, deterministic = False, Himbert = False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.noise_std = np.sqrt(2 * gamma)
        self.sigma = sigma
        self.iter = 0
        self.deterministic = deterministic
        self.Himbert = Himbert

    def forward(self, x, y, physics, likelihood, prior):
        self.iter += 1
        if not self.Himbert:
            lhood = - likelihood.grad(x, y, physics)
            lprior = - prior(x + (self.gamma * lhood), self.sigma) * self.alpha
        else:
            lhood = - likelihood.grad(x, y, physics)
            lprior = - prior(x - (self.gamma * lhood), self.sigma) * self.alpha           
        if self.deterministic:
            return x + self.gamma * (lhood + lprior) 
        else:
            noise = torch.randn_like(x) * self.noise_std
            return x + self.gamma * (lhood + lprior) + noise 

            
        # noise = torch.randn_like(x) * self.noise_std
        # # grad_likelihood = likelihood.grad(x, y, physics)
        # # P = grad_likelihood - prior((x - self.gamma * grad_ikelihood), self.sigma)
        # # return x - self.gamma * P + noise
        # lhood = - likelihood.grad(x, y, physics)
        # lprior = - prior(x + (self.gamma * lhood), self.sigma) * self.alpha
        # return x + self.gamma * (lhood + lprior) + noise
        # # lhood = - likelihood.grad(x, y, physics)
        # # lprior = - prior(x + (self.gamma * lhood), self.sigma) * self.alpha
        # # return x + self.gamma * (lhood + lprior) + noise
        
class PMCPnP(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigma, gamma, alpha, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(0, 1), verbose=True, deterministic = False, Himbert = False,
    ):
        # generate an iterator
        iterator = PMCPnPIterator(gamma=gamma, sigma=sigma, alpha=alpha, deterministic = deterministic, Himbert = Himbert)
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
        