import deepinv as dinv
import numpy as np
import torch

########################
### Iterator classes ###
########################

class Iterator(torch.nn.Module):
    def __init__(self, gamma, sigma, alpha=1, deterministic = False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.noise_std = np.sqrt(2 * gamma)
        self.sigma = sigma
        self.iter = -1
        self.deterministic = deterministic

    def forward(self, x, y, physics, likelihood, prior):
        self.iter += 1
        lhood = likelihood.grad(x, y, physics)
        lprior = self.step(x, prior, lhood, self.alpha, self.sigma)
        # lprior = prior(x - (self.gamma * lhood), self.sigma) * self.alpha
          
        if self.deterministic:
            return x - self.gamma * (lhood + lprior) 
        else:
            noise = torch.randn_like(x) * self.noise_std
            return x - self.gamma * (lhood + lprior) + noise 

    def step(self, x, prior, lhood, alpha, sigma):
        pass

class AnnealingIterator(torch.nn.Module):
    def __init__(self, gamma, sigmas, alphas=torch.ones(1000), max_iter=1e7, deterministic = False):
        super().__init__()
        self.gamma = gamma
        self.alphas = alphas
        self.noise_std = np.sqrt(2 * gamma)
        self.sigmas = sigmas
        self.iter = -1
        self.max_iter = max_iter
        self.deterministic = deterministic

    def forward(self, x, y, physics, likelihood, prior):
        self.iter += 1
        lhood = likelihood.grad(x, y, physics)
        lprior = self.step(x, prior, lhood, self.alphas[self.iter], self.sigmas[self.iter])          
        if self.deterministic:
            return x - self.gamma * (lhood + lprior) 
        else:
            noise = torch.randn_like(x) * self.noise_std
            return x - self.gamma * (lhood + lprior) + noise 

    def step(self, x, prior, lhood, alpha, sigma):
        pass

###########################
### PnP and Red Classes ###
###########################

class PMCPnPIterator(Iterator):

    def __init__(self, gamma, sigma, alpha=1, deterministic = False):
        super().__init__(gamma, sigma, alpha, deterministic)

    def step(self, x, prior, lhood, alpha, sigma):
        return prior(x - (self.gamma * lhood), sigma) * alpha

class PMCReDIterator(Iterator):

    def __init__(self, gamma, sigma, alpha=1, deterministic = False):
        super().__init__(gamma, sigma, alpha, deterministic)

    def step(self, x, prior, lhood, alpha, sigma):
        return prior(x, sigma) * alpha

class PMCPnPAnnealingIterator(AnnealingIterator):

    def __init__(self, gamma, sigmas, alphas=torch.ones(1000), max_iter=1e7, deterministic = False):
        super().__init__(gamma, sigmas, alphas, deterministic)

    def step(self, x, prior, lhood, alpha, sigma):
        return prior(x - (self.gamma * lhood), sigma) * alpha

class PMCReDAnnealingIterator(AnnealingIterator):

    def __init__(self, gamma, sigmas, alphas=torch.ones(1000), max_iter=1e7, deterministic = False):
        super().__init__(gamma, sigmas, alphas, deterministic)

    def step(self, x, prior, lhood, alpha, sigma):
        return prior(x, sigma) * alpha

###################
### PMC classes ###
###################

class PMCPnP(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigma, gamma, alpha, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(0, 1), verbose=True, deterministic = False,
    ):
        # generate an iterator
        iterator = PMCPnPIterator(gamma=gamma, sigma=sigma, alpha=alpha, deterministic = deterministic)
        # set the params of the base class
        super().__init__(
            iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, burnin_ratio=burnin_ratio, clip=clip, verbose=verbose,
        )

class PMCReD(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigma, gamma, alpha, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(0, 1), verbose=True, deterministic = False,
    ):
        # generate an iterator
        iterator = PMCReDIterator(gamma=gamma, sigma=sigma, alpha=alpha, deterministic = deterministic)
        # set the params of the base class
        super().__init__(
            iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, burnin_ratio=burnin_ratio, clip=clip, verbose=verbose,
        )

class PMCPnPAnnealing(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigmas, gamma, alphas, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(-1, 2), verbose=True, deterministic = False,
    ):
        # generate an iterator
        iterator = PMCPnPAnnealingIterator(gamma=gamma, sigmas=sigmas, alphas=alphas, max_iter=max_iter,  deterministic = deterministic)
        # set the params of the base class
        super().__init__(
            iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, burnin_ratio=burnin_ratio, clip=clip, verbose=verbose,
        )
        

class PMCReDAnnealing(dinv.sampling.MonteCarlo):
    def __init__(
        self, prior, data_fidelity, sigmas, gamma, alphas, max_iter=1e3, thinning=1, burnin_ratio=0.4, clip=(-1, 2), verbose=True, deterministic = False,
    ):
        # generate an iterator
        iterator = PMCReDAnnealingIterator(gamma=gamma, sigmas=sigmas, alphas=alphas, max_iter=max_iter,  deterministic = deterministic)
        # set the params of the base class
        super().__init__(
            iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, burnin_ratio=burnin_ratio, clip=clip, verbose=verbose,
        )
        