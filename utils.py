import deepinv as dinv
import numpy as np
import torch
import matplotlib.pyplot as plt

from Models import PMCPnP, PMCPnPAnnealing, PMCReD, PMCReDAnnealing

def get_image(data_directory, *, name='face1', img_size=256, device='cpu'):
    """ Possible images are 'face1', 'face2', 'face3', 'bedroom', 'butterfly' 
    """
    img = plt.imread(data_directory / f'{name}.png')
    return torch.tensor(img[:img_size, :img_size, :]).permute(2, 0, 1).unsqueeze(0).float().to(device)


def get_blur_physics(img_size, sigma, device, random_filter=False):
    if random_filter:
        kernel = torch.randn(1, 1, img_size, img_size)
    else:
        kernel = torch.tensor(np.loadtxt("kernel8.txt"), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    physics = dinv.physics.Blur(
        filter=kernel,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )
    return physics

def get_inpainting_physics(img_size, sigma, device, mask_random=True):
    if mask_random:
        mask = 0.8
    else:
        hole_size = 60
        mask = torch.ones(img_size, img_size, device=device)
        mask[(img_size - hole_size)//2:(img_size + hole_size)//2, (img_size - hole_size)//2:(img_size + hole_size)//2] = torch.zeros(hole_size, hole_size)
        
    physics = dinv.physics.Inpainting(
        mask=mask,
        tensor_size=(3, img_size, img_size),
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )
    return physics

def get_downsampling_physics(img_size, sigma, device, factor=2, filter=None):
    """ Possible filters are None, 'gaussian', 'bilinear' and 'bicubic'
    """
    physics = dinv.physics.Downsampling(
        img_size=(3, img_size, img_size),
        factor=factor,
        filter=filter,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )
    return physics

def get_denoising_physics(img_size, sigma, device):
    """ Simple identity forward operator.
    """
    physics = dinv.physics.Denoising(
        img_size=(3, img_size, img_size),
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
        device=device,
    )
    return physics

def get_compressed_sensing_physics(img_size, sigma, device, factor=10):
    """ Outputs a (1, 3 * img_size * img_size // factor // factor) tensor.
    """
    physics = dinv.physics.CompressedSensing(
        m=3 * img_size * img_size // factor // factor,
        img_shape=(3, img_size, img_size),
        device=device,
        fast=True,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )
    return physics

def get_physics(transformation_name, img_size, sigma, device='cpu', factor=2, filter=None):
    if transformation_name == 'downsampling':
        return get_downsampling_physics(img_size, sigma, device, factor, filter)
    elif transformation_name == 'denoising':
        return get_denoising_physics(img_size, sigma, device)
    elif transformation_name == 'compressed_sensing':
        return get_compressed_sensing_physics(img_size, sigma, device, factor)
    elif transformation_name == 'blur':
        return get_blur_physics(img_size, sigma, device)
    elif transformation_name == 'inpainting':
        return get_inpainting_physics(img_size, sigma, device)
    else:
        raise ValueError("Invalid transformation_name")
    
    
def get_model(model_name, denoiser, likelihood, max_iter=100, stepsize=1e-3, lambd=0.01,  sigma=0.05, device='cpu'):
    if model_name == 'PnP':
        model = dinv.optim.optim_builder(
            iteration="PGD",
            prior=dinv.optim.prior.PnP(denoiser=denoiser),
            data_fidelity=likelihood,
            early_stop=True,
            max_iter=max_iter,
            verbose=True,
            params_algo={"stepsize": stepsize, "g_param": sigma, "lambda": lambd},
        )
    elif model_name == 'RED':
        model = dinv.optim.optim_builder(
            iteration="PGD",
            prior=dinv.optim.prior.RED(denoiser=denoiser),
            data_fidelity=likelihood,
            early_stop=True,
            max_iter=max_iter,
            verbose=True,
            params_algo={"stepsize": stepsize, "g_param": sigma, "lambda": lambd},
        )
    elif model_name == 'PnP-ULA':
        model = dinv.sampling.ULA(
            prior= dinv.optim.ScorePrior(denoiser=denoiser).to(device),
            data_fidelity=likelihood,
            max_iter=max_iter,
            alpha=1,
            step_size=stepsize,
            verbose=True,
            sigma=sigma,
        )
    elif model_name == 'DPS':
        model = dinv.sampling.DPS(
            denoiser,
            data_fidelity=likelihood,
            max_iter=max_iter,
            verbose=True,
            device=device,
            save_iterates=False,
        )
    
    elif model_name == 'PMCPnP':
        model = PMCPnP(
            prior=dinv.optim.ScorePrior(denoiser=denoiser).to(device),
            data_fidelity=likelihood,
            max_iter=max_iter,
            verbose=True,
            alpha=1,
            sigma=sigma,
            gamma=stepsize,
        )
    
    elif model_name == 'PMCReD':
        model = PMCReD(
            prior=dinv.optim.ScorePrior(denoiser=denoiser).to(device),
            data_fidelity=likelihood,
            max_iter=max_iter,
            verbose=True,
            alpha=1,
            sigma=sigma,
            gamma=stepsize,
        )
    
    elif model_name == 'PMCPnPAnnealing':
        model = PMCPnPAnnealing(
            prior=dinv.optim.ScorePrior(denoiser=denoiser).to(device),
            data_fidelity=likelihood,
            max_iter=max_iter,
            verbose=True,
            alphas=exponential_decay_alpha(10, 1, max_iter),
            sigmas=exponential_decay_sigma(10, sigma, max_iter),
            gamma=stepsize,
        )
    
    elif model_name == 'PMCReDAnnealing':
        model = PMCReDAnnealing(
            prior=dinv.optim.ScorePrior(denoiser=denoiser).to(device),
            data_fidelity=likelihood,
            max_iter=max_iter,
            verbose=True,
            alphas=exponential_decay_alpha(10, 1, max_iter),
            sigmas=exponential_decay_sigma(10, sigma, max_iter),
            gamma=stepsize,
        )
        
    else:
        raise ValueError(f"Model {model_name} not implemented")
    return model

def exponential_decay_alpha(alpha_0, alpha_final, iterations):
    # decay_rate = 1 / alpha_0
    decay_rate = .975
    t_values = torch.arange(0, iterations, dtype=torch.float32)
    alphas = alpha_0 * (decay_rate ** t_values)
    alphas[alphas < alpha_final] = alpha_final
    return alphas

def exponential_decay_sigma(sigma_0, sigma_final, iterations):
    decay_rate = .975
    t_values = torch.arange(0, iterations, dtype=torch.float32)
    sigmas = sigma_0 * (decay_rate ** t_values)
    sigmas[sigmas < sigma_final] = sigma_final
    return sigmas
    
