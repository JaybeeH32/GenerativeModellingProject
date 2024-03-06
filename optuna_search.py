import numpy as np
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from tqdm import tqdm, trange  # to visualize progress

from PMCPnP import PMCPnPAnnealing, PMCPnP
from PMCPnP import PMCPnPIterator, PMCPnPAnnealingIterator
from utils import get_blur_physics, get_inpainting_physics

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("butterfly.png")
img_size = 256
x_true = load_url_image(url=url, img_size=img_size).to(device)
x = x_true.clone()

sigma = 0.2  # noise level

physics = get_blur_physics(sigma, device)

#physics = get_inpainting_physics(img_size, sigma, device)

torch.manual_seed(0)

# load Gaussian Likelihood

y = physics(x)

prior = dinv.optim.ScorePrior(
    denoiser=dinv.models.DRUNet(pretrained="download")
).to(device)

import optuna

def objective(trial):
    gamma = 5e-4
    alpha = 1
    iterations = 4000
    sigma_denoiser = np.sqrt(2 * gamma)
    
    gamma = trial.suggest_float("gamma", 1e-5, 1e-2, log=True)
    multiple = trial.suggest_float("sigma_denoiser", 0.8, 10, log=True)
    sigma_denoiser = np.sqrt(2 * gamma) * multiple

    likelihood = dinv.optim.L2(sigma=sigma_denoiser)

    pmc_pnp = PMCPnP(prior=prior,
                    data_fidelity=likelihood,
                    max_iter=iterations,
                    gamma=gamma,
                    sigma=sigma_denoiser,
                    alpha=alpha,)
    
    pula_mean, _ = pmc_pnp(y, physics)
    
    return dinv.utils.metric.cal_psnr(x, pula_mean)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

study.best_params 


