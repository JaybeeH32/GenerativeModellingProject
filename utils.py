import deepinv as dinv
import numpy as np
import torch



def get_blur_physics(sigma, device, random_filter=False):
    if random_filter:
        kernel = torch.randn(1, 1, 256, 256)
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
        mask = 0.5
    else:
        hole_size = 60
        mask = torch.ones(img_size, img_size)
        mask[(img_size - hole_size)//2:(img_size + hole_size)//2, (img_size - hole_size)//2:(img_size + hole_size)//2] = torch.zeros(hole_size, hole_size)
        mask.to(device)
        
    physics = dinv.physics.Inpainting(
        mask=mask,
        tensor_size=(3, img_size, img_size),
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )
    return physics
    
