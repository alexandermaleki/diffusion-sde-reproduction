import math

import torch


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_model.eval()
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1.0 - decay)


def vp_beta_t(t, beta_min=0.1, beta_max=20.0):
    return beta_min + t * (beta_max - beta_min)


def vp_perturbation_kernel(t, x0, beta_min=0.1, beta_max=20.0):
    beta_integral = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    mean_coeff = torch.exp(-0.5 * beta_integral)
    mean = mean_coeff[:, None, None, None] * x0
    std = torch.sqrt(1.0 - torch.exp(-beta_integral))
    return mean, std


def ve_sigma_t(t, sigma_min=0.01, sigma_max=50.0):
    return sigma_min * (sigma_max / sigma_min) ** t


def ve_perturbation_kernel(t, x0, sigma_min=0.01, sigma_max=50.0):
    sigma = ve_sigma_t(t, sigma_min=sigma_min, sigma_max=sigma_max)
    return x0, sigma[:, None, None, None]


def ve_g_squared_t(t, sigma_min=0.01, sigma_max=50.0):
    sigma = ve_sigma_t(t, sigma_min=sigma_min, sigma_max=sigma_max)
    return 2.0 * math.log(sigma_max / sigma_min) * sigma**2

