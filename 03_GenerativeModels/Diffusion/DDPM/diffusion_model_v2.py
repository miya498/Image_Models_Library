import numpy as np
import torch
from diffusion_utils import GaussianDiffusion, get_beta_schedule

# Diffusion Model Configuration
def create_diffusion_model(num_timesteps=1000):
    betas = get_beta_schedule(
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=num_timesteps,
    )

    return GaussianDiffusion(
        betas=betas,
        model_mean_type="eps",  # Noise prediction
        model_var_type="fixedsmall",  # Fixed variance
        loss_type="mse",  # Mean Squared Error Loss
    )