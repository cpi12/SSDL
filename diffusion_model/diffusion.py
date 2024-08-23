import torch
import numpy as np
from tqdm import tqdm

class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.1, schedule_type='linear'):
        """
        Initialize the DiffusionProcess class with basic settings.

        Args:
            timesteps (int): Number of timesteps.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
            schedule_type (str): Type of noise schedule ('linear', 'cosine', etc.).
        """
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device='cuda')
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute values used in the diffusion process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, t):
        """
        Add noise to the clean input data.

        Args:
            x0 (torch.Tensor): Original input data.
            t (torch.Tensor): Timesteps.

        Returns:
            torch.Tensor: Noisy data.
            torch.Tensor: Noise added.
        """
        noise = torch.randn_like(x0)
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        xt = alpha_t * x0 + one_minus_alpha_t * noise

        return xt, noise

    def denoise(self, xt, context, t, model):
        """
        Denoise the input data using the model.

        Args:
            xt (torch.Tensor): Noisy data.
            context (torch.Tensor): Context or conditioning input.
            t (torch.Tensor): Timestep.
            model (torch.nn.Module): Model used for denoising.

        Returns:
            torch.Tensor: Denoised data.
        """
        pred_noise = model(xt, context, t)
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x0_pred = (xt - one_minus_alpha_t * pred_noise) / alpha_t

        return x0_pred

    @torch.no_grad()
    def sample(self, model, context, xt, steps, device):
        """
        Sample data using the diffusion model.

        Args:
            model (torch.nn.Module): Model used for sampling.
            context (torch.Tensor): Context or conditioning input.
            xt (torch.Tensor): Initial noisy data.
            steps (int): Number of steps.
            device (torch.device): Device to use for sampling.

        Returns:
            torch.Tensor: Sampled data.
        """
        for step in tqdm(reversed(range(steps)), desc="Sampling"):
            t = torch.tensor([step], device=device, dtype=torch.long).expand(xt.size(0))
            xt = self.denoise(xt, context, t, model)
        return xt

    @torch.no_grad()
    def generate(self, model, context, shape, steps, device):
        """
        Generate samples using the diffusion model.

        Args:
            model (torch.nn.Module): Model used for generation.
            context (torch.Tensor): Context or conditioning input.
            shape (tuple): Shape of the generated data.
            steps (int): Number of steps.
            device (torch.device): Device to use for generation.

        Returns:
            torch.Tensor: Generated samples.
        """
        xt = torch.randn(shape, device=device)
        generated_samples = self.sample(model, context, xt, steps, device)
        return generated_samples
