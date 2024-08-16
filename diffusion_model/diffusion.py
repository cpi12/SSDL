

import torch
import numpy as np
from tqdm import tqdm
from .util import get_noise_schedule, get_keypoint_groups

class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        """
        Initialize the DiffusionProcess class.

        Args:
            timesteps (int): Number of timesteps.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
            schedule_type (str): Type of noise schedule ('linear', 'cosine', 'quadratic', 'sigmoid').
        """
        self.timesteps = timesteps
        self.betas = get_noise_schedule(schedule_type, timesteps, beta_start, beta_end)
        self.alphas = torch.tensor(1.0 - self.betas, dtype=torch.float32, device='cuda')  # Convert to PyTorch tensor
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Append 1 at the end of self.alphas_cumprod to match shapes
        self.alphas_cumprod = torch.cat([self.alphas_cumprod, torch.tensor([1.0], dtype=torch.float32, device='cuda')])
        
        epsilon = 1e-10
        self.posterior_variance = (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:] + epsilon) * torch.tensor(self.betas, dtype=torch.float32, device='cuda')

        # Anchor indices for joints (e.g., hips, shoulders)
        self.anchor_indices = get_keypoint_groups()["anchors"]


    def add_noise(self, x0, t, epoch=None, total_epochs=None):
        """
        Add noise to the input data for a dataset with 36 values (12 joints).

        Args:
            x0 (torch.Tensor): Original input data of shape [batch_size, 90, 36].
            t (torch.Tensor): Timesteps (should be a PyTorch tensor).
            epoch (int): Current training epoch.
            total_epochs (int): Total number of training epochs.

        Returns:
            torch.Tensor: Noisy data.
            torch.Tensor: Noise added.
        """
        # Random noise
        noise = torch.randn_like(x0)

        # Apply adaptive scaling based on joint type
        for i in range(0, 36, 3):  # Iterate over every 3 values (x, y, z for each joint)
            joint_type = i // 3
            if joint_type in self.anchor_indices:
                # Less noise for anchor joints
                noise[:, :, i:i+3] *= 0.1
            else:
                # More noise for joints that tend to move more
                noise[:, :, i:i+3] *= 1.5

        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        xt = alpha_t * x0 + one_minus_alpha_t * noise

        return xt, noise


    # def add_noise(self, x0, t):
    #     """
    #     Add noise to the input data while applying constraints to anchor joints.

    #     Args:
    #         x0 (torch.Tensor): Original input data of shape [batch_size, 90, 36].
    #         t (torch.Tensor): Timesteps (should be a PyTorch tensor).

    #     Returns:
    #         torch.Tensor: Noisy data with constrained anchor joints.
    #         torch.Tensor: Noise added.
    #     """
    #     noise = torch.randn_like(x0)
    #     alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
    #     one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    #     xt = alpha_t * x0 + one_minus_alpha_t * noise

    #     # Apply constraints to anchor joints
    #     for idx in self.anchor_indices:
    #         xt[:, :, idx*3:(idx+1)*3] = x0[:, :, idx*3:(idx+1)*3]  # Keep anchor joints stable

    #     return xt, noise

    def denoise(self, xt, context, t, device, model, noise=None, clipping=True):
        """
        Denoise the input data using the model.

        Args:
            xt (torch.Tensor): Noisy data.
            t (int): Timestep.
            model (torch.nn.Module): Model to use for denoising.
            noise (torch.Tensor, optional): Noise to add. Defaults to None.
            clipping (bool, optional): Whether to clip the output. Defaults to True.

        Returns:
            torch.Tensor: Denoised data.
        """
        if noise is None:
            noise = torch.randn_like(xt)

        # Ensure t is a 1-dimensional tensor (batch dimension)
        t_tensor = torch.tensor([t], device=device, dtype=torch.long).expand(xt.size(0))

        alpha_t = self.alphas_cumprod[t].to(device)
        beta_t = torch.tensor(self.betas[t], device=xt.device, dtype=xt.dtype)
        xt = xt.to(device)
        context = context.to(device)
        pred_noise = model(xt, context, t_tensor)

        x0_pred = (xt - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
        if clipping:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        xt_minus_1 = self.sqrt_recip_alphas[t] * xt - beta_t / self.sqrt_one_minus_alphas_cumprod[t] * pred_noise
        if t > 1:
            xt_minus_1 += self.posterior_variance[t] * noise

        return xt_minus_1

    @torch.no_grad()
    def sample(self, model, context, xt, steps, device, clipping=True):
        """
        Sample from the diffusion model.

        Args:
            model (torch.nn.Module): Model to use for sampling.
            xt (torch.Tensor): Initial noisy data.
            steps (int): Number of steps.
            clipping (bool, optional): Whether to clip the output. Defaults to True.

        Returns:
            torch.Tensor: Sampled data.
        """
        for step in tqdm(reversed(range(steps)), desc="Sampling"):
            xt = self.denoise(xt, context, step, device, model, clipping=clipping)
        return xt

    @torch.no_grad()
    def generate(self, model, context, shape, steps, device, clipping=True):
        """
        Generate samples using the diffusion model.

        Args:
            model (torch.nn.Module): Model to use for generation.
            shape (tuple): Shape of the generated data.
            steps (int): Number of steps.
            device (torch.device): Device to use for generation.
            clipping (bool, optional): Whether to clip the output. Defaults to True.

        Returns:
            torch.Tensor: Generated samples.
        """
        xt = torch.randn(shape, device=device)
        generated_samples = self.sample(model, context, xt, steps, device, clipping=clipping)
        return generated_samples
