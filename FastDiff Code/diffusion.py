import torch
import torch.nn as nn
import numpy as np
from config import cfg

class Diffusion:
    def __init__(self, config=cfg):
        self.n_T = config.TIMESTEPS
        self.device = config.DEVICE
        self.drop_prob = config.DROP_PROB
        
        # Pre-compute schedules
        self.beta = torch.linspace(config.BETA_START, config.BETA_END, self.n_T).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
        
        self.oneover_sqrta = 1 / torch.sqrt(self.alpha)
        self.sqrt_beta_t = torch.sqrt(self.beta)
        self.mab_over_sqrtmab = (1 - self.alpha) / torch.sqrt(1 - self.alpha_hat)

    def noise_images(self, x, t):
        """
        Adds noise to images x at timestep t.
        Returns: (Noisy Image, Noise)
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def train_step(self, model, x, c, loss_fn):
        model.train()
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        
        x_t = (
            self.sqrt_alpha_hat[_ts-1][:, None, None, None] * x
            + self.sqrt_one_minus_alpha_hat[_ts-1][:, None, None, None] * noise
        )

        context_mask = torch.bernoulli(torch.zeros_like(c).float() + self.drop_prob).to(self.device)
        predicted_noise = model(x_t, c, _ts / self.n_T, context_mask)
        return loss_fn(noise, predicted_noise)

    def sample_trajectory(self, model, n_sample=1, labels=None, guide_w=0.0, capture_steps=[]):
        """
        Generates images and captures snapshots for visualization.
        """
        model.eval()
        size = (cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        x_i = torch.randn(n_sample, *size).to(self.device)
        
        # If labels are provided, use them. Otherwise, use 0.
        if labels is not None:
            c_i = labels.to(self.device)
        else:
            c_i = torch.zeros(n_sample).long().to(self.device)
        
        c_i = c_i.repeat(2)
        context_mask = torch.zeros_like(c_i).float().to(self.device)
        context_mask[n_sample:] = 1.0
        
        trajectory = {}
        
        # Capture T (Pure Noise)
        if self.n_T in capture_steps:
            trajectory[self.n_T] = x_i.clone()

        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample * 2, 1, 1, 1)
                z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0
                
                x_input = x_i.repeat(2, 1, 1, 1)
                eps = model(x_input, c_i, t_is, context_mask)
                eps = (1 + guide_w) * eps[:n_sample] - guide_w * eps[n_sample:]
                x_i = self.oneover_sqrta[i-1] * (x_i - eps * self.mab_over_sqrtmab[i-1]) + self.sqrt_beta_t[i-1] * z
                
                current_step = i - 1
                if current_step in capture_steps:
                    trajectory[current_step] = x_i.clone()
                    
        return trajectory