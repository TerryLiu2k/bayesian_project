from tkinter.messagebox import NO
import torch
import numpy as np


def get_grid(sde, input_channels, input_height, n=4, num_steps=20, transform=None, 
             mean=0, std=1, clip=True, log_epoch=None):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    results = []

    def output(y):
        t = y.view(
            n, n, input_channels, input_height, input_height).permute(
            2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)
        if transform is not None:
            t = transform(t)
        if clip:
            t = torch.clip(t, 0, 1)
        return t.data.cpu().numpy()

    if log_epoch is not None:
        results.append(output(y0))

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

            if log_epoch is not None and (i+1) % log_epoch == 0:
                results.append(output(y0))
    
    if log_epoch is not None:
        return output(y0), np.stack(results, axis=0)
    return output(y0)

def get_grid_improved(sde, input_channels, input_height, n=4, num_steps=20, transform=None, 
             mean=0, std=1, clip=True, log_epoch=None):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    results = []

    def output(y):
        t = y.view(
            n, n, input_channels, input_height, input_height).permute(
            2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)
        if transform is not None:
            t = transform(t)
        if clip:
            t = torch.clip(t, 0, 1)
        return t.data.cpu().numpy()

    if log_epoch is not None:
        results.append(output(y0))

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            S = float(np.random.random() < 0.5)
            K1 = delta * mu + delta ** 0.5 * (torch.randn_like(y0) + S) * sigma
            y1 = y0 + K1
            mu1 = sde.mu(ones * ts[i+1], y1)
            sigma1 = sde.sigma(ones * ts[i+1], y1)
            K2 = delta * mu1 + delta ** 0.5 * (torch.randn_like(y0) - S) * sigma1
            y0 = y0 + (K1 + K2) / 2

            if log_epoch is not None and (i+1) % log_epoch == 0:
                results.append(output(y0))
    
    if log_epoch is not None:
        return output(y0), np.stack(results, axis=0)
    return output(y0)

def get_grid_corrector(sde, input_channels, input_height, K=1, n=4, num_steps=20, transform=None, 
             mean=0, std=1, clip=True, epsilon=1e-5, log_epoch=None):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    results = []

    def output(y):
        t = y.view(
            n, n, input_channels, input_height, input_height).permute(
            2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)
        if transform is not None:
            t = transform(t)
        if clip:
            t = torch.clip(t, 0, 1)
        return t.data.cpu().numpy()

    if log_epoch is not None:
        results.append(output(y0))


    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

            for k in range(K):
                score = sde.a(y0, sde.T - (ones * ts[i]).squeeze())
                g = sde.base_sde.g(sde.T - ones * ts[i], y0)
                y0 = y0 + epsilon * score / g + (2 * epsilon) ** 0.5 * torch.randn_like(y0) # Corrector

            if log_epoch is not None and (i+1) % log_epoch == 0:
                results.append(output(y0))
    
    if log_epoch is not None:
        return output(y0), np.stack(results, axis=0)
    return output(y0)


def get_grid_ode(sde, input_channels, input_height, n=4, num_steps=20, transform=None, 
             mean=0, std=1, clip=True, log_epoch=None):
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)

    results = []

    def output(y):
        t = y.view(
            n, n, input_channels, input_height, input_height).permute(
            2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)
        if transform is not None:
            t = transform(t)
        if clip:
            t = torch.clip(t, 0, 1)
        return t.data.cpu().numpy()

    if log_epoch is not None:
        results.append(output(y0))


    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            y0 = y0 + delta * mu

            if log_epoch is not None and (i+1) % log_epoch == 0:
                results.append(output(y0))
    
    if log_epoch is not None:
        return output(y0), np.stack(results, axis=0)
    return output(y0)
