import numpy as np
import torch
def inpaint(sde, data, mask, input_channels, input_height, num_steps=20, transform_forward=None, 
             transform_reverse=None, mean=0, std=1, clip=True, epsilon=1e-3, log_epoch=None):
    """
    The mask represents the known positions of the sample.
    """
    delta = sde.T / num_steps
    noise = torch.randn(input_channels, input_height, input_height).to(sde.T)
    noise = noise * std + mean
    data = data.to(sde.T).view(-1, input_channels, input_height, input_height)
    mask = mask.to(sde.T).view(-1, 1, input_height, input_height)

    data = data * 255 / 256 + torch.rand_like(data) / 256
    if transform_forward is not None:
        data_transformed = transform_forward.forward_transform(data)
    else:
        data_transformed = data
    y0 = noise * (1 - mask) + data_transformed * mask
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(y0.shape[0], 1, 1, 1).to(y0)

    results = []

    def output(y):
        t = 0
        if transform_reverse is not None:
            t = transform_reverse(y)
        if clip:
            t = torch.clip(t, 0, 1)
        return t.data.cpu().numpy()

    if log_epoch is not None:
        results.append(output(y0))

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + (delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0))
            data_perturbed = sde.base_sde.sample(sde.T - ones * ts[i], data_transformed)
            y0 = y0 * (1. - mask) + data_perturbed * mask

            score = sde.a(y0, (ones * ts[i]).squeeze())
            g = sde.base_sde.g(sde.T - ones * ts[i], y0)
            y0 = y0 + (epsilon * score / g + (2 * epsilon) ** 0.5 * torch.randn_like(y0)) # Corrector
            data_perturbed = sde.base_sde.sample(sde.T - ones * ts[i], data_transformed)
            y0 = y0 * (1. - mask) + data_perturbed * mask
            if log_epoch is not None and (i+1) % log_epoch == 0:
                results.append(output(y0))
            if torch.isnan(y0).any():
                break
    y0 = output(y0)
    if log_epoch is not None:
        return y0, np.stack(results, axis=0)
    return y0