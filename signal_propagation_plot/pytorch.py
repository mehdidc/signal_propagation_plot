"""
Signal propagation plots (SPP) for PyTorch models.
"""
from functools import partial
import torch
import torch.nn as nn
import numpy as np

def plot(name_values, *args, **kwargs):
    import matplotlib.pyplot as plt
    labels = [name for name, value in name_values]
    values  = [value for name, value in name_values]
    depth = np.arange(len(labels))
    plt.plot(depth, values, *args, **kwargs)
    plt.xticks(depth, labels, rotation="vertical")

def get_average_channel_squared_mean_by_depth(model,  *args, **kwargs):
    acts = extract_activations(model, *args, **kwargs)
    values = []
    for name, tensor in acts:
        values.append((name, average_channel_squared_mean(tensor)))
    return values

def get_average_channel_variance_by_depth(model,  *args, **kwargs):
    acts = extract_activations(model, *args, **kwargs)
    values = []
    for name, tensor in acts:
        values.append((name, average_channel_variance(tensor)))
    return values


def average_channel_squared_mean(x):
    if x.ndim == 4:
        return (x.mean(dim=(0,2,3))**2).mean().item()
    elif x.ndim == 2:
        return (x**2).mean().item()
    else:
        raise ValueError(f"not supported shape: {x.shape}")

def average_channel_variance(x):
    if x.ndim == 4:
        return x.var(dim=(0,2,3)).mean().item()
    elif x.ndim == 2:
        return x.var(dim=0).mean().item()
    else:
        raise ValueError(f"not supported shape: {x.shape}")

def extract_activations(model, *args, **kwargs):
    acts = []
    handles = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(partial(hook, name=name, store=acts))
        handles.append(handle)
    model(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return acts

def hook(self, input, output, store=None, name=None):
    if store is None:
        store = []
    store.append((name, output))

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    model = torchvision.models.resnet101()
    x = torch.randn(64,3,224,224)
    name_values = get_average_channel_squared_mean_by_depth(model, x)
    fig = plt.figure(figsize=(15, 10))
    plot(name_values)
    plt.show()
