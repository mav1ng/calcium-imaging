"""Training utilities for the neuron segmentation pipeline.

This module provides:

- **poly_lr**: Polynomial learning-rate decay schedule that smoothly anneals the
  learning rate from a base value to near zero over the course of training.
  Preferred over step-wise schedules because it avoids sudden jumps that can
  destabilize the contrastive embedding loss.

- **weight_init**: Comprehensive weight initialization dispatcher that applies
  layer-type-specific strategies (Xavier Normal for convolutions and linear
  layers, orthogonal for RNNs, etc.) via ``model.apply(weight_init)``.
  Proper initialization is critical for training deep encoder–decoder
  networks: Xavier Normal preserves gradient variance through many layers,
  preventing both vanishing and exploding gradients in the U-Net.

  Initialization strategies by layer type:
    - Conv2d / ConvTranspose2d: Xavier Normal (maintains variance under ReLU)
    - BatchNorm: weight ~ N(1, 0.02), bias = 0 (identity-like start)
    - Linear: Xavier Normal
    - LSTM / GRU: Orthogonal (preserves gradient norms in recurrent paths)

Typical usage::

    import training as t
    model = UNetMS(...)
    model.apply(t.weight_init)  # initialize all layers

    for epoch in range(nb_epochs):
        lr = t.poly_lr(epoch, nb_epochs, base_lr=0.002)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
"""

import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch import optim
import os

from src.utils import config as c
from src.data import data
from src.data import corr
from src.models import network as n
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.visualization import visualization as v

logger = logging.getLogger(__name__)


def poly_lr(
    iter: int,
    maxiter: int,
    base_lr: float,
    exp: float = 0.9,
) -> float:
    """Polynomial learning-rate decay schedule.

    Computes the learning rate at iteration *iter* using::

        lr = base_lr * (1 - iter / maxiter) ^ exp

    This provides a smooth, monotonically decreasing schedule that spends
    more iterations at lower learning rates, which is beneficial for
    fine-tuning embedding representations in the later stages of training.

    Args:
        iter: Current training iteration (0-indexed).
        maxiter: Total number of training iterations.
        base_lr: Initial (maximum) learning rate.
        exp: Polynomial exponent controlling decay curvature.
            Higher values produce faster initial decay.

    Returns:
        Learning rate for the current iteration.
    """
    return base_lr * np.power((1 - iter / maxiter), exp)


# def weights_init(m):
#     """Initialize Weights if the Model"""
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight.data, gain=1.0)
#         nn.init.xavier_uniform_(m.bias.data, gain=1.0)


def weight_init(m: nn.Module) -> None:
    """Layer-type-aware weight initialization dispatcher.

    Apply to a model via ``model.apply(weight_init)`` to initialize all
    parameters with strategies appropriate for each layer type.

    Initialization strategy rationale:
        - **Xavier Normal** (Conv2d, ConvTranspose2d, Linear): Preserves
          the variance of activations and gradients through the network,
          which is essential for training the deep U-Net encoder–decoder
          without vanishing/exploding gradients.
        - **Orthogonal** (LSTM, GRU): Maintains gradient norms across
          recurrent time steps, preventing long-range dependency loss.
        - **BatchNorm** weights ∼ N(1, 0.02), bias = 0: Starts as a
          near-identity transform so early training is stable.

    Reference:
        Adapted from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

    Args:
        m: A single PyTorch module (called per-layer by ``model.apply``).
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
