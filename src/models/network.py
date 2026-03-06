"""Neural network architectures for neuron segmentation in calcium imaging data.

This module implements the core deep learning components of the segmentation pipeline:

- **UNet**: A 4-level encoder–decoder with skip connections that maps multi-channel
  correlation images to dense pixel-wise embeddings.  Architecture follows
  Ronneberger et al. (2015) with BatchNorm and progressive dropout.
- **MS (Mean-Shift)**: A differentiable mean-shift module that iteratively refines
  pixel embeddings by shifting them toward local density modes in the learned
  embedding space.  Kernel bandwidth is derived from the contrastive margin.
- **UNetMS**: End-to-end wrapper combining U-Net feature extraction, L2
  normalization onto the unit hypersphere, and mean-shift refinement.
- **EmbeddingLoss**: Contrastive loss with inverse-frequency weighting that
  encourages same-neuron pixels to cluster and different-neuron pixels to
  separate by at least a configurable margin.

Typical usage::

    model = UNetMS(embedding_dim=32, margin=0.5, scaling=4.0)
    model.to(torch.device('cuda:0'))
    output, emb_loss, bg_pred = model(images, labels)

References:
    - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
      Segmentation", MICCAI 2015.
    - Comaniciu & Meer, "Mean Shift: A Robust Approach Toward Feature Space
      Analysis", IEEE TPAMI 2002.
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mm
import numpy as np
import matplotlib.pyplot as plt

from src.utils import config as c
from src.training import helpers as h
from src.training import helpers as he
from src.visualization import visualization as v

logger = logging.getLogger(__name__)


def get_conv_layer(num_input: int, num_output: int) -> nn.Sequential:
    """Create a standard convolution block used throughout the U-Net.

    Each block consists of a 3×3 convolution (stride 1, same padding) followed
    by Batch Normalization with momentum 0.5.  ReLU activation is applied
    externally in the forward pass so that the block can be reused flexibly.

    Args:
        num_input: Number of input feature channels.
        num_output: Number of output feature channels.

    Returns:
        A ``nn.Sequential`` container with Conv2d → BatchNorm2d.
    """
    return nn.Sequential(
        nn.Conv2d(num_input, num_output, 3, stride=1, padding=(1, 1)),
        nn.BatchNorm2d(num_output, momentum=0.5),
    )


def get_up_layer(num_input: int, num_output: int) -> nn.Sequential:
    """Create a transposed-convolution upsampling block for the U-Net decoder.

    Uses a 2×2 transposed convolution with stride 2 to double spatial resolution,
    followed by Batch Normalization.  This learned upsampling replaces simpler
    bilinear interpolation for better reconstruction quality.

    Args:
        num_input: Number of input feature channels.
        num_output: Number of output feature channels.

    Returns:
        A ``nn.Sequential`` container with ConvTranspose2d → BatchNorm2d.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(num_input, num_output, kernel_size=2, stride=2),
        nn.BatchNorm2d(num_output, momentum=0.5),
    )


def normalize(input_matrix: torch.Tensor) -> torch.Tensor:
    """L2-normalize a tensor along dimension 2.

    Args:
        input_matrix: Input tensor of arbitrary shape with at least 3 dimensions.

    Returns:
        L2-normalized tensor with unit norm along dim=2.
    """
    return F.normalize(input_matrix, p=2, dim=2)


def cos_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Compute the rescaled cosine similarity in [0, 1].

    Maps the standard cosine similarity from [-1, 1] to [0, 1] via
    ``0.5 * (1 + cos(v1, v2))``.

    Args:
        vec1: First 1-D embedding vector.
        vec2: Second 1-D embedding vector.

    Returns:
        Scalar similarity score in [0, 1].
    """
    return 1 / 2 * (1 + torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2)))


class UNet(nn.Module):
    """U-Net encoder–decoder for dense pixel-wise embedding prediction.

    A 4-level U-Net architecture with skip connections, designed for producing
    per-pixel embeddings from multi-channel calcium-imaging correlation images.

    Architecture overview (default config: 6 input channels, 32-dim embedding)::

        Encoder path (spatial resolution halved at each level):
            Level 1:  6  → 32  → 32   (H×W)        no dropout
            Level 2:  32 → 64  → 64   (H/2×W/2)    dropout=0.25
            Level 3:  64 → 128 → 128  (H/4×W/4)    dropout=0.50
            Level 4: 128 → 256 → 256  (H/8×W/8)    dropout=0.50

        Bottleneck:
            256 → 512 → 512            (H/16×W/16)  dropout=0.50

        Decoder path (spatial resolution doubled at each level):
            Level 4: 512→256 (up) + cat(256) → 256  dropout=0.50
            Level 3: 256→128 (up) + cat(128) → 128  dropout=0.50
            Level 2: 128→64  (up) + cat(64)  → 64   dropout=0.25
            Level 1:  64→32  (up) + cat(32)  → 32

        Head: Conv1×1 → embedding_dim (+ 2 if background_pred)

    Design rationale:
        - Skip connections preserve fine spatial detail for precise segmentation.
        - BatchNorm stabilizes training with the small batch sizes typical of
          large medical images.
        - Progressive dropout (0.25→0.50) regularizes deeper layers that have
          more parameters and are more prone to overfitting.
        - The 1×1 convolution head projects to the target embedding dimensionality
          without altering spatial resolution.

    Args:
        input_channels: Number of input image channels (default: from config).
        embedding_dim: Dimensionality of the output embedding space.
        dropout_rate: Base dropout rate; deeper layers use 2× this value.
        background_pred: If True, adds 2 extra output channels for
            foreground/background classification.

    Input:
        Tensor of shape ``(B, C_in, H, W)``.

    Output:
        Tensor of shape ``(B, D, H, W)`` where D = embedding_dim
        (or D = embedding_dim + 2 when background_pred=True).
    """

    def __init__(
        self,
        input_channels: int = c.UNet['input_channels'],
        embedding_dim: int = c.UNet['embedding_dim'],
        dropout_rate: float = c.UNet['dropout_rate'],
        background_pred: bool = False,
    ) -> None:
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.background_pred = background_pred
        # Base filter count; doubles at each encoder level (32→64→128→256→512)
        self.filters = 32

        # ── Encoder Level 1: C_in → 32, full resolution ──────────────────
        self.conv_layer_1 = get_conv_layer(self.input_channels, self.filters)
        self.conv_layer_2 = get_conv_layer(self.filters, self.filters)

        # ── Encoder Level 2: 32 → 64, resolution /2 ─────────────────────
        self.max_pool_2D_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_3 = get_conv_layer(self.filters, self.filters * 2)
        self.conv_layer_4 = get_conv_layer(self.filters * 2, self.filters * 2)
        self.dropout_1 = nn.Dropout(p=self.dropout_rate)

        # ── Encoder Level 3: 64 → 128, resolution /4 ────────────────────
        self.max_pool_2D_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_5 = get_conv_layer(self.filters * 2, self.filters * 4)
        self.conv_layer_6 = get_conv_layer(self.filters * 4, self.filters * 4)
        self.dropout_2 = nn.Dropout(p=self.dropout_rate * 2)

        # ── Encoder Level 4: 128 → 256, resolution /8 ───────────────────
        self.max_pool_2D_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_7 = get_conv_layer(self.filters * 4, self.filters * 8)
        self.conv_layer_8 = get_conv_layer(self.filters * 8, self.filters * 8)
        self.dropout_3 = nn.Dropout(p=self.dropout_rate * 2)

        # ── Bottleneck: 256 → 512, resolution /16 ───────────────────────
        self.max_pool_2D_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_9 = get_conv_layer(self.filters * 8, self.filters * 16)
        self.conv_layer_10 = get_conv_layer(self.filters * 16, self.filters * 16)
        self.up_layer_1 = get_up_layer(self.filters * 16, self.filters * 8)
        self.dropout_4 = nn.Dropout(p=self.dropout_rate * 2)

        # ── Decoder Level 4: cat(256+256)=512 → 256, resolution /8 ──────
        self.conv_layer_11 = get_conv_layer(self.filters * 16, self.filters * 8)
        self.conv_layer_12 = get_conv_layer(self.filters * 8, self.filters * 8)
        self.up_layer_2 = get_up_layer(self.filters * 8, self.filters * 4)
        self.dropout_5 = nn.Dropout(p=self.dropout_rate * 2)

        # ── Decoder Level 3: cat(128+128)=256 → 128, resolution /4 ──────
        self.conv_layer_13 = get_conv_layer(self.filters * 8, self.filters * 4)
        self.conv_layer_14 = get_conv_layer(self.filters * 4, self.filters * 4)
        self.up_layer_3 = get_up_layer(self.filters * 4, self.filters * 2)
        self.dropout_6 = nn.Dropout(p=self.dropout_rate * 2)

        # ── Decoder Level 2: cat(64+64)=128 → 64, resolution /2 ─────────
        self.conv_layer_15 = get_conv_layer(self.filters * 4, self.filters * 2)
        self.conv_layer_16 = get_conv_layer(self.filters * 2, self.filters * 2)
        self.up_layer_4 = get_up_layer(self.filters * 2, self.filters)
        self.dropout_7 = nn.Dropout(p=self.dropout_rate)

        # ── Decoder Level 1: cat(32+32)=64 → 32, full resolution ────────
        self.conv_layer_17 = get_conv_layer(self.filters * 2, self.filters)
        self.conv_layer_18 = get_conv_layer(self.filters, self.filters)

        # ── Output Head: 1×1 conv to target embedding dimensionality ─────
        # When background_pred is enabled, 2 extra channels are appended for
        # a foreground/background classification head trained with CrossEntropyLoss.
        if self.background_pred:
            self.conv_layer_end = nn.Conv2d(self.filters, self.embedding_dim + 2, 1)
        else:
            self.conv_layer_end = nn.Conv2d(self.filters, self.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net encoder–decoder.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, D, H, W)``.
        """
        # ── Encoder Level 1 ──────────────────────────────────────────────
        x = self.conv_layer_1(x)
        x = F.relu(x)
        x = self.conv_layer_2(x)
        x = F.relu(x)
        conc_in_1 = x.clone()  # skip connection for decoder level 1

        # ── Encoder Level 2 ──────────────────────────────────────────────
        x = self.max_pool_2D_1(x)
        x = self.conv_layer_3(x)
        x = F.relu(x)
        x = self.conv_layer_4(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        conc_in_2 = x.clone()  # skip connection for decoder level 2

        # ── Encoder Level 3 ──────────────────────────────────────────────
        x = self.max_pool_2D_2(x)
        x = self.conv_layer_5(x)
        x = F.relu(x)
        x = self.conv_layer_6(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        conc_in_3 = x.clone()  # skip connection for decoder level 3

        # ── Encoder Level 4 ──────────────────────────────────────────────
        x = self.max_pool_2D_3(x)
        x = self.conv_layer_7(x)
        x = F.relu(x)
        x = self.conv_layer_8(x)
        x = F.relu(x)
        x = self.dropout_3(x)
        conc_in_4 = x.clone()  # skip connection for decoder level 4

        # ── Bottleneck ───────────────────────────────────────────────────
        x = self.max_pool_2D_4(x)
        x = self.conv_layer_9(x)
        x = F.relu(x)
        x = self.conv_layer_10(x)
        x = F.relu(x)
        x = self.up_layer_1(x)
        x = F.relu(x)
        x = self.dropout_4(x)

        # ── Decoder Level 4: cat with encoder level 4 skip ───────────────
        x = torch.cat((x.clone(), conc_in_4), dim=1)
        x = self.conv_layer_11(x)
        x = F.relu(x)
        x = self.conv_layer_12(x)
        x = F.relu(x)
        x = self.up_layer_2(x)
        x = F.relu(x)
        x = self.dropout_5(x)

        # ── Decoder Level 3: cat with encoder level 3 skip ───────────────
        x = torch.cat((x.clone(), conc_in_3), dim=1)
        x = self.conv_layer_13(x)
        x = F.relu(x)
        x = self.conv_layer_14(x)
        x = F.relu(x)
        x = self.up_layer_3(x)
        x = F.relu(x)
        x = self.dropout_6(x)

        # ── Decoder Level 2: cat with encoder level 2 skip ───────────────
        x = torch.cat((x.clone(), conc_in_2), dim=1)
        x = self.conv_layer_15(x)
        x = F.relu(x)
        x = self.conv_layer_16(x)
        x = F.relu(x)
        x = self.up_layer_4(x)
        x = F.relu(x)
        x = self.dropout_7(x)

        # ── Decoder Level 1: cat with encoder level 1 skip ───────────────
        x = torch.cat((x.clone(), conc_in_1), dim=1)
        x = self.conv_layer_17(x)
        x = F.relu(x)
        x = self.conv_layer_18(x)
        x = F.relu(x)
        x = self.conv_layer_end(x)

        return x


class MS(nn.Module):
    """Differentiable Mean-Shift module for embedding space refinement.

    Iteratively refines pixel embeddings by shifting each embedding toward the
    mode of its local density in the learned space.  The Gaussian kernel
    bandwidth is derived from the contrastive margin: ``h = 3 / (1 - margin)``
    so that the loss landscape and clustering share the same spatial scale.

    During training the module also computes the contrastive embedding loss;
    during inference it can optionally perform mean-shift iterations to sharpen
    cluster boundaries before agglomerative clustering.

    Args:
        embedding_dim: Dimensionality of pixel embeddings.
        kernel_bandwidth: Gaussian kernel bandwidth for mean-shift.  If None,
            automatically derived from *margin*.
        margin: Contrastive loss margin separating different-neuron embeddings.
        step_size: Mean-shift step size (0–1 blending factor).
        nb_iterations: Number of mean-shift iterations; 0 disables mean-shift.
        use_in_val: Whether to run mean-shift during validation.
        use_embedding_loss: Whether to compute contrastive embedding loss.
        scaling: Multiplicative scaling factor for the embedding loss.
        use_background_pred: Whether the model includes a background head.
        include_background: Whether to include background pixels in the loss.
    """

    def __init__(
        self,
        embedding_dim: int = c.mean_shift['embedding_dim'],
        kernel_bandwidth: Optional[float] = c.mean_shift['kernel_bandwidth'],
        margin: float = c.embedding_loss['margin'],
        step_size: float = c.mean_shift['step_size'],
        nb_iterations: int = c.mean_shift['nb_iterations'],
        use_in_val: bool = c.mean_shift['use_in_val'],
        use_embedding_loss: bool = c.embedding_loss['on'],
        scaling: float = c.embedding_loss['scaling'],
        use_background_pred: bool = c.UNet['background_pred'],
        include_background: bool = c.embedding_loss['include_background'],
    ) -> None:
        super(MS, self).__init__()

        self.emb = embedding_dim
        self.margin = margin
        self.scaling = scaling
        self.include_background = include_background

        # Kernel bandwidth: when not explicitly set, derive from the contrastive
        # margin so that the mean-shift mode-seeking and the loss margin are
        # consistent: h = 3 / (1 - margin).
        self.kernel_bandwidth = kernel_bandwidth
        if self.kernel_bandwidth is None:
            self.kernel_bandwidth = 3. / (1. - self.margin)
        self.step_size = step_size
        self.nb_iterations = nb_iterations
        self.iter = nb_iterations

        self.use_in_val = use_in_val
        self.use_embedding_loss = use_embedding_loss
        self.use_background_pred = use_background_pred

        self.bs = None
        self.w = None
        self.h = None
        self.val = False
        self.test = False

        self.prefer_cell = 0.5

        # NOTE: .cuda() hard-codes device cuda:0.  For multi-GPU or CPU
        # execution the device should be passed as a parameter.
        self.criterion = EmbeddingLoss(margin=self.margin).cuda()
        self.L2Norm = L2Norm()

    def forward(
        self,
        x_in: torch.Tensor,
        lab_in: torch.Tensor,
        subsample_size: Optional[int],
        background_pred: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Run mean-shift refinement and compute embedding loss.

        Args:
            x_in: L2-normalized embedding tensor ``(B, D, H, W)``.
            lab_in: Ground-truth label tensor ``(B, H, W)``.
            subsample_size: Number of pixels to subsample for the loss
                computation (memory-efficient training).  ``None`` uses all.
            background_pred: Optional background prediction tensor.

        Returns:
            Tuple of (refined embeddings ``(B, D, H, W)``, scalar loss value).
        """

        # Device is hard-coded to cuda:0; see config.py cuda settings for
        # multi-device support.
        d = torch.device('cuda:0')

        (self.bs, self.emb, self.w, self.h) = x_in.size()
        (self.sw, self.sh) = (self.w, self.h)

        x = x_in.view(self.bs, self.emb, self.w, self.h)
        out = x_in.view(self.bs, self.emb, self.w, self.h)
        y = torch.zeros(self.emb, self.w * self.h, device=d)

        if self.val and not self.test:
            """Validation SUBSAMPLING"""
            # val_sub_size = 100 * 100
            val_sub_size = 91 * 91
            emb, lab, ind = he.emb_subsample(x.clone(), lab_in.clone(), include_background=True,
                                             backpred=None, prefer_cell=0.5,
                                             sub_size=val_sub_size)

            x = emb.view(self.bs, self.emb, -1)
            y = torch.zeros(self.emb, val_sub_size, device=d)
            wurzel = int(np.sqrt(val_sub_size))
            self.sw = wurzel
            self.sh = wurzel
        elif subsample_size is not None and not self.test:
            """Training SUBSAMPLING"""
            emb, lab, ind = he.emb_subsample(x.clone(), lab_in.clone(), include_background=self.include_background,
                                             backpred=background_pred, prefer_cell=self.prefer_cell,
                                             sub_size=subsample_size)
            x = emb.view(self.bs, self.emb, -1)
            y = torch.zeros(self.emb, subsample_size, device=d)
            wurzel = int(np.sqrt(subsample_size))
            self.sw = wurzel
            self.sh = wurzel

        ret_loss = 0.

        if (not self.use_in_val) and (not self.training or self.val):
            self.iter = 0
        elif self.iter == 0:
            self.iter = self.nb_iterations

        for t in range(self.iter + 1):
            # iterating over all samples in the batch
            for b in range(self.bs):
                y = x[b, :, :]

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # f = y.clone().detach().cpu().numpy()
                # f = f.reshape(3, -1).T
                # ax.scatter(f[:, 0], f[:, 1], f[:, 2])
                # plt.xlim(-1, 1)
                # plt.ylim(-1, 1)
                # ax.set_zlim(-1, 1)
                # plt.show()

                if t != 0:
                    kernel_mat = torch.exp(torch.mul(self.kernel_bandwidth, mm(y.clone().t(), y.clone())))
                    kernel_mat = torch.where(kernel_mat < 1., torch.tensor(0., device=d), kernel_mat)

                    diag_mat = torch.diag(mm(kernel_mat.t(), torch.ones(self.sw * self.sh, 1, device=d))[:, 0],
                                          diagonal=0)
                    y = mm(y.clone(),
                           torch.add(torch.mul(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))),
                                     torch.mul(1. - self.step_size, torch.eye(self.sw * self.sh, device=d))))

                if subsample_size is not None and not self.test:
                    out = out.view(self.bs, self.emb, -1).clone()
                    out[b, :, ind] = y
                    out = out.view(self.bs, self.emb, self.w, self.h).clone()
                else:
                    out[b, :, :, :] = y.view(self.emb, self.w, self.h)

            out = self.L2Norm(out)

            x = out.view(self.bs, self.emb, -1)

            if subsample_size is not None and not self.test:
                x = out.view(self.bs, self.emb, -1)[:, :, ind]

            # print('self.training', self.training)

            if self.training and self.use_embedding_loss and not self.val and self.scaling != 0.:
                lab_in_ = torch.tensor(h.get_diff_labels(lab_in.detach().cpu().numpy()), device=d)

                if subsample_size is None:
                    emb = out
                    lab = lab_in_
                elif subsample_size is not None:
                    emb = out.view(self.bs, self.emb, -1)[:, :, ind].view(self.bs, self.emb, wurzel, wurzel)
                    lab = lab_in_.view(self.bs, -1)[:, ind].view(self.bs, wurzel, wurzel)

                loss = self.criterion(emb, lab)

                loss = (loss / self.bs) * self.scaling * (1/(self.iter + 1))

                with torch.no_grad():
                    ret_loss = ret_loss + loss.item()

                if t == self.iter and not self.use_background_pred and not t == 0:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)

            elif self.val and not self.test:
                lab_in_ = torch.tensor(h.get_diff_labels(lab_in.detach().cpu().numpy()), device=d)

                emb = out
                lab = lab_in_

                if subsample_size is not None:
                    emb = out.view(self.bs, self.emb, -1)[:, :, ind].view(self.bs, self.emb, wurzel, wurzel)
                    lab = lab_in_.view(self.bs, -1)[:, ind].view(self.bs, wurzel, wurzel)

                val_criterion = EmbeddingLoss(margin=0.5).cuda()
                loss = val_criterion(emb, lab)

                loss = (loss / self.bs) * 25 * (1/(self.iter + 1))

                with torch.no_grad():
                    ret_loss = ret_loss + loss.item()

        return x_in, ret_loss


class UNetMS(nn.Module):
    """Complete segmentation model: U-Net encoder–decoder + Mean-Shift refinement.

    This is the top-level model used for training and inference.  It chains:

    1. **UNet** — produces dense pixel embeddings from correlation images.
    2. **L2Norm** — projects embeddings onto the unit hypersphere.
    3. **MS** — (optional) mean-shift iterations to sharpen cluster structure.

    When ``use_background_pred=True``, the last 2 channels of the U-Net output
    are split off as a foreground/background classification head trained with
    CrossEntropyLoss, while the remaining channels provide the embedding.

    Args:
        input_channels: Number of input correlation channels.
        embedding_dim: Embedding space dimensionality.
        dropout_rate: Base dropout rate for the U-Net.
        kernel_bandwidth: Mean-shift Gaussian kernel bandwidth.
        margin: Contrastive loss margin.
        step_size: Mean-shift step size.
        nb_iterations: Number of mean-shift iterations.
        use_in_val: Run mean-shift during validation.
        use_embedding_loss: Compute contrastive loss during training.
        scaling: Embedding loss scaling factor.
        use_background_pred: Enable auxiliary background prediction head.
        subsample_size: Pixel subsample count for the embedding loss.
        include_background: Include background pixels in the loss.
    """

    def __init__(
        self,
        input_channels: int = c.UNet['input_channels'],
        embedding_dim: int = c.UNet['embedding_dim'],
        dropout_rate: float = c.UNet['dropout_rate'],
        kernel_bandwidth: Optional[float] = c.mean_shift['kernel_bandwidth'],
        margin: float = c.embedding_loss['margin'],
        step_size: float = c.mean_shift['step_size'],
        nb_iterations: int = c.mean_shift['nb_iterations'],
        use_in_val: bool = c.mean_shift['use_in_val'],
        use_embedding_loss: bool = c.embedding_loss['on'],
        scaling: float = c.embedding_loss['scaling'],
        use_background_pred: bool = c.UNet['background_pred'],
        subsample_size: int = c.embedding_loss['subsample_size'],
        include_background: bool = c.embedding_loss['include_background'],
    ) -> None:
        super(UNetMS, self).__init__()

        # Store all hyperparameters for checkpointing and reproducibility
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels

        self.margin = margin
        self.include_background = include_background

        self.step_size = step_size
        self.kernel_bandwidth = kernel_bandwidth
        self.nb_iterations = nb_iterations

        self.subsample_size = subsample_size

        self.use_in_val = use_in_val
        self.use_embedding_loss = use_embedding_loss
        self.scaling = scaling
        self.use_background_pred = use_background_pred

        # Instantiate sub-modules
        self.UNet = UNet(
            background_pred=self.use_background_pred,
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
        )
        self.MS = MS(
            embedding_dim=self.embedding_dim,
            margin=self.margin,
            step_size=self.step_size,
            kernel_bandwidth=self.kernel_bandwidth,
            nb_iterations=self.nb_iterations,
            include_background=self.include_background,
            use_in_val=self.use_in_val,
            use_embedding_loss=self.use_embedding_loss,
            scaling=self.scaling,
            use_background_pred=self.use_background_pred,
        )
        self.L2Norm = L2Norm()

    def forward(
        self, x: torch.Tensor, lab: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, float], Tuple[torch.Tensor, float, torch.Tensor]]:
        """Forward pass: U-Net → L2Norm → Mean-Shift → Loss.

        Args:
            x: Input correlation image ``(B, C_in, H, W)``.
            lab: Ground-truth labels ``(B, H, W)``.

        Returns:
            If ``use_background_pred`` is True:
                ``(embeddings, embedding_loss, background_logits)``
            Otherwise:
                ``(embeddings, embedding_loss)``
        """
        x = self.UNet(x)

        if self.use_background_pred and self.embedding_dim == 0:
            # Pure background prediction mode (no embedding head)
            return x, 0., x
        elif self.use_background_pred:
            # Split: first D channels are embeddings, last 2 are bg/fg logits
            x = x.clone()[:, :-2]
            y = x.clone()[:, -2:]
            x = self.L2Norm(x)
            x, ret_loss = self.MS(x, lab, background_pred=y, subsample_size=self.subsample_size)
            return x, ret_loss, y
        else:
            x = self.L2Norm(x)
            x, ret_loss = self.MS(x, lab, subsample_size=self.subsample_size)
            return x, ret_loss


class L2Norm(nn.Module):
    """L2-normalize embeddings along the channel dimension.

    Projects each pixel's embedding vector onto the unit hypersphere,
    enabling cosine similarity as a natural distance metric.
    """

    def __init__(self) -> None:
        super(L2Norm, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each pixel embedding to unit L2 norm.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of the same shape with unit-norm embeddings.
        """
        (bs, c, w, h) = x.size()

        for b in range(bs):
            x[b] = (x[b].clone().view(c, -1) / torch.norm(x[b].clone().view(c, -1), p=2, dim=0)).view(c, w, h)

        return x


class EmbeddingLoss(nn.Module):
    """Contrastive embedding loss with inverse-frequency weighting.

    For every pair of pixel embeddings the loss encourages:
    - Same-neuron pairs: cosine similarity → 1
    - Different-neuron pairs: cosine similarity ≤ margin

    Weighting by inverse label frequency ensures that rare small neurons
    contribute equally to the loss despite class imbalance.

    Args:
        margin: Minimum cosine-similarity separation for negative pairs.
    """

    def __init__(self, margin: float) -> None:
        super(EmbeddingLoss, self).__init__()
        self.margin = margin

    def forward(self, emb: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """Compute the embedding loss.

        Args:
            emb: Embedding tensor ``(B, D, H, W)``.
            lab: Label tensor ``(B, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        return embedding_loss(emb, lab, margin=self.margin)


def comp_similarity_matrix(input: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise cosine-similarity matrix for all pixels.

    Similarities are rescaled to [0, 1] via ``0.5 * (1 + cos)``.

    Args:
        input: Embedding tensor ``(B, C, H, W)``.

    Returns:
        Similarity matrix ``(N, N, 1, B)`` where N = H*W.
    """
    # NOTE: device is hard-coded to cuda:0 — see config.py for multi-GPU.
    d = torch.device('cuda:0')

    (bs, ch, w, h) = input.size()
    out = torch.zeros((h * w, h * w, 1, bs), device=d)

    for i in range(bs):
        sim = input[i].view(ch, w * h)

        sim_ = torch.mean(sim, dim=0)
        sim_n = sim - sim_
        sim__ = torch.sqrt(torch.sum(sim_n ** 2, dim=0))
        sim = (sim_n / sim__).t()
        # Replace NaN (from zero-norm pixels) with zeros
        sim = torch.where(torch.isnan(sim) != 1, sim, torch.zeros_like(sim, device=d))

        out[:, :, 0, i] = torch.mm(sim, sim.t()) * 0.5 + 0.5

    return out


def compute_pre_weight_matrix(input: torch.Tensor) -> torch.Tensor:
    """Compute inverse-frequency weights per pixel from ground-truth labels.

    Each pixel is assigned a weight of ``1 / count(its_label)`` so that
    neurons with fewer pixels contribute equally to the loss.

    Args:
        input: Label tensor ``(B, H, W)``.

    Returns:
        Weight tensor ``(B, H, W)`` with per-pixel inverse-frequency weights.
    """

    d = torch.device('cuda:0')

    (bs, w, h) = input.size()
    out = torch.zeros_like(input, device=d)

    for i in range(bs):
        with torch.no_grad():
            # calculate pre-weighting matrix
            out[i] = input[i]
            unique_labels = torch.unique(input[i], sorted=False)
            for _, l in enumerate(unique_labels):
                out[i] = torch.where(out[i] == l.float(), torch.div(torch.tensor(
                    1., device=d), torch.tensor(((input[i] == l).nonzero().size(0)), device=d)), out[i])
    return out


def compute_weight_matrix(input: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise weight matrix for the embedding loss.

    Forms the outer product of per-pixel inverse-frequency weights so that
    each pixel pair ``(i, j)`` receives weight ``w_i * w_j``.

    Args:
        input: Per-pixel weight tensor ``(B, H, W)`` from
            :func:`compute_pre_weight_matrix`.

    Returns:
        Pairwise weight matrix ``(N, N, 1, B)`` where N = H*W.
    """

    d = torch.device('cuda:0')

    (bs, w, h) = input.size()

    out = torch.zeros((h * w, h * w, 1, bs), device=d)

    for i in range(bs):
        sim = input[i].view(h * w, 1)
        out[:, :, 0, i] = torch.mm(sim, sim.t())

    return out


def compute_label_pair(input: torch.Tensor) -> torch.Tensor:
    """Compute a pairwise label-match indicator matrix.

    For every pixel pair, assigns:
    - ``+1`` if both pixels share the same label (positive pair)
    - ``-1`` if labels differ (negative pair)
    - ``0`` for background-involving pairs (excluded from loss)

    Args:
        input: Label tensor ``(B, H, W)``.

    Returns:
        Label-pair matrix ``(N, N, 1, B)`` where N = H*W.
    """

    d = torch.device('cuda:0')

    (bs, w, h) = input.size()

    # +1 such that background has label != 0 such that following computation works
    y = torch.add(input.to(torch.float), 1.)

    out = torch.zeros((h * w, h * w, 1, bs), device=d)

    # 1 if positive pair, -1 if negative pair, 0 if repeating information
    for i in range(bs):
        sim = y[i].view(h * w, 1)
        out[:, :, 0, i] = torch.mm(sim, sim.t())
        out[:, :, 0, i] = torch.where(torch.sqrt(out[:, :, 0, i]) == torch.mm(sim, torch.ones_like(sim.t())),
                                      torch.tensor(1., device=d), torch.tensor(-1., device=d))
        out[:, :, 0, i] = torch.where(torch.mm(sim, sim.t()) == 0., torch.tensor(0., device=d), out[:, :, 0, i])
    return out


def embedding_loss(emb: torch.Tensor, lab: torch.Tensor, margin: float) -> torch.Tensor:
    """Compute the contrastive embedding loss over all pixel pairs.

    For positive pairs (same neuron): ``loss = 1 - similarity``.
    For negative pairs (different neurons): ``loss = max(0, similarity - margin)``.
    Each pair is weighted by inverse label frequency to handle class imbalance.

    Args:
        emb: Embedding tensor ``(B, D, H, W)``.
        lab: Label tensor ``(B, H, W)``.
        margin: Minimum separation for negative pairs in [0, 1].

    Returns:
        Scalar loss summed over all pairs and batch elements.
    """

    d = torch.device('cuda:0')

    (bs, ch, w, h) = emb.size()

    loss = torch.zeros(bs, w * h, w * h, device=d)
    weights = compute_weight_matrix(compute_pre_weight_matrix(lab))
    label_pairs = compute_label_pair(lab)
    sim_mat = comp_similarity_matrix(emb)

    for b in range(bs):
        loss[b] = torch.where(label_pairs[:, :, 0, b] == 1., torch.sub(1., sim_mat[:, :, 0, b]),
                              loss[b])
        # correcting machine inaccuracies
        loss[b] = torch.where(loss[b] < 0., torch.tensor(0., device=d), loss[b])
        loss[b] = torch.where(label_pairs[:, :, 0, b] == -1.,
                              torch.where(sim_mat[:, :, 0, b] - margin >= 0,
                                          sim_mat[:, :, 0, b] - margin, torch.tensor(
                                      0., device=d)), loss[b])

        loss[b] = torch.mul(1. / (w * h), torch.mul(weights[:, :, 0, b], loss[b].clone()))

    return torch.sum(loss)


def scaling_loss(loss_vec: torch.Tensor, bs: int, nb_gpus: int) -> float:
    """Scale the loss correctly when using DataParallel across multiple GPUs.

    Each GPU processes ``bs / nb_gpus`` samples (with a possible remainder on
    the last GPU).  This function re-weights each GPU's partial loss to produce
    the correct mean loss over the full batch.

    Args:
        loss_vec: Per-GPU loss values.
        bs: Total batch size.
        nb_gpus: Number of active GPUs.

    Returns:
        Correctly weighted scalar loss.
    """
    logger.info('number of gpus: %d, batch size: %d', nb_gpus, bs)
    assert bs >= nb_gpus, 'Batch Size should be bigger than the number of working gpus'
    out = 0.
    rem = bs % nb_gpus

    # weighing the single gpus regarding their batches
    if rem != 0:
        nb_gpus = nb_gpus - 1
        out = out + loss_vec[-1] / rem
    b = (bs - rem) / float(nb_gpus)
    for g in range(nb_gpus):
        out = out + loss_vec[g] / b

    # weighing the loss depending on the total number of gpus
    if rem != 0:
        nb_gpus = nb_gpus + 1

    return out / nb_gpus
