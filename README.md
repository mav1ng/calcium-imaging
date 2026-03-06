# Neuron Segmentation in Calcium Imaging Data Using Deep Learning

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Bachelor Thesis Project** — End-to-end deep learning pipeline for automated neuron detection and segmentation in calcium imaging recordings, built from scratch in PyTorch.

---

## Abstract

Calcium imaging is a widely used technique in neuroscience for recording the activity of large neuronal populations *in vivo*. A central challenge is the automated identification and segmentation of individual neurons from the resulting fluorescence image sequences. Traditional approaches rely on hand-crafted feature extraction and manual annotation, which are time-consuming and do not scale to modern large-field-of-view recordings.

This project presents a **fully convolutional deep learning approach** to neuron segmentation based on a **U-Net encoder–decoder architecture** combined with **pixel-wise embedding learning** and **mean-shift clustering**. The model learns to map each pixel in a calcium imaging correlation image to a high-dimensional embedding space where pixels belonging to the same neuron are close together and pixels from different neurons are far apart. At inference time, a mean-shift procedure and agglomerative clustering recover individual neuron instances from the learned embedding space.

The complete pipeline — from raw calcium imaging data preprocessing through correlation feature extraction, model training with custom loss functions, to post-processing and quantitative evaluation using the [Neurofinder](http://neurofinder.codeneuro.org/) benchmark — is implemented end-to-end in Python and PyTorch.

---

## Methodology

### Data Preprocessing & Correlation Features

Raw calcium imaging videos (T × H × W) are transformed into multi-channel **spatial correlation images** that capture the temporal co-activity patterns between neighboring pixels. Several correlation mask geometries are supported:

| Mask | Neighbors | Description |
|------|-----------|-------------|
| `suit` | 8 | Immediate 8-connected neighborhood |
| `small_star` | 10 | Star pattern with close and medium-range offsets |
| `starmy` | 26 | Extended star pattern including long-range spatial correlations |

Additional preprocessing includes temporal slicing with max-pooling for noise robustness, summary image generation, and optional data augmentation (random crops, rotations, flips) during training.

### Neural Network Architecture

The core model (`UNetMS`) consists of three components:

#### 1. U-Net Encoder–Decoder (`UNet`)

A 4-level U-Net with skip connections that maps a multi-channel correlation input image to a dense pixel-wise embedding:

```
Input (B × C_in × H × W)       C_in = 6 (correlation channels)
        │
   ┌────▼────┐
   │ Encoder  │  Conv3×3 → BN → ReLU (×2) + MaxPool2×2 at each level
   │ Level 1  │  32 filters
   │ Level 2  │  64 filters  + Dropout(0.25)
   │ Level 3  │  128 filters + Dropout(0.50)
   │ Level 4  │  256 filters + Dropout(0.50)
   └────┬────┘
   ┌────▼────┐
   │Bottleneck│  512 filters + Dropout(0.50)
   └────┬────┘
   ┌────▼────┐
   │ Decoder  │  ConvTranspose2×2 → BN → ReLU + skip-concat + Conv3×3 (×2)
   │ Level 4  │  256 filters + Dropout(0.50)
   │ Level 3  │  128 filters + Dropout(0.50)
   │ Level 2  │  64 filters  + Dropout(0.25)
   │ Level 1  │  32 filters
   └────┬────┘
   ┌────▼────┐
   │  Head    │  Conv1×1 → embedding_dim (or embedding_dim + 2 for background)
   └─────────┘
Output (B × D × H × W)         D = 32 (embedding dimensions)
```

Each encoder block consists of two `Conv2d(3×3, padding=1) → BatchNorm2d → ReLU` layers. Spatial resolution is halved at each level via `MaxPool2d(2×2)`. Decoder blocks use `ConvTranspose2d(2×2, stride=2)` for learned upsampling, followed by concatenation of the corresponding encoder feature map (skip connection) and two convolutional blocks. Dropout rates increase with depth to regularize the higher-capacity deeper layers.

#### 2. L2 Normalization (`L2Norm`)

The U-Net output embeddings are projected onto the **unit hypersphere** via L2 normalization along the channel dimension. This constrains the embedding space and makes cosine similarity a natural distance metric for clustering.

#### 3. Mean-Shift Module (`MS`)

An optional differentiable mean-shift procedure that iteratively refines the embeddings by shifting each pixel embedding toward the mode of its local density in embedding space. The kernel bandwidth is derived from the embedding loss margin:

$$h = \frac{3}{1 - m}$$

where $m$ is the contrastive margin. This creates a self-consistent system where the loss landscape and the clustering procedure share the same spatial scale.

### Loss Functions

**Embedding Loss (Contrastive):** For each pair of pixels, the loss encourages embeddings of the same neuron to have cosine similarity → 1 and embeddings of different neurons to be separated by at least a margin $m$:

$$\mathcal{L} = \frac{1}{N} \sum_{i,j} w_{ij} \cdot \begin{cases} 1 - s_{ij} & \text{if same neuron} \\ \max(0,\; s_{ij} - m) & \text{if different neurons} \end{cases}$$

where $s_{ij}$ is the cosine similarity and $w_{ij}$ is a per-pair weight based on inverse label frequency to handle class imbalance.

**Cross-Entropy Loss:** When background prediction is enabled, a separate 2-channel head produces per-pixel foreground/background logits trained with standard cross-entropy loss.

The total loss is a weighted combination: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CEL}} + \lambda \cdot \mathcal{L}_{\text{emb}}$ where $\lambda$ (scaling factor) balances the two objectives.

### Training Procedure

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimizer | Adam | Adaptive learning rates for sparse gradients |
| Learning Rate | 0.002 | Tuned via random search |
| LR Schedule | Poly decay: $\text{lr} \cdot (1 - \frac{t}{T})^{0.9}$ | Smooth annealing to fine-tune embeddings |
| Batch Size | 1–20 | Dataset-dependent; searched via hyperparameter optimization |
| Embedding Dim | 32 | Sufficient capacity for ~20 neurons per field of view |
| Margin | 0.5 | Balances intra- vs inter-cluster distance |
| Subsample Size | 1024 | Random pixel subsampling for memory-efficient embedding loss |
| Weight Init | Xavier Normal | Stable gradients in deep encoder–decoder |
| Epochs | 50–1000 | Early stopping based on validation F1 |

Hyperparameter optimization was performed via **random search** over learning rate, batch size, embedding dimension, margin, and scaling factor, with model selection based on the Neurofinder F1 metric.

### Post-Processing & Evaluation

1. **Agglomerative Clustering** (single-linkage) or nearest-neighbor clustering converts continuous embeddings into discrete neuron instance labels
2. **Background Prediction Thresholding** removes non-neuron pixels
3. **Morphological Operations** (small object removal, hole filling) clean up the segmentation mask
4. **Neurofinder F1 Metric** — precision and recall of neuron center detection against ground truth annotations

### Systematic Ablation Studies

The project includes comprehensive experiments varying:
- Embedding dimension (2–64)
- Margin and scaling factor
- Mean-shift iterations and kernel bandwidth
- Subsample size for embedding loss
- With/without background prediction head
- Different optimizer and learning rate configurations

---

## Project Structure

```
calcium-imaging/
│
├── src/                          # Source package
│   ├── models/
│   │   └── network.py            # U-Net, Mean-Shift, L2Norm, EmbeddingLoss
│   ├── data/
│   │   ├── data.py               # Dataset classes, preprocessing, I/O
│   │   └── corr.py               # Spatial correlation computation
│   ├── training/
│   │   ├── training.py           # Weight init, LR schedules
│   │   └── helpers.py            # Training loop orchestration, evaluation utilities
│   ├── analysis/
│   │   ├── analysis.py           # Model evaluation, scoring, ablation analysis
│   │   └── clustering.py         # K-means, agglomerative clustering, post-processing
│   ├── visualization/
│   │   └── visualization.py      # PCA/UMAP embedding plots, learning curves
│   └── utils/
│       └── config.py             # Centralized hyperparameter configuration
│
├── tests/                        # Unit and integration tests
├── examples/                     # Usage examples and notebooks
│
├── main.py                       # CLI entry point for training and evaluation
├── requirements.txt              # Pinned dependencies
├── .gitignore                    # Python/PyTorch ignore rules
├── LICENSE                       # MIT License
├── CHANGELOG.md                  # Version history
└── README.md                     # This file
```

---

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended; the pipeline uses `cuda:0` by default)

### Setup

```bash
# Clone the repository
git clone https://github.com/mav1ng/calcium-imaging.git
cd calcium-imaging

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Configuration

All hyperparameters are centralized in [`src/utils/config.py`](src/utils/config.py). Key configuration groups:

```python
import src.utils.config as c

# Model architecture
c.UNet['embedding_dim']    # 32 — embedding space dimensionality
c.UNet['dropout_rate']     # 0.25 — base dropout rate
c.UNet['background_pred']  # True — enable background prediction head

# Training
c.training['lr']           # 0.002 — initial learning rate
c.training['nb_epochs']    # 1000 — maximum training epochs
c.training['batch_size']   # 1 — batch size

# Embedding loss
c.embedding_loss['margin']  # 0.5 — contrastive margin
c.embedding_loss['scaling'] # 25.0 — loss scaling factor
```

### Training a Model

```python
from src.utils.helpers import Setup

# Initialize a training run with custom hyperparameters
setup = Setup(
    model_name='my_experiment',
    embedding_dim=32,
    margin=0.5,
    scaling=4.0,
    learning_rate=0.002,
    batch_size=1,
    nb_epochs=100,
    background_pred=True,
    embedding_loss=True,
    subsample_size=1024,
    save_config=True,
)

# Run training with validation
setup.main()
```

### Evaluating a Trained Model

```python
from src.utils.helpers import test, val_score

# Run inference and visualize predictions
test('my_experiment', cl_th=1.5, pp_th=0.2, obj_size=20, hole_size=20, show_image=True)

# Compute Neurofinder F1 score on validation set
val_score('my_experiment', use_metric=True, iter=100, cl_th=1.5, pp_th=0.175)
```

### Running Hyperparameter Search

```python
from src.utils.helpers import Setup
from src.analysis.analysis import score, save_images
import numpy as np

lr_list = np.linspace(0.0001, 0.01, 10000)

for i in range(50):
    lr = np.around(np.random.choice(lr_list), decimals=5)
    setup = Setup(
        model_name=f'search_{lr}',
        learning_rate=lr,
        embedding_dim=32,
        margin=0.5,
        nb_epochs=50,
        save_config=True,
    )
    setup.main()

# Aggregate and compare results
score('search_', include_metric=True)
save_images('search_')
```

---

## Results & Key Findings

- The **embedding-based approach** with U-Net + mean-shift clustering successfully segments individual neurons in calcium imaging recordings, achieving competitive performance on the Neurofinder benchmark.
- **Background prediction** as an auxiliary task improves segmentation quality by providing an explicit foreground/background prior.
- **Embedding dimension of 32** provides sufficient representational capacity; lower dimensions (e.g., 3) are useful for visualization but sacrifice segmentation accuracy.
- **Random search** over hyperparameters is effective for this problem, with learning rate and margin being the most sensitive parameters.
- **Contrastive embedding loss with inverse-frequency weighting** handles the severe class imbalance between background and neuron pixels.
- **Morphological post-processing** (small object removal, hole filling) provides a consistent improvement in F1 score across all model variants.

---

## Skills Demonstrated

| Domain | Details |
|--------|---------|
| **Deep Learning** | Custom U-Net architecture design, embedding/metric learning, contrastive loss functions, mean-shift clustering, multi-task learning (segmentation + background prediction) |
| **PyTorch** | Custom `nn.Module` implementations, autograd-compatible differentiable mean-shift, custom loss functions, DataLoader pipelines, multi-GPU support, TensorBoard integration, model checkpointing |
| **Python** | Object-oriented design, NumPy/SciPy numerical computing, data pipeline engineering, configuration management, reproducible experiment scripting |
| **Data Science** | Hyperparameter optimization (random search), ablation studies, quantitative evaluation metrics (F1, precision, recall), statistical analysis of experimental results |
| **Computer Vision** | Image segmentation, spatial correlation features, morphological operations, data augmentation (crop, rotate, flip) |
| **Software Engineering** | Modular project structure, separation of concerns (model/data/training/analysis), configuration-driven experiments, version-controlled reproducible research |
| **Scientific Communication** | Systematic experimental methodology, quantitative result reporting, architecture documentation |

---

## Acknowledgments

- **Neurofinder** benchmark and evaluation framework: [neurofinder.codeneuro.org](http://neurofinder.codeneuro.org/)
- U-Net architecture based on [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Mean-shift clustering inspired by [Comaniciu & Meer, 2002](https://doi.org/10.1109/34.1000236)
- Embedding loss based on contrastive learning principles from metric learning literature

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.