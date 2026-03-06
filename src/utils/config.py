"""Centralized configuration for all pipeline hyperparameters.

All configuration is stored as plain Python dictionaries, grouped by pipeline
stage.  This design allows experiment scripts to override individual values
without modifying this file, by passing keyword arguments to the ``Setup``
class in ``helpers.py``.

Configuration groups:
    UNet            Model architecture (channels, embedding dim, dropout)
    mean_shift      Mean-shift clustering (bandwidth, iterations, step size)
    embedding_loss  Contrastive loss (margin, scaling, subsampling)
    data            I/O paths, dtypes, compression settings
    corr            Correlation feature extraction mask settings
    training        Optimizer and training loop (lr, epochs, batch size)
    val             Validation frequency and thresholds
    test            Testing display options
    cuda            Device configuration (single/multi-GPU, CPU threads)
    tb              TensorBoard logging settings
    debug           Debug visualization flags (disable in production)
"""

import torch

# ── Model Architecture ──────────────────────────────────────────────────────
UNet = dict(
    input_channels=6,                       # number of correlation channels fed to the U-Net
    embedding_dim=32,                       # dimensionality of the learned pixel embeddings
    dropout_rate=0.25,                      # base dropout; deeper layers use 2× this value
    background_pred=True,                   # enable auxiliary foreground/background head (+2 channels)
)

# ── Mean-Shift Clustering ───────────────────────────────────────────────────
mean_shift = dict(
    embedding_dim=32,                       # must match UNet.embedding_dim
    kernel_bandwidth=None,                  # None → auto-derive from margin: h = 3 / (1 - margin)
    step_size=1.,                           # blending factor for each mean-shift iteration
    nb_iterations=0,                        # 0 = disable mean-shift; >0 = number of iterations
    use_in_val=False,                       # whether to run mean-shift during validation
)

# ── Embedding (Contrastive) Loss ────────────────────────────────────────────
embedding_loss = dict(
    margin=0.5,                             # contrastive margin for negative pairs
    on=True,                                # enable embedding loss during training
    include_background=True,                # include background pixels in loss computation
    scaling=25.,                            # multiplicative weight λ for embedding loss
    use_subsampling=True,                   # subsample pixels for memory-efficient loss
    subsample_size=100,                     # number of pixels to subsample per image
)

# ── Data I/O ────────────────────────────────────────────────────────────────
data = dict(
    different_labels=True,                  # use differentiated (instance) labels
    use_compression=True,                   # compress saved data with hickle/gzip
    dtype=torch.float,                      # tensor dtype for training (float32)
    snapshots='models/',                    # directory for model checkpoint snapshots
    model_name='saving_config_test',        # default model name for saving
    num_workers=0,                          # DataLoader worker processes (0 = main thread)
)

# ── Correlation Features ────────────────────────────────────────────────────
corr = dict(
    corr_form='suit',                       # correlation mask geometry (suit/starmy/big_star/...)
    use_slicing=False,                      # temporal slicing for noise-robust correlation
)

# ── Training Loop ───────────────────────────────────────────────────────────
training = dict(
    train=True,                             # master training switch
    lr=0.002,                               # initial learning rate (Adam optimizer)
    nb_epochs=1000,                         # maximum training epochs
    nb_samples=100,                         # samples per epoch (via RandomSampler)
    img_size=128,                           # spatial crop size for training patches
    batch_size=1,                           # training batch size

    aux_network=False,                      # reserved for auxiliary network experiments

    min_neuron_pixels=0.1,                  # minimum neuron pixel fraction threshold

    resume=False,                           # resume from checkpoint path (False = start fresh)
)

# ── Validation ──────────────────────────────────────────────────────────────
val = dict(
    val_freq=1,                             # validate every N epochs
    th_nn=0.8,                              # nearest-neighbor clustering threshold
    th_sl=1.,                               # single-linkage clustering threshold

    show_img=False,                         # display prediction images during validation
)

# ── Testing ─────────────────────────────────────────────────────────────────
test = dict(
    show_img=True,                          # display prediction images during testing
)

# ── CUDA / Device Configuration ─────────────────────────────────────────────
# NOTE: Default device is cuda:0. For CPU-only execution, change device to
# torch.device('cpu'). Multi-GPU training uses DataParallel with device_ids.
cuda = dict(
    use_mult=False,                         # enable multi-GPU DataParallel
    device=torch.device("cuda:0"),          # primary compute device
    mult_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_devices=[0, 1, 2, 3],              # GPU device IDs for DataParallel
    nb_cpu_threads=3,                       # torch.set_num_threads() value
)

# ── TensorBoard Logging ────────────────────────────────────────────────────
tb = dict(
    loss_name='saving_config_test',         # TensorBoard run identifier
    pre_train=False,                        # load pre-trained weights before training
    pre_train_name='background',            # model name to load pre-trained weights from
)

# ── Debug Flags ─────────────────────────────────────────────────────────────
# Set all to False for production training runs.
debug = dict(
    add_emb=False,                          # add embeddings to TensorBoard
    umap_img=False,                         # generate UMAP visualizations
    print_img=False,                        # display intermediate images
    print_input=False,                      # display input images
    print_img_steps=20,                     # display frequency (every N batches)
    print_grad_upd=False,                   # print gradient update statistics
)
