import torch

UNet = dict(
    input_channels=26,                       # specifies the number of channels of the input image
    embedding_dim=20,                       # sets the base embedding dimension of UNet
    dropout_rate=0.25,                      # sets the dropout rate in UNet Model
)

mean_shift = dict(
    embedding_dim=20,
    kernel_bandwidth=None,                  # set to float if should be used, margin is now used to calculate bandwidth
    step_size=1,                            # mean shift step size
    nb_iterations=0,                       # number of iterations, if < 1 model UNet with Unit Sphere Normalization
)

embedding_loss = dict(
    margin=0.5,
)

data = dict(
    different_labels=True,
    use_compression=True,
    dtype=torch.float,
)

corr = dict(
    corr_form='big_star',
    use_slicing=False,
)

training = dict(
    lr=0.00025,
    nb_epochs=1000,
    img_size=64,

    min_neuron_pixels=0.1,
)

cuda = dict(
    device=torch.device('cuda'),
)
