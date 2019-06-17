import torch

UNet = dict(
    input_channels=28,                       # specifies the number of channels of the input image
    embedding_dim=20,                       # sets the base embedding dimension of UNet
    dropout_rate=0.25,                      # sets the dropout rate in UNet Model
)

mean_shift = dict(
    embedding_dim=20,
    kernel_bandwidth=None,                  # set to float if should be used, margin is now used to calculate bandwidth
    step_size=1.,                            # mean shift step size
    nb_iterations=5,                       # number of iterations, if < 1 model UNet with Unit Sphere Normalization
)

embedding_loss = dict(
    margin=0.5,
)

data = dict(
    different_labels=True,
    use_compression=True,
    dtype=torch.float,
    snapshots='models/'
)

corr = dict(
    corr_form='starmy',
    use_slicing=False,
)

training = dict(
    train=True,
    lr=0.00025,
    nb_epochs=10,
    img_size=64,
    batch_size=20,

    min_neuron_pixels=0.1,
)

cuda = dict(
    use_mult=False,
    device=torch.device('cuda:0'),
    mult_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_devices=[0, 1, 2, 3],
)

tb = dict(
    loss_name='iter_0'
)

debug = dict(
    add_emb=False,
    umap_img=False,
    print_img=False,
    print_img_steps=20,
)
