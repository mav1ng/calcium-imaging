import torch

UNet = dict(
    input_channels=6,                       # specifies the number of channels of the input image
    embedding_dim=32,                       # sets the base embedding dimension of UNet
    dropout_rate=0.25,                      # sets the dropout rate in UNet Model
    background_pred=True,
)

mean_shift = dict(
    embedding_dim=32,
    kernel_bandwidth=None,                  # set to float if should be used, margin is now used to calculate bandwidth
    step_size=1.,                            # mean shift step size
    nb_iterations=5,                            # number of iterations, if < 1 model UNet with Unit Sphere Normalization
    use_in_val=False,
)

embedding_loss = dict(
    margin=0.5,
    on=True,
    include_background=True,
    scaling=25.,
    use_subsampling=True,
    subsample_size=100,
)

data = dict(
    different_labels=True,
    use_compression=True,
    dtype=torch.float,
    snapshots='models/',
    model_name='saving_config_test',
    num_workers=0,
)

corr = dict(
    corr_form='suit',
    use_slicing=False,
)

training = dict(
    train=True,
    lr=0.002,
    nb_epochs=1000,
    nb_samples=100,
    img_size=64,
    batch_size=20,

    aux_network=False,

    min_neuron_pixels=0.1,

    resume=False,
)

val = dict(
    val_freq=1,
    th_nn=0.8,
    th_sl=1.,

    show_img=False,
)

test = dict(
    show_img=True,
)

cuda = dict(
    use_mult=False,
    device=torch.device("cuda:0"),
    mult_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_devices=[0, 1, 2, 3],
    nb_cpu_threads=5,
)

tb = dict(
    loss_name='saving_config_test',
    pre_train=False,
    pre_train_name='background',
)

debug = dict(
    add_emb=False,
    umap_img=False,
    print_img=False,
    print_input=False,
    print_img_steps=20,
    print_grad_upd=False,
)
