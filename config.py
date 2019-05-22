UNet = dict(
    input_channels=1,                       # specifies the number of channels of the input image
    embedding_dim=32,                       # sets the base embedding dimension of UNet
    dropout_rate=0.25,                      # sets the dropout rate in UNet Model
)

mean_shift = dict(
    embedding_dim=32,
    kernel_bandwidth=None,                  # set to float if should be used, margin is now used to calculate bandwidth
    step_size=1,                            # mean shift step size
    nb_iterations=10,                       # number of iterations
)

embedding_loss = dict(
    margin=0.5,
)

data = dict(
    different_labels=True,
    use_compression=True,
)

corr = dict(
    corr_form='small_star',
    use_slicing=False,
)