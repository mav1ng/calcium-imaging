def get_input_diag(part_nb, dataset):
    """method that returns a sliced image along the diagonal from the dataset"""

    comb_dataset = dataset
    tn = part_nb
    input = comb_dataset[0]['image'][:, tn * 64:(tn + 1) * 64, tn * 64:(tn + 1) * 64].view(1, 10, 64, 64).cuda()
    label = comb_dataset[0]['label'][tn * 64:(tn + 1) * 64, tn * 64:(tn + 1) * 64].view(1, 64, 64).cuda()

    return input, label

