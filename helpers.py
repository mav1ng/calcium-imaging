import numpy as np
import clustering as cl
import skimage.measure as skm


from torch import optim
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
import umap
import torch.nn as nn
import config as c
import data
import corr
import network as n
import visualization as v
import training as t
import clustering as cl
import helpers as h
import argparse
import time
import random
import numpy as np
import neurofinder as nf


def main():

    writer = SummaryWriter(log_dir='training_log/' + str(c.tb['loss_name']) + '/')

    dtype = c.data['dtype']
    batch_size = c.training['batch_size']
    device = c.cuda['device']

    device_ids = c.cuda['use_devices']
    lr = c.training['lr']
    resume = c.training['resume']
    start_epoch = 0
    img_size = c.training['img_size']
    num_workers = c.data['num_workers']
    # num_samples = c.data['num_samples']
    nb_epochs = c.training['nb_epochs']
    val_freq = c.val['val_freq']

    device = c.cuda['device']
    model = n.UNetMS(background_pred=c.UNet['background_pred'])

    model.to(device)
    model.type(dtype)

    # define loss function (criterion) and optimizer
    # perhaps add additional loss functions if this one does not work

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = n.EmbeddingLoss().cuda()
    criterionCEL = nn.CrossEntropyLoss().cuda()

    # # creating different parameter groups
    base_params = list(map(id, model.parameters()))


    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            best_f1 = 0.
    else:
        best_f1 = 0.

    # models are save only when their loss obtains the best value in the validation
    f1track = 0

    # data preparation, loaders
    # cudnn.benchmark = True

    # preparing the training loader
    transform_train = transforms.Compose([data.RandomCrop(img_size)])
    train_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/transformed_4/',
                                        transform=transform_train, device=device, dtype=dtype)
    random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=(100 * batch_size))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler,
                                               num_workers=num_workers)
    print('Training loader prepared.')

    # preparing validation loader
    transform_val = transforms.Compose([data.RandomCrop(img_size)])
    val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/transformed_4/',
                                       transform=transform_val, device=device, dtype=dtype, test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    print('Validation loader prepared.')

    # run epochs
    for epoch in range(start_epoch, nb_epochs):

        # train for one epoch
        train(train_loader, model, criterion, criterionCEL, optimizer, epoch, writer)

        # evaluate on validation set
        if (epoch + 1) % val_freq == 0 and epoch != 0:
            'rewrite THIS PART BECAUSE USING NEUROFINDER METRIC'
            f1_metric = validate(val_loader, model)

            # check patience
            if f1_metric <= best_f1:
                f1track += 1
            else:
                f1track = 0

            # save the best model
            is_best = f1_metric < best_f1
            best_f1 = min(f1_metric, best_f1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_f1,
                'optimizer': optimizer.state_dict(),
                'valtrack': f1track,
                'curr_val': f1_metric,
            }, is_best)

            print('** Validation: %f (best) - %d (f1track)' % (best_f1, f1track))


def train(train_loader, model, criterion, criterionCEL, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    emb_losses = AverageMeter()
    cel_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        input = batch['image'].cuda()
        label = batch['label'].cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        target_var = list()
        for j in range(len(label)):
            label[j] = label[j].cuda()
            target_var.append(torch.autograd.Variable(label[j]))

        input.requires_grad = True
        label.requires_grad = True

        if c.debug['print_input']:
            v.plot_emb_pca(input[0].detach(), label.detach())
            v.plot_input(input[0].detach(), label.detach())

        # zero the parameter gradients
        optimizer.zero_grad()

        if c.UNet['background_pred']:
            output, ret_loss, y = model(input, label)

            '''Cross Entropy Loss on Background Prediction'''
            cel_loss = torch.tensor(0.).cuda()
            for b in range(y.size(0)):
                lab = torch.where(label[b].flatten().long() > 0, torch.tensor(1, dtype=torch.long).cuda(),
                                  torch.tensor(0, dtype=torch.long).cuda())
                cel_loss = cel_loss.clone() + criterionCEL(y[b].view(2, -1).t(), lab)
            cel_loss = cel_loss / y.size(0)

            writer.add_scalar('Cross Entropy Loss', cel_loss.item())

            cel_loss.backward(retain_graph=True)

            if c.debug['print_img']:
                v.plot_pred_back(y[0].detach(), label.detach())
        else:
            output, ret_loss = model(input, label)

        # measure performance and record loss
        emb_losses.update(ret_loss.item())
        cel_losses.update(cel_loss.item())

        if c.debug['print_img']:
            # fig = v.draw_umap(data=output[0].detach().view(c.UNet['embedding_dim'], -1),
            #                   color=label[0].detach().flatten())
            # plt.show()

            pred_labels = cl.label_embeddings(output[0].view(c.UNet['embedding_dim'], -1).t().detach(), th=0.75)
            pred_labels2 = cl.label_emb_sl(output[0].view(c.UNet['embedding_dim'], -1).t().detach(), th=0.5)

            print('There are ' + str(torch.unique(label).size(0)) + ' clusters.')

            v.plot_sk_img(pred_labels, label.detach())
            v.plot_sk_img(pred_labels2, label.detach())

            v.plot_emb_pca(output[0].detach(), label.detach())

        writer.add_scalar('Embedding Loss', ret_loss)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: {0}\t'
              'Emb Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'CEL Loss {lossCEL.val:.4f} ({lossCEL.avg:.4f})\t'
              'Param ({param}))\t'.format(
            epoch, loss=emb_losses, lossCEL=cel_losses, param=optimizer.param_groups[0]['lr']))


def validate(val_loader, model):

    # nf_threshold = c.val['nf_threshold']

    # switch to evaluate mode
    model.eval()

    end = time.time()

    f1_metric = 0.
    (recall, precision) = (0., 0.)
    count = 0

    for i, batch in enumerate(val_loader):
        print('here we are1')
        input = batch['image'].cuda()
        label = batch['label'].cuda()

        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j], volatile=True).cuda())

        target_var = list()
        for j in range(len(label)):
            label[j] = label[j].cuda()
            target_var.append(torch.autograd.Variable(label[j], volatile=True))

        # compute output
        output, _ = model(input)

        '''Write Clustering METHOD such that Embeddings get clustered
        Write From Labels to Coordinates Method
        Write Method that saves input and target data as Jason file appropriately'''

        print(output.size())
        predict = cl.label_emb_sl(output)

        f1_metric_ = 0.
        (recall_, precision_) = (0., 0.)

        print('here we are2')
        for i in range(input.size(0)):
            (mask_true, mask_predict) = (data.toCoords(label[i]), data.toCoords(predict[i]))
            data.write_to_json(mask_true, 'data/val_mask/mask_true.json')
            data.write_to_json(mask_predict, 'data/val_mask/mask_predict.json')
            true_mask = nf.load('data/val_mask/mask_true.json')
            predict_mask = nf.load('data/val_mask/mask_predict.json')
            (recall_, precision_) = (recall_, precision_) + nf.centers(true_mask, predict_mask)
            f1_metric_ = f1_metric_ + 2 * (recall_ * precision_) / (recall_ + precision_)

        f1_metric_ = f1_metric_ / input.size(0)
        recall_ = recall_ / input.size(0)
        precision_ = precision_ / input.size()

        f1_metric = f1_metric + f1_metric_
        recall = recall + recall_
        precision = precision + precision_

        count = i

    f1_metric = f1_metric / (count + 1)
    recall = recall / (count + 1)
    precision = precision / (count + 1)

    print('* F1 Metric {f1:.4f}\t'
          'Recall {recall}\t'
          'Precision'.format(f1=f1_metric, recall=recall, precision=precision))

    return f1_metric


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    snapshots = c.data['snapshots']
    filename = snapshots + 'model_e%03d_v-%.3f.pth.tar' % (state['epoch'], state['best_val'])
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_diff_labels(mask):
    """
    returns mask with different labels
    :param mask: input np array B x w x h
    :return: mask with a different label for every neuron
    """
    (bs, w, h) = mask.shape

    for b in range(bs):
        mask[b] = skm.label(mask[b], background=0)

    return mask


def get_input_diag(part_nb, dataset):
    """method that returns a sliced image along the diagonal from the dataset"""

    comb_dataset = dataset
    tn = part_nb
    input = comb_dataset[0]['image'][:, tn * 64:(tn + 1) * 64, tn * 64:(tn + 1) * 64].view(1, 10, 64, 64).cuda()
    label = comb_dataset[0]['label'][tn * 64:(tn + 1) * 64, tn * 64:(tn + 1) * 64].view(1, 64, 64).cuda()

    return input, label


def denoise(img, weight=0.1, eps=1e-3, num_iter_max=10000):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more
        de-noising (at the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost
        function that determines the stop criterion.
        The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the
        optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)

    nm = np.prod(img.shape[:2])
    tau = 0.125

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy

        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p

        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)

        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error

        # don't forget to update iterator
        i += 1
    return u