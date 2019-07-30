import json

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
from PIL import Image


class Setup:
    def __init__(self, th_nn=c.val['th_nn']):
        self.th_nn = th_nn
        self.writer = SummaryWriter(log_dir='training_log/' + str(c.data['model_name']) + '/')
        self.device = torch.device('cuda:0')
        self.nb_cpu_threads = c.cuda['nb_cpu_threads']
        self.epoch = 0

    def train(self, train_loader, model, criterion, criterionCEL, optimizer, epoch):
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
            input = batch['image']
            label = batch['label']

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = list()
            for j in range(len(input)):
                input_var.append(torch.autograd.Variable(input[j]))

            target_var = list()
            for j in range(len(label)):
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
                cel_loss = torch.tensor(0., device=self.device)
                for b in range(y.size(0)):
                    lab = torch.where(label[b].flatten().long() > 0,
                                      torch.tensor(1, dtype=torch.long, device=self.device),
                                      torch.tensor(0, dtype=torch.long, device=self.device))
                    cel_loss = cel_loss.clone() + criterionCEL(y[b].view(2, -1).t(), lab)
                cel_loss = cel_loss / y.size(0)

                self.writer.add_scalar('Cross Entropy Loss', cel_loss.item())

                cel_loss.backward(retain_graph=True)
                cel_losses.update(cel_loss.item())

                if c.debug['print_img']:
                    v.plot_pred_back(y[0].detach(), label.detach())
            else:
                output, ret_loss = model(input, label)

            # measure performance and record loss
            try:
                emb_losses.update(ret_loss.item())
            except AttributeError:
                emb_losses.update(ret_loss)

            if c.debug['print_img']:
                # fig = v.draw_umap(data=output[0].detach().view(c.UNet['embedding_dim'], -1),
                #                   color=label[0].detach().flatten())
                # plt.show()

                pred_labels = cl.label_embeddings(output[0].view(c.UNet['embedding_dim'], -1).t().detach(),
                                                  th=c.val['th_nn'])
                pred_labels2 = cl.label_emb_sl(output[0].view(c.UNet['embedding_dim'], -1).t().detach(),
                                               th=c.val['th_sl'])

                print('There are ' + str(torch.unique(label).size(0)) + ' clusters.')

                v.plot_sk_img(pred_labels, label.detach())
                v.plot_sk_img(pred_labels2, label.detach())

                v.plot_emb_pca(output[0].detach(), label.detach())

            self.writer.add_scalar('Embedding Loss', ret_loss)

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: {0}\t'
                  'Emb Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CEL Loss {lossCEL.val:.4f} ({lossCEL.avg:.4f})\t'
                  'Param ({param}))\t'.format(
                epoch, loss=emb_losses, lossCEL=cel_losses, param=optimizer.param_groups[0]['lr']))


    def validate(self, val_loader, model, model_name=c.data['model_name']):

        # nf_threshold = c.val['nf_threshold']
        # switch to evaluate mode
        model.eval()
        model.MS.val = True

        with torch.no_grad():

            end = time.time()

            f1_metric = 0.
            (recall, precision) = (0., 0.)

            for i, batch in enumerate(val_loader):
                input = batch['image']
                label = batch['label']

                input_var = list()
                for j in range(len(input)):
                    input_var.append(torch.autograd.Variable(input[j]))

                target_var = list()
                for j in range(len(label)):
                    target_var.append(torch.autograd.Variable(label[j]))

                # compute output
                output, _, __ = model(input, label)
                '''Write Clustering METHOD such that Embeddings get clustered
                Write From Labels to Coordinates Method
                Write Method that saves input and target data as Jason file appropriately'''

                (bs, ch, w, h) = output.size()
                # predict = cl.label_emb_sl(output.view(ch, -1).t(), th=.5)
                predict = cl.label_embeddings(output.view(ch, -1).t(), th=self.th_nn)
                predict = predict.reshape(bs, w, h)

                f1_metric_ = 0.
                (recall_, precision_) = (0., 0.)

                label = get_diff_labels(label.cpu().numpy())

                for b in range(bs):
                    self.writer.add_image('Prediction Epoch ' + str(self.epoch), predict[b].reshape(1, w, h),
                                          global_step=1)

                if c.val['show_img']:
                    for b in range(bs):
                        f, axarr = plt.subplots(2)
                        axarr[0].imshow(predict[b])
                        axarr[1].imshow(label[b])

                        plt.title('Predicted Background (upper) vs Ground Truth (lower)')
                        plt.show()

                for k in range(bs):
                    (mask_true, mask_predict) = (data.toCoords(label[k]), data.toCoords(predict[k]))
                    data.write_to_json(mask_true, 'data/val_mask/mask_true_' + str(model_name) + '.json')
                    data.write_to_json(mask_predict, 'data/val_mask/mask_predict_' + str(model_name) + '.json')
                    true_mask = nf.load('data/val_mask/mask_true_' + str(model_name) + '.json')
                    predict_mask = nf.load('data/val_mask/mask_predict_' + str(model_name) + '.json')
                    recall_c, precision_c = nf.centers(true_mask, predict_mask)
                    recall_ += recall_c
                    precision_ += precision_c
                    f1_metric_ = f1_metric_ + 2 * (recall_ * precision_) / (recall_ + precision_)

                f1_metric_ = f1_metric_ / bs
                recall_ = recall_ / bs
                precision_ = precision_ / bs

                f1_metric = f1_metric + f1_metric_
                recall = recall + recall_
                precision = precision + precision_

            count = val_loader.__len__()

            f1_metric = f1_metric / count
            recall = recall / count
            precision = precision / count

            print('* F1 Metric {f1:.4f}\t'
                  'Recall {recall}\t'
                  'Precision {precision}\t'.format(f1=f1_metric, recall=recall, precision=precision))

            model.MS.val = False

            self.writer.add_scalar('F1', f1_metric)
            self.writer.add_scalar('Recall', recall)
            self.writer.add_scalar('Precision', precision)

        return f1_metric


    def main(self):

        dtype = c.data['dtype']
        batch_size = c.training['batch_size']
        device = c.cuda['device']

        device_ids = c.cuda['use_devices']
        lr = c.training['lr']
        resume = c.training['resume']
        start_epoch = 0
        img_size = c.training['img_size']
        num_workers = c.data['num_workers']
        nb_samples = c.training['nb_samples']
        nb_epochs = c.training['nb_epochs']
        val_freq = c.val['val_freq']

        torch.set_num_threads(self.nb_cpu_threads)

        device = c.cuda['device']
        model = n.UNetMS(background_pred=c.UNet['background_pred'])

        model.to(device)
        model.type(dtype)

        # loading old weights
        try:
            if c.tb['pre_train']:
                model.load_state_dict(torch.load('model/model_weights_' + str(c.tb['pre_train_name']) + '.pt'))
            else:
                model.load_state_dict(torch.load('model/model_weights_' + str(c.tb['loss_name']) + '.pt'))
            print('Loaded Model!')
        except IOError:
            print('No Model to load from!')
            print('Initializing Weights and Bias')
            model.apply(t.weight_init)
            print('Finished Initializing')

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
                best_val = checkpoint['best_val']
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
        train_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                             corr_sum_folder='data/corr_sum_img/',
                                             sum_folder='data/sum_img/',
                                             mask_folder='data/sum_masks/',
                                             transform=transform_train, device=device, dtype=dtype)
        random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True,
                                                        num_samples=(nb_samples * batch_size))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler,
                                                   num_workers=num_workers)
        print('Training loader prepared.')

        # preparing validation loader
        transform_val = transforms.Compose([data.RandomCrop(img_size, val=True)])
        val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                           corr_sum_folder='data/corr_sum_img/',
                                           sum_folder='data/sum_img/',
                                           mask_folder='data/sum_masks/',
                                           transform=None, device=device, dtype=dtype, test=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=num_workers)
        print('Validation loader prepared.')

        # run epochs
        for epoch in range(start_epoch, nb_epochs):

            self.epoch = epoch

            # train for one epoch
            self.train(train_loader, model, criterion, criterionCEL, optimizer, epoch)

            # evaluate on validation set
            if (epoch + 1) % val_freq == 0 and epoch != 0:
                'rewrite THIS PART BECAUSE USING NEUROFINDER METRIC'
                f1_metric = self.validate(val_loader, model)

                # check patience
                if f1_metric <= best_f1:
                    f1track += 1
                else:
                    f1track = 0

                # save the best model
                is_best = f1_metric > best_f1
                best_f1 = max(f1_metric, best_f1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val': best_f1,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': f1track,
                    'curr_val': f1_metric,
                }, is_best)

                print('** Validation: %f (best) - %d (f1track)' % (best_f1, f1track))

            print('Saved Model After Epoch')
            torch.save(model.state_dict(), 'model/model_weights_' + str(c.tb['loss_name']) + '.pt')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    snapshots = c.data['snapshots']
    filename = snapshots + c.data['model_name'] +'_e%03d_v-%.3f.pth.tar' % (state['epoch'], state['best_val'])
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


def pad_nf(array, img_size=512, labels=False):
    """
    Padding the input array
    :param array: T x W x H , T time series, width , height
    :param img_size: wanted image size W x H
    :return: padded array
    """

    w = array.shape[-2]
    h = array.shape[-1]

    assert w <= img_size and h <= img_size, 'Image cannot be padded to be smaller'

    if w == img_size and h == img_size:
        return array

    w_ = w % 2
    h_ = h % 2

    w_pad = int((img_size - w) / 2)
    h_pad = int((img_size - h) / 2)

    if not labels:
        return np.pad(array, ((0, 0), (w_pad + w_, w_pad), (h_pad + h_, h_pad)), 'constant', constant_values=0)
    else:
        return np.pad(array, ((w_pad + w_, w_pad), (h_pad + h_, h_pad)), 'constant', constant_values=0)


def emb_subsample(embedding_tensor, label_tensor, sub_size=100):
    """
    Method that returns a subsampled amount of pixel to calculate the embedding loss
    :param embedding_tensor: Bs x Ch x w x h
    :param label_tensor: Bs x w x h
    :param sub_size: Parameter that specifies the number of pixels in subsampled image
    :return: subsampled embedding and labels
    """

    assert np.sqrt(sub_size) % 2 == 0, 'Sub Size needs to be a valid Image size, e.g 10 x 10 = 100'

    (bs, ch, w, h) = embedding_tensor.size()
    (new_w, new_h) = (int(np.sqrt(sub_size)), int(np.sqrt(sub_size)))

    while True:
        ind = torch.unique(torch.randint(0, w * h, (sub_size * 2,)))
        if ind.size(0) > sub_size:
            ind = ind.clone()[:sub_size]
            break

    emb = embedding_tensor.view(bs, ch, -1)[:, :, ind].view(bs, ch, new_w, new_h)
    lab = label_tensor.view(bs, -1)[:, ind].view(bs, new_w, new_h)

    return emb, lab


def test(model_name=c.tb['pre_train_name'], corr_path='data/test_corr/starmy/maxpool/transformed_4',
         corr_sum_folder='data/test_corr_sum_img/',
         th=c.val['th_nn'], bp=c.UNet['background_pred']):
    dtype = c.data['dtype']
    device = c.cuda['device']
    model = n.UNetMS(background_pred=bp)

    results = []
    result_dict = {'dataset': None, 'regions': None}
    namelist = ['00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test', '02.01.test', '03.00.test',
                '04.00.test', '04.01.test']

    model.to(device)
    model.type(dtype)
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))
    test_dataset = data.TestCombinedDataset(corr_path=corr_path, corr_sum_folder=corr_sum_folder, device=device,
                                            dtype=dtype)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
    print('Test loader prepared.')

    model.eval()
    model.MS.val = True

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            input = batch['image']
            input_var = list()
            for j in range(len(input)):
                input_var.append(torch.autograd.Variable(input[j]))

            # compute output
            output, _, __ = model(input, None)

            (bs, ch, w, h) = output.size()
            predict = cl.label_embeddings(output.view(ch, -1).t(), th=th)
            predict = predict.reshape(bs, w, h)
            if c.test['show_img']:
                for b in range(bs):
                    plt.imshow(predict[b])
                    plt.title('Predicted Background (upper) vs Ground Truth (lower)')
                    plt.show()
            for k in range(bs):
                mask_predict = data.toCoords(predict[k])
                result_dict['dataset'] = namelist[k]
                result_dict['regions'] = mask_predict
                results.append(result_dict)

        if not os.path.exists('data/test_results'):
            os.makedirs('data/test_results')

        with open('data/test_results/' + str(model_name) + '.json', 'w') as f:
            f.write(json.dumps(results))

        model.MS.val = False

    pass


def val_score(iter=10, th=c.val['th_nn'], model_name=c.tb['pre_train_name'], background_pred=c.UNet['background_pred']):
    """
    Method to quantify the overall performance of a model
    :param iter: the more iterations the more accurate
    :param th: threshold of the clustering method
    :param model_name: model that should be evaluated
    :background_pred: wether model uses background pred
    :return: Overall Score of the current model
    """
    dtype = c.data['dtype']
    device = c.cuda['device']
    model = n.UNetMS(background_pred=background_pred)

    model.to(device)
    model.type(dtype)
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))
    val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/transformed_4/',
                                       transform=None, device=device, dtype=dtype, test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)
    print('Validation loader prepared.')

    set = h.Setup(th_nn=th)
    f1_ = []
    for i in range(iter):
        f1_metric = set.validate(val_loader, model, model_name=model_name)
        f1_.append(f1_metric)
    ret = sum(f1_)/f1_.__len__()
    print('Average Val Score of model ' + str(model_name) + ':\t' + str(ret))
    return ret


def test_th(np_arange=(0.005, 2.05, 0.005), model_name=c.tb['pre_train_name'], iter=10):
    dtype = c.data['dtype']
    device = c.cuda['device']
    model = n.UNetMS(background_pred=c.UNet['background_pred'])

    model.to(device)
    model.type(dtype)
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))
    val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/transformed_4/',
                                       transform=None, device=device, dtype=dtype, test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)
    print('Validation loader prepared.')

    list = np.arange(np_arange[0], np_arange[1], np_arange[2])
    f1_list = []
    f1_ind_list = []
    for th in list:
        f1_ = []
        set = h.Setup(th_nn=th)
        print('Threshold = ' + str(th))
        for i in range(iter):
            f1_metric = set.validate(val_loader, model)
            f1_.append(f1_metric)
        f1_list.append(sum(f1_)/f1_.__len__())
        f1_ind_list.append(th)
    print('Best Possible Th:\t', np.array(f1_ind_list)[np.argmax(np.array(f1_list))], max(f1_list))
    return np.array(f1_list), np.array(f1_ind_list)