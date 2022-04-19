#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:ShawnWang
##### System library #####
import os
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
import shutil
##### pytorch library #####
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
##### My own library #####
import data.seg_transforms as dt
from data.Seg_dataset import SegList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.loss import loss_builder
from utils.utils import compute_average_dice, AverageMeter, save_checkpoint, count_param, target_seg2target_cls
from utils.utils import aic_fundus_lesion_segmentation, aic_fundus_lesion_classification, compute_segment_score, \
    compute_single_segment_score
from utils import loss


# logger vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

densecrflosslayer = loss.DenseCRFLoss(weight=0., sigma_rgb=100.0, sigma_xy=15.0, scale_factor=0.5)


###### train #######

def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0, 2, 3, 1))
    images *= std
    images += mean
    images *= 255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0, 3, 1, 2))
    return torch.tensor(images)


def softargmax2d(input, beta=1000):
    B, N, h, w = input.shape

    input = torch.nn.functional.softmax(beta * input, dim=1)
    # print(input[0,:,0,0])

    indices_c = (N - 1) * torch.linspace(0, 1, N).cuda()

    result1 = input[:, 0, :, :] * indices_c[0]
    result2 = input[:, 1, :, :] * indices_c[1]
    result3 = input[:, 2, :, :] * indices_c[2]
    result = torch.stack([result1, result2, result3], dim=1)

    # print(result[0, :, 0,0])
    result_c = torch.sum(result, dim=1)
    # print(result_c[0, 0, 0 ])

    return result_c

def Information_maximization_loss(output):
    entropy_loss = self_entropy_loss(output)

    softmax_out = torch.nn.Softmax(dim=1)(output)
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = -msoftmax * torch.log(msoftmax + 1e-5)
    gentropy_loss = torch.sum(gentropy_loss, dim=0)
    gentropy_loss = gentropy_loss.mean()

    im_loss = entropy_loss+gentropy_loss

    return im_loss

def SelfEntropy(probs):
        p = torch.nn.Softmax(dim=1)(probs)
        log_p = (p + 1e-10).log()
        mask = probs.type((torch.float32))
        #weights = [1,1,1,1]
        #mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, torch.Tensor(weights).to(mask.device)])
        entropy_map = torch.mul(-p,log_p)
        loss = - torch.einsum("bcwh,bcwh->", [p, log_p])
        loss /= mask.sum() + 1e-10

        #import matplotlib.pyplot as plt
        #plt.imshow(probs[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
        #plt.imshow(target[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())

        return loss, entropy_map

def self_entropy_loss(output, weights=[1,1,1]):
    p = torch.nn.Softmax(dim=1)(output)
    entropy_loss = loss.Entropy(p)
    entropy_loss = entropy_loss.mean()
    return entropy_loss


def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    x = -torch.mul(prob,torch.log2(prob+1e-30))
    x = x/np.log2(c)
    return x

def Discriminator_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def train(args, train_source_loader, train_target_loader, source_model, model, layer_head, discriminator, criterion, optimizer, optimizer_d, epoch, print_freq=10):
    #writer = SummaryWriter()
    # set the AverageMeter
    batch_time = AverageMeter()
    losses_c = AverageMeter()
    losses_d = AverageMeter()
    losses_g = AverageMeter()
    dice = AverageMeter()
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    source_dice = AverageMeter()
    source_Dice_1 = AverageMeter()
    source_Dice_2 = AverageMeter()
    source_Dice_3 = AverageMeter()
    # switch to train mode

    end = time.time()
    correct, total = 0, 0
    layer_final = layer_head[0]
    layer_final.train()
    optimizer_lay = layer_head[1]

    Discriminator = discriminator
    source_model.eval()
    model.train()

    optimizer_d = optimizer_d


    for i,data in enumerate(zip(train_source_loader,train_target_loader)):
        optimizer.zero_grad()

        source_input, source_label, source_layer = data[0]
        target_input, target_label, target_layer = data[1]
        source_cls_label = target_seg2target_cls(source_label.cpu().data.numpy()).cuda()

        source_input_var = source_input.cuda()
        source_label_var = source_label.cuda()
        source_layer_var = source_layer.cuda()
        target_input_var = target_input.cuda()
        target_label_var = target_label.cuda()
        target_layer_var = target_layer.cuda()

        predict_seg, predict_cls, _, _ = source_model(target_input_var)
        fake_label = torch.max(predict_seg,dim=1)[1].long()
        fake_cls = target_seg2target_cls(fake_label.cpu().data.numpy()).cuda()

        source_seg = source_label.numpy()
        target_seg = target_label.numpy()
          # transfer seg target to cls target
        # Variable




        # forward
        source_output_seg, source_output_cls, source_cls_logit, source_seg_logit = model(source_input_var)
        target_output_seg, target_output_cls, target_cls_logit, target_seg_logit = model(target_input_var)

        source_output_layer, source_logit_layer = layer_final(source_seg_logit)
        target_output_layer, target_logit_layer = layer_final(target_seg_logit)

        for param in Discriminator.parameters():
            param.requires_grad = False
        Discriminator.eval()


        ce_loss = criterion[0](source_output_seg, source_label_var) + criterion[0](target_output_seg, fake_label)
        bce_loss = criterion[2](source_output_cls,source_cls_label) + criterion[2](target_output_cls, fake_cls)
        dice_loss = criterion[1](source_output_seg, source_label_var) + criterion[1](target_output_seg,fake_label)
        sim_lay = torch.nn.CosineSimilarity()(source_output_layer.detach(), target_output_layer.detach())

        lay_loss = criterion[4](source_output_layer, source_layer_var) + criterion[4](target_output_layer,
                                                                                      target_layer_var)

        loss_c = ce_loss + bce_loss + dice_loss# + lay_loss  # + loss_background +

        loss_c.backward() #source supervised loss
        optimizer.step()

        loss_g = torch.tensor(0)
        loss_d = torch.tensor(0)

        losses_c.update(loss_c.cpu().data.numpy(), target_input_var.size(0))
        losses_g.update(loss_g.cpu().data.numpy(), target_input_var.size(0))
        losses_d.update(loss_d.cpu().data.numpy(), target_input_var.size(0))

        # metric dice for seg
        _, target_pred_seg = torch.max(target_output_seg, 1)
        target_pred_seg = target_pred_seg.cpu().data.numpy()
        target_label_seg = target_label_var.cpu().data.numpy()
        target_dice_score, target_dice_1, target_dice_2, target_dice_3 = compute_average_dice(target_pred_seg.flatten(), target_label_seg.flatten())

        _, source_pred_seg = torch.max(source_output_seg, 1)
        source_pred_seg = source_pred_seg.cpu().data.numpy()
        source_label_seg = source_label_var.cpu().data.numpy()
        source_dice_score, source_dice_1, source_dice_2, source_dice_3 = compute_average_dice(source_pred_seg.flatten(),
                                                                                              source_label_seg.flatten())

        dice.update(target_dice_score)
        Dice_1.update(target_dice_1)
        Dice_2.update(target_dice_2)
        Dice_3.update(target_dice_3)

        source_dice.update(source_dice_score)
        source_Dice_1.update(source_dice_1)
        source_Dice_2.update(source_dice_2)
        source_Dice_3.update(source_dice_3)
        # backwards
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger vis
        if i % print_freq == 0:
            logger_vis.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
                            'Dice_1 {dice_1.val:.6f} ({dice_1.avg:.4f})\t'
                            'Dice_2 {dice_2.val:.6f} ({dice_2.avg:.4f})\t'
                            'Dice_3 {dice_3.val:.6f} ({dice_3.avg:.4f})\t'.format(
                epoch, i, len(train_target_loader), batch_time=batch_time, dice=dice, dice_1=Dice_1, dice_2=Dice_2, dice_3=Dice_3))
            print('loss_c: %.3f (%.4f)' %(losses_c.val,losses_c.avg))
            print('loss_g: %.3f (%.4f)' % (losses_g.val, losses_g.avg))
            print('loss_d: %.3f (%.4f)' % (losses_d.val, losses_d.avg))


            logger_vis.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'source_Dice {source_dice.val:.4f} ({source_dice.avg:.4f})\t'
                            'source_Dice_1 {source_dice_1.val:.6f} ({source_dice_1.avg:.4f})\t'
                            'source_Dice_2 {source_dice_2.val:.6f} ({source_dice_2.avg:.4f})\t'
                            'source_Dice_3 {source_dice_3.val:.6f} ({source_dice_3.avg:.4f})\t'.format(
                epoch, i, len(train_target_loader), batch_time=batch_time, source_dice=source_dice, source_dice_1=source_Dice_1, source_dice_2=source_Dice_2, source_dice_3=source_Dice_3))
        # writer.add_scalar('Loss/c', losses_c.val)
        # writer.add_scalar('Loss/g', losses_g.val)
        # writer.add_scalar('Loss/d', losses_d.val)
        # writer.add_scalar('Loss/c_avg', losses_c.avg)
        # writer.add_scalar('Loss/g_avg', losses_g.avg)
        # writer.add_scalar('Loss/d_avg', losses_d.avg)
            #'Loss {loss.val:.6f} ({liss.avg:.4f})\t'
    return losses_c.avg, losses_g.avg, losses_d.avg, dice.avg, Dice_1.avg, Dice_2.avg, Dice_3.avg


def train_seg(args, result_path, logger):
    for k, v in args.__dict__.items():
        print(k, ':', v)
    # load the net

    # set the loss criterion
    criterion = loss_builder(args.loss)
    # Data loading code
    info = json.load(open(osp.join(args.list_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])
    # data transforms
    t = []
    if args.resize:
        t.append(dt.Resize(args.resize))
    if args.random_rotate > 0:
        t.append(dt.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(dt.RandomScale(args.random_scale))
    if args.crop_size:
        t.append(dt.RandomCrop(args.crop_size))
    t.extend([dt.Label_Transform(),
              dt.RandomHorizontalFlip(),
              dt.ToTensor()])
              #normalize])

    train_source_loader = torch.utils.data.DataLoader(SegList(args.data_dir, 'train', dt.Compose(t), list_dir=args.list_dir,
                                                              device=args.source,hist_equal=False), batch_size = args.batch_size,shuffle=True,
                                                      num_workers=args.workers, pin_memory=True,drop_last=True)
    train_target_loader = torch.utils.data.DataLoader(
        SegList(args.data_dir, 'train', dt.Compose(t), list_dir=args.list_dir, device = args.device,hist_equal=False), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    optim_param = []




    net = net_builder(args).cuda()
    source_net = net_builder(args).cuda()
    #model = torch.nn.DataParallel(net).cuda()
    layer_head = net_builder('layer_head').cuda()

    optimizer_lay = torch.optim.Adam(layer_head.parameters(),
                                     args.lr,weight_decay=args.weight_decay)

    layer_head = [layer_head, optimizer_lay]



    param = count_param(net)
    print('###################################')
    print('Model #%s# parameters: %.2f M' % (args.name, param / 1e6))

    # load the pretrained model
    if args.model_path:
        print("=> loading pretrained model '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['state_dict'])
        source_net.load_state_dict(checkpoint['state_dict'])

    #for i, child in enumerate(net.children()):
    #     if i >=9:
    #         print(child)
    #         for p in child.parameters():
    #             p.requires_grad=False

    for child in net.final_1.children():
          for p in child.parameters():
              p.requires_grad=False



    cudnn.benchmark = True
    best_dice = 0
    start_epoch = 0

    if args.optimizer == 'SGD':  # SGD optimizer
        optimizer = torch.optim.SGD(net.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':  # Adam optimizer
        optimizer = torch.optim.Adam(net.parameters(),#filter(lambda p: p.requires_grad, net.parameters()),
                                     # optimizer = torch.optim.Adam(net.parameters(),
                                     args.lr,
                                     weight_decay=args.weight_decay)

    Discriminator = get_fc_discriminator(num_classes=args.number_classes).cuda()
    optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    for epoch in range(start_epoch, args.epochs):
        args.lr = adjust_learning_rate(args, optimizer, epoch)
        logger_vis.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, args.lr))

        # train for one epoch
        # loss, dice_train, dice_1, dice_2 = train(args, train_loader, model, criterion, optimizer, epoch)
        loss_c, loss_g, loss_d, dice_train, dice_1, dice_2, dice_3 = \
            train(args, train_source_loader, train_target_loader, source_net, net, layer_head, Discriminator, criterion, optimizer, optimizer_d, epoch)
        # evaluate on validation set
        dice_val, dice_11, dice_22, dice_33, dice_list, auc, auc_1, auc_2, auc_3 = val_seg(args, net)
        # dice_val, dice_11, dice_22, dice_list, auc, auc_1, auc_2 = val_seg(args, model)
        # save best checkpoints
        is_best = dice_val > best_dice
        best_dice = max(dice_val, best_dice)
        checkpoint_dir = osp.join(result_path, 'checkpoint')
        if not exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest = checkpoint_dir + '/checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'dice_epoch': dice_val,
            'best_dice': best_dice,
        }, is_best, checkpoint_dir, filename=checkpoint_latest)
        if args.save_every_checkpoint:
            if (epoch + 1) % 1 == 0:
                history_path = checkpoint_dir + '/checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_latest, history_path)

        # logger.append([epoch, dice_train, dice_val, auc, dice_1, dice_11, dice_2, dice_22, auc_1, auc_2])
        logger.append([epoch, dice_train, dice_val, auc, dice_11, dice_22, dice_33, auc_1, auc_2, auc_3, loss_c, loss_g, loss_d])


####### validation ###########

def val(args, eval_data_loader, model):
    model.eval()

    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    ret_segmentation = []

    for iter, (image, label, _) in enumerate(eval_data_loader):
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)

        with torch.no_grad():
            # batch test for memory reduce
            batch = args.batch_size
            pred_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
            pred_cls = torch.zeros(image.shape[0], 3)
            for i in range(0, image.shape[0], batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]
                image_batch = image[start_id:end_id, :, :, :]
                image_var = Variable(image_batch).cuda()
                # model forward
                output_seg, output_cls, _,_ = model(image_var)
                _, pred_batch = torch.max(output_seg, 1)
                pred_seg[start_id:end_id, :, :] = pred_batch.cpu().data
                pred_cls[start_id:end_id, :] = output_cls.cpu().data
            # merice dice for seg
            pred_seg = pred_seg.numpy().astype('uint8')
            batch_time.update(time.time() - end)
            label_seg = label.numpy().astype('uint8')
            ret = aic_fundus_lesion_segmentation(label_seg, pred_seg)
            ret_segmentation.append(ret)
            dice_score = compute_single_segment_score(ret)
            dice_list.append(dice_score)
            dice.update(dice_score)
            Dice_1.update(ret[1])
            Dice_2.update(ret[2])
            Dice_3.update(ret[3])
            # metric auc for cls
            ground_truth = target_cls.numpy().astype('float32')
            prediction = pred_cls.numpy().astype('float32')
            if iter == 0:
                detection_ref_all = ground_truth
                detection_pre_all = prediction
            else:
                detection_ref_all = np.concatenate((detection_ref_all, ground_truth), axis=0)
                detection_pre_all = np.concatenate((detection_pre_all, prediction), axis=0)

        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                        'Dice {dice.val:.3f} ({dice.avg:.3f})\t'
                        'Dice_1 {dice_1.val:.3f} ({dice_1.avg:.3f})\t'
                        'Dice_2 {dice_2.val:.3f} ({dice_2.avg:.3f})\t'
                        'Dice_3 {dice_3.val:.3f} ({dice_3.avg:.3f})'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(iter, len(eval_data_loader), dice=dice, dice_1=Dice_1, dice_2=Dice_2, dice_3=Dice_3,
                                batch_time=batch_time))
        # break
    # compute average dice for seg
    final_seg, seg_1, seg_2, seg_3 = compute_segment_score(ret_segmentation, len(eval_data_loader))
    print('### Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))
    # compute average auc for cls
    ret_detection = aic_fundus_lesion_classification(detection_ref_all, detection_pre_all,
                                                     num_samples=len(eval_data_loader) * 128)
    auc = np.array(ret_detection).mean()
    print('AUC :', auc)
    auc_1 = ret_detection[0]
    auc_2 = ret_detection[1]
    auc_3 = ret_detection[2]

    return final_seg, seg_1, seg_2, seg_3, dice_list, auc, auc_1, auc_2, auc_3


def val_seg(args, model):
    info = json.load(open(osp.join(args.list_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])

    t = []
    if args.resize:
        t.append(dt.Resize(args.resize))
    if args.crop_size:
        t.append(dt.RandomCrop(args.crop_size))
    t.extend([dt.Label_Transform(),
              dt.ToTensor()])
              #normalize])

    dataset = SegList(args.data_dir, 'val', dt.Compose(t), list_dir=args.list_dir, device=args.device,hist_equal=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    cudnn.benchmark = True
    # dice_avg, dice_1, dice_2, dice_list, auc, auc_1, auc_2 = val(args, val_loader, model)
    dice_avg, dice_1, dice_2, dice_3, dice_list, auc, auc_1, auc_2, auc_3 = val(args, val_loader, model)

    return dice_avg, dice_1, dice_2, dice_3, dice_list, auc, auc_1, auc_2, auc_3


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-dir', default='./data/dataset/')
    parser.add_argument('--device', default='Spectralis', type=str, help='Topcon, Cirrus, Spectralis')
    parser.add_argument('-l', '--list-dir', default='./data/data_path/',
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('--name', dest='name', help='change model', default='unet', type=str)
    parser.add_argument('--number-classes', default=4, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-j', '--workers', type=int, default=0)
    # Train Setting
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1.0e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--loss', help='change model', default='ce+bce+dice', #self_entropy+clhpy+loss_backgroundce#entropy_pseudo_+[1, 20, 20, 20]+bce+hype+loss_backcls-+pseudo_seg_ce+pseudo_cls_bce_loss+pseudo_seg_ce+cls_bce
                        type=str)
    parser.add_argument('-o', '--optimizer', default='Adam', type=str)
    # Data Transform
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--resize', default=[496,496], type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    # Pretrain and Checkpoint
    parser.add_argument('-p', '--pretrained', type=bool)
    parser.add_argument('--model-path', default='/home/isalab102/Documents/hxx/OTSU_layer_UDA_RETOUCH/result/train/unet_Cirrus_to_Spectralis_SCAN_1.25v1/checkpoint/model_best.pth.tar', type=str)
    parser.add_argument('--save-every-checkpoint', action='store_true')
    parser.add_argument('--describe', default='Adv_self_training_3.21v4',#fixed_freeze_cls2final_1_freeze_cls2final_1_
                        help='additional information')  # 9.15v3_point_expanded_DSRG_th1=.96_th2=.99

    args = parser.parse_args()

    return args


def main():
    ##### config #####
    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('torch version:', torch.__version__)
    ##### logger setting #####
    task_name = args.list_dir.split('/')[-1]
    pretrained = 'pre' if args.pretrained else 'nopre'
    source = args.model_path.split('/')[-3]
    source = source.split('_')[1]
    assert source in ['Spectralis','Topcon', 'Cirrus']
    result_path = osp.join('result', task_name, 'train',
                           args.name + '_' + source + '_to_' + args.device + '_' + pretrained + '_' + args.loss + '_' + str(
                               args.lr) + '_' + args.describe)
    args.source = source
    if not exists(result_path):
        os.makedirs(result_path)
    resume = True if args.resume else False
    logger = Logger(osp.join(result_path, 'dice_epoch.txt'), title='dice', resume=resume)
    logger.set_names(
        ['Epoch', 'Dice_Train', 'Dice_Val', 'AUC', 'Dice_11', 'Dice_22', 'Dice_33', 'AUC_1', 'AUC_2', 'AUC_3', 'loss_c', 'loss_g', 'loss_d'])

    train_seg(args, result_path, logger)


if __name__ == '__main__':
    main()
