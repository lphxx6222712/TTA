import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
##### pytorch library #####
import torch
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
##### My own library #####
import data.seg_transforms as st
from data.Seg_dataset import SegList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.utils import AverageMeter, aic_fundus_lesion_segmentation, compute_segment_score, \
    compute_single_segment_score, target_seg2target_cls, aic_fundus_lesion_classification
from utils.vis import vis_multi_class, vis_multi_class_w_entropy
from Instances import Instances



FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum(1).sum(1).sum(1).mean(0)
    return E

def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):

    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * torch.einsum('bchw,bkhw->bkhw',kernel,Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            print(f'Converged in {i} iterations')
            break
        else:
            oldE = E

    return Y

class kNN_affinity():
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N,C,W,H = X.shape
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=2, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

         # [N, knn]
        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, :, :, 1:].topk(n_neighbors, -2,
                                                                                        largest=False).indices[:, :, 1:,
                    :]
        W = torch.zeros([N,N,W,H], device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0).scatter_(dim=-1, index=knn_index, value=1.0)

        return W

def format_result(self, batched_inputs, probas):
    image_sizes = [x.shape for x in batched_inputs]
    results = probas
    gt = torch.tensor([obj["instances"].gt_classes[0] for obj in batched_inputs]).to(self.device)
    results['gts'] = gt
    results['one_hot_gts'] = F.one_hot(gt, self.num_classes)
    results["instances"] = []
    with torch.no_grad():
        for i, image_size in enumerate(image_sizes):
            res = Instances(image_size)
            # print(list(batched_inputs[i].keys()))
            gt_instances = batched_inputs[i]["instances"]
            res.gt_classes = gt_instances.gt_classes
            n_instances = len(gt_instances.gt_classes)
            res.pred_classes = probas[i].argmax(-1, keepdim=True).repeat(n_instances)
            res.scores = probas[i].max(-1, keepdim=True).values.repeat(n_instances)
            results["instances"].append(res)
    return results
###### eval ########

def SelfEntropy_per_pixel(probs):
    p = torch.nn.Softmax(dim=1)(probs)
    log_p = (p + 1e-10).log()
    mask = probs.type((torch.float32))
    # weights = [1,1,1,1]
    # mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, torch.Tensor(weights).to(mask.device)])
    self_entropy = torch.mul(-p,log_p)
    #loss = - torch.einsum("bcwh,bcwh->", [p, log_p])
    #loss /= mask.sum() + 1e-10

    # import matplotlib.pyplot as plt
    # plt.imshow(probs[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
    # plt.imshow(target[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())

    return self_entropy

def draw_features(x, savename, iter, j):
    width, height, channel = x.shape[2], x.shape[3], x.shape[1]
    savepath = os.path.join(
        '/home/hxx/Documents/hxx_code/pytorch/Weakly-supervised-OCT-segmentation/result/eval', savename)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    x = x.cpu().numpy()
    for i in range(channel):
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        cv2.imwrite((savepath + '/' + '%s_%s_%s.jpg' % (iter, j, i)), img)
        print("{}/{}".format(i, width * height))



def TTAeval(args, eval_data_loader, model, optimizer, result_path, logger):

    #eval before adaptation

    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    # dice_1 = green = irf, dice_2 = blue = SRF, dice_3 = red = PED
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    ret_segmentation = []


    for iter, (image, label, imt) in enumerate(eval_data_loader):
        model.train()
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)

        # batch test for memory reduce
        batch = 4
        pred_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
        pred_cls = torch.zeros(image.shape[0], 3)
        entropy_seg = torch.zeros(image.shape[0], image.shape[1]+1, image.shape[2], image.shape[3])
        for i in range(0, image.shape[0], batch):
            start_id = i
            end_id = i + batch
            if end_id > image.shape[0]:
                end_id = image.shape[0]
            image_batch = image[start_id:end_id, :, :, :]
            image_var = Variable(image_batch).cuda()

            output_seg, output_cls, center, output_feats = model(image_var)
            ### intra_slice_loss
            #
            back_region = image_var[:, :, 496 - 70:496 - 30, :]
            background = torch.zeros_like(image_var)
            yushu = image_var.shape[2] % 40
            background[:, :, :yushu, :] = back_region[:, :, :yushu, :]
            for k in range(int(496 / 40)):
                background[:, :, yushu + k * 40:yushu + (k + 1) * 40, :] = back_region

            region_filter = torch.sum(F.softmax(1000 * output_seg, dim=1)[:, 1:, :, :], dim=1)
            seg_region_0 = torch.mul(region_filter, image_var[:, 0, :, :]).unsqueeze(1)
            seg_region_1 = torch.mul(region_filter, image_var[:, 1, :, :]).unsqueeze(1)
            seg_region_2 = torch.mul(region_filter, image_var[:, 2, :, :]).unsqueeze(1)
            seg_region = torch.cat([seg_region_0, seg_region_1, seg_region_2], dim=1)
            loss_back = torch.nn.KLDivLoss(reduction='mean').cuda()
            loss_background = loss_back(torch.log_softmax(seg_region,dim=1), torch.softmax(background,dim=1).cuda())
            optimizer.zero_grad()
            loss_background.backward()
            optimizer.step()

            #Tent
            # self_entropy_loss = SelfEntropy_per_pixel(output_seg).sum(1).sum(1).sum(1).mean(0)
            #
            #
            # loss = loss_background+self_entropy_loss
            # optimizer.zero_grad()
            # loss.backward()
            # #self_entropy_loss.backward()
            # optimizer.step()

            #SHOT
            # softmax_out = F.softmax(output_seg,dim=1)
            # msoftmax = softmax_out.mean(0)
            # cond_ent = - (softmax_out * torch.log(softmax_out + 1e-10)).sum(1).sum(1).sum(1).mean(0)
            # ent = - (msoftmax * torch.log(msoftmax + 1e-10)).sum(0).sum(0).sum(0)
            # classifier_loss = cond_ent - ent
            # optimizer.zero_grad()
            # classifier_loss.backward()
            # optimizer.step()

            #cotta, Continual Test-Time Domain Adaptation, CVPR 2022

            #DUA, The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization



            n_instances = 4
            #pred_classes = Y.argmax(-1, keepdim=True).repeat(n_instances)


            #evaluation
            with torch.no_grad():
                output_seg, output_cls, center, output_feats = model(image_var)
                self_entropy = SelfEntropy_per_pixel(output_seg)
                # draw_features(output_pro,'probility_map',iter,i )


                # LAME
                probas = F.softmax(output_seg, dim=1)  # [N, K]

                # --- Get unary and terms and kernel ---
                unary = - torch.log(probas + 1e-10)  # [N, K]

                feats = output_feats  # [N, d]
                feats = F.normalize(feats, p=2, dim=-1)  # [N, d]
                affinity = kNN_affinity(4)
                kernel = affinity(feats)  # [N, N]

                # kernel = 1 / 2 * (kernel + kernel.t())

                # --- Perform optim ---

                Y = laplacian_optimization(unary, kernel)
                output_seg = Y

                _, pred_batch = torch.max(output_seg, 1)

            pred_seg[start_id:end_id, :, :] = pred_batch.cpu().data
            pred_cls[start_id:end_id, :] = output_cls.cpu().data
            entropy_seg[start_id:end_id,:,:,:] = self_entropy.cpu().data

        pred_seg = pred_seg.numpy().astype('uint8')
        entropy_seg = entropy_seg.numpy()

        if args.vis:
            imt = (imt.squeeze().numpy()).astype('uint8')
            ant = label.numpy().astype('uint8')
            model_name = args.seg_path.split('/')[-3]
            save_dir = osp.join(result_path, 'vis', '%04d' % iter)
            if not exists(save_dir): os.makedirs(save_dir)
            #vis_multi_class(imt, ant, pred_seg, save_dir)
            vis_multi_class_w_entropy(imt,ant,pred_seg,entropy_seg,save_dir)
            print('save vis, finished!')

        batch_time.update(time.time() - end)
        label_seg = label.numpy().astype('uint8')

        pred_seg = pred_seg.astype('uint8')
        # pred_seg = pred_seg.numpy().astype('uint8')
        ret = aic_fundus_lesion_segmentation(label_seg, pred_seg)
        ret_segmentation.append(ret)
        dice_score = compute_single_segment_score(ret)
        dice_list.append(dice_score)
        dice.update(dice_score)
        Dice_1.update(ret[1])
        Dice_2.update(ret[2])
        Dice_3.update(ret[3])

        ground_truth = target_cls.numpy().astype('float32')
        prediction = pred_cls.numpy().astype('float32')  # predict label

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

    final_seg, seg_1, seg_2, seg_3 = compute_segment_score(ret_segmentation,len(eval_data_loader))
    print('### TTA Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))

    ret_detection = aic_fundus_lesion_classification(detection_ref_all, detection_pre_all,
                                                     num_samples=len(eval_data_loader) * 128)
    auc = np.array(ret_detection).mean()
    print('AUC :', auc)
    auc_1 = ret_detection[0]
    auc_2 = ret_detection[1]
    auc_3 = ret_detection[2]

    epoch = 0
    logger.append([epoch, final_seg, seg_1, seg_2, seg_3, auc, auc_1, auc_2, auc_3])  # ,auc_3])

def eval(args, eval_data_loader, model, result_path, logger):

    #eval before adaptation
    model.eval()
    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    # dice_1 = green = irf, dice_2 = blue = SRF, dice_3 = red = PED
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    ret_segmentation = []


    for iter, (image, label, imt) in enumerate(eval_data_loader):
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)
        with torch.no_grad():
            # batch test for memory reduce
            batch = 2
            pred_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
            pred_cls = torch.zeros(image.shape[0], 3)
            entropy_seg = torch.zeros(image.shape[0], image.shape[1]+1, image.shape[2], image.shape[3])
            for i in range(0, image.shape[0], batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]
                image_batch = image[start_id:end_id, :, :, :]
                image_var = Variable(image_batch).cuda()
                # wangshen model forward
                output_seg, output_cls, _,_ = model(image_var)
                #output_seg = model(image_var)
                self_entropy = SelfEntropy_per_pixel(output_seg)
                # draw_features(output_pro,'probility_map',iter,i )
                _, pred_batch = torch.max(output_seg, 1)

                # expand_out = my_DSRG(output_seg, target_cls[start_id:end_id], pred_batch)

                pred_seg[start_id:end_id, :, :] = pred_batch.cpu().data
                pred_cls[start_id:end_id, :] = output_cls.cpu().data
                entropy_seg[start_id:end_id,:,:,:] = self_entropy.cpu().data

            pred_seg = pred_seg.numpy().astype('uint8')
            entropy_seg = entropy_seg.numpy()

            if args.vis:
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                model_name = args.seg_path.split('/')[-3]
                save_dir = osp.join(result_path, 'vis', '%04d' % iter)
                if not exists(save_dir): os.makedirs(save_dir)
                #vis_multi_class(imt, ant, pred_seg, save_dir)
                vis_multi_class_w_entropy(imt,ant,pred_seg,entropy_seg,save_dir)
                print('save vis, finished!')

            batch_time.update(time.time() - end)
            label_seg = label.numpy().astype('uint8')

            pred_seg = pred_seg.astype('uint8')
            # pred_seg = pred_seg.numpy().astype('uint8')
            ret = aic_fundus_lesion_segmentation(label_seg, pred_seg)
            ret_segmentation.append(ret)
            dice_score = compute_single_segment_score(ret)
            dice_list.append(dice_score)
            dice.update(dice_score)
            Dice_1.update(ret[1])
            Dice_2.update(ret[2])
            Dice_3.update(ret[3])

            ground_truth = target_cls.numpy().astype('float32')
            prediction = pred_cls.numpy().astype('float32')  # predict label

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

    final_seg, seg_1, seg_2, seg_3 = compute_segment_score(ret_segmentation,len(eval_data_loader))
    print('### Vanilla Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))

    ret_detection = aic_fundus_lesion_classification(detection_ref_all, detection_pre_all,
                                                     num_samples=len(eval_data_loader) * 128)
    auc = np.array(ret_detection).mean()
    print('AUC :', auc)
    auc_1 = ret_detection[0]
    auc_2 = ret_detection[1]
    auc_3 = ret_detection[2]

    epoch = 0
    logger.append([epoch, final_seg, seg_1, seg_2, seg_3, auc, auc_1, auc_2, auc_3])  # ,auc_3])

def eval_fusion(args, eval_data_loader, model_list, result_path, logger):
    for model in model_list:
        model.eval()

    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    ret_segmentation = []

    for iter, (image, label, imt) in enumerate(eval_data_loader):
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)

        with torch.no_grad():
            # batch test for memory reduce
            batch = 8
            pred_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
            pred_cls = torch.zeros(image.shape[0], 2)
            for i in range(0, image.shape[0], batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]

                image_batch = image[start_id:end_id, :, :, :]
                image_var = Variable(image_batch).cuda()

                Output_Seg = Variable(torch.zeros(batch, 4, image.shape[2], image.shape[3])).cuda()
                Output_Cls = Variable(torch.zeros(batch, 128)).cuda()
                # wangshen model forward
                weight = torch.tensor([0.5, 0.5]).cuda()
                for j, model in enumerate(model_list):
                    output_seg, output_cls, seg_logit = model(image_var)
                    Output_Seg += weight[j] * torch.exp(output_seg)
                    Output_Cls += weight[j] * output_cls

                _, pred_batch = torch.max(Output_Seg, 1)
                pred_seg[start_id:end_id, :, :] = pred_batch.cpu().data
                pred_cls[start_id:end_id, :] = Output_Cls.cpu().data

            pred_seg = pred_seg.numpy().astype('uint8')  # predict label

            if args.vis:
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                save_dir = osp.join(result_path, 'vis', '%04d' % iter)
                if not exists(save_dir):
                    os.makedirs(save_dir)
                vis_multi_class(imt, ant, pred_seg, save_dir)
                print('save vis, finished!')

            batch_time.update(time.time() - end)
            # metrice dice for seg
            label_seg = label.numpy().astype('uint8')
            ret = aic_fundus_lesion_segmentation(label_seg, pred_seg)
            ret_segmentation.append(ret)
            dice_score = compute_single_segment_score(ret)
            dice_list.append(dice_score)
            # update dice
            dice.update(dice_score)
            Dice_1.update(ret[1])
            Dice_2.update(ret[2])
            Dice_3.update(ret[3])
            # metrice auc for cls
            ground_truth = target_cls.numpy().astype('float32')
            prediction = pred_cls.numpy().astype('float32')  # predict label

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

    final_seg, seg_1, seg_2, seg_3 = compute_segment_score(ret_segmentation,len(eval_data_loader))
    print('### Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))

    ret_detection = aic_fundus_lesion_classification(detection_ref_all, detection_pre_all,
                                                     num_samples=len(eval_data_loader) * 128)
    auc = np.array(ret_detection).mean()
    print('AUC :', auc)
    auc_1 = ret_detection[0]
    auc_2 = ret_detection[1]
    auc_3 = ret_detection[2]

    epoch = 0
    logger.append([epoch, final_seg, seg_1, seg_2, seg_3, auc, auc_1, auc_2, auc_3])  # ,auc_3])


def eval_seg(args, result_path, logger):


    info = json.load(open(osp.join(args.list_dir, 'info.json'), 'r'))
    normalize = st.Normalize(mean=info['mean'], std=info['std'])

    t = []
    if args.resize:
        t.append(st.Resize(args.resize))
    t.extend([st.Label_Transform(), st.ToTensor()])#, normalize])
    dataset = SegList(args.data_dir, 'test', st.Compose(t), list_dir=args.list_dir, device=args.device,hist_equal=False)
    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

    cudnn.benchmark = True

    print('Loading eval model ...')
    net = net_builder(args).cuda()
    #net = nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.seg_path)
    net.load_state_dict(checkpoint['state_dict'])
    print('model loaded!')

    if args.TTA:
        #eval(args, eval_loader, net, result_path, logger)

        for child in net.final_1.children():
            for p in child.parameters():
                p.requires_grad = False
        optimizer = torch.optim.Adam(net.parameters(),
                                     args.lr,
                                     betas=(0.9, 0.99),
                                     weight_decay=args.weight_decay)
        TTAeval(args, eval_loader, net, optimizer, result_path, logger)
    else:
        eval(args, eval_loader, net, result_path, logger)


def parse_args():
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir', default='./data/dataset/')
    parser.add_argument('-l', '--list-dir', default='./data/data_path',
                        help='List dir to look for tra'
                             'in_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('--device', default='Topcon', type=str,help='Topcon, '
                                                                    'Cirrus, Spectralis')
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('--name', help='seg model', default='unet', type=str)
    parser.add_argument('--number-classes', default=4, type=int)
    parser.add_argument('--seg-path', help='pretrained model test',
                        default='/home/isalab201/Documents/hxx/SCAN/result/train/unet_Topcon_nopre_NLL+dice+bce+intra_0.001_intra_DG_4.02v1/checkpoint/model_best.pth.tar',
                        type=str)
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--vis', default=False)
    parser.add_argument('--resize', default=[496,496], type=int, help='')
    parser.add_argument('--Transmodel', default='R50-ViT-B_16', type=str)
    parser.add_argument('--cam', default='CAM', help='CAM, GradCAM, GradCAMPLUS, None')
    parser.add_argument('--TTA', default=True, help='perform TTA or not')
    parser.add_gment('--TTA_Method', default='Intra_slice', help='Intra_slice, TENT, SHOT')
    parser.add_argument('--Output_refine_method', default='LAME')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    task_name = args.list_dir.split('/')[-1]
    model_name = args.seg_path.split('/')[-3] if len(args.seg_path) > 2 else 'fusion'
    result_path = osp.join('result', 'eval', model_name+'_to_'+args.device)

    if not exists(result_path):
        os.makedirs(result_path)
    logger = Logger(osp.join(result_path, 'dice_epoch.txt'), title='dice', resume=False)
    logger.set_names(['Epoch', 'Dice_val', 'Dice_1', 'Dice_2','Dice_3', 'AUC', 'AUC_1', 'AUC_2', 'AUC_3'])

    eval_seg(args, result_path, logger)


if __name__ == '__main__':
    main()

