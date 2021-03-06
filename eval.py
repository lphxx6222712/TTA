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

# from cam import CAM, GradCAM, GradCAMPlus

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)


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
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float???[0???1]??????????????????0-255
        img = img.astype(np.uint8)  # ??????unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # ??????heat map
        img = img[:, :, ::-1]  # ??????cv2???BGR??????matplotlib(RGB)??????????????????
        cv2.imwrite((savepath + '/' + '%s_%s_%s.jpg' % (iter, j, i)), img)
        print("{}/{}".format(i, width * height))

def eval(args, eval_data_loader, model, result_path, logger):

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
    target_layer = 'center'
    label2cat = {0: 'SRF', 1: 'PED'}
    use_gt_label = True

    for iter, (image, label, imt) in enumerate(eval_data_loader):
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)
        # if args.CAM != None:
        #     heatmap, cats = cam(image.unsqueeze(0), label, use_gt_label)
        # visualize_cam(imgdir, file_name, heatmap, save_dir, cats, is_split=is_split)
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


    eval(args, eval_loader, net, result_path, logger)


def parse_args():
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir', default='./data/dataset/')
    parser.add_argument('-l', '--list-dir', default='./data/data_path',
                        help='List dir to look for tra'
                             'in_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('--device', default='Spectralis', type=str,help='Topcon, '
                                                                    'Cirrus, Spectralis')
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('--name', help='seg model', default='unet', type=str)
    parser.add_argument('--number-classes', default=4, type=int)
    parser.add_argument('--seg-path', help='pretrained model test',
                        default='/home/isalab201/Documents/hxx/SCAN/result/train/unet_Topcon_nopre_NLL+dice+bce+intra_0.001_intra_DG_4.03v1/checkpoint/model_best.pth.tar',
                        type=str)
    parser.add_argument('--vis', default=False)
    parser.add_argument('--resize', default=[496,496], type=int, help='')
    parser.add_argument('--Transmodel', default='R50-ViT-B_16', type=str)
    parser.add_argument('--cam', default='CAM', help='CAM, GradCAM, GradCAMPLUS, None')

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

