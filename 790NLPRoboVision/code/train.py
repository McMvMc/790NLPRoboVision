# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from dataloader import data_collector as dc
from dataloader import data_loader as dl
from models import MultiviewTransformer
from loss_evaluation import loss_evaluation, loss_evaluation_depth
from tensorboardX import SummaryWriter
import skimage
import time
import logging
from setup_logging import setup_logging

import matplotlib.pyplot as plt
import PIL


parser = argparse.ArgumentParser(description='DeepPruner')
parser.add_argument('--datapath_monkaa', default=None,
                    help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_flying', default=None,
                    help='datapath for sceneflow flying dataset')
parser.add_argument('--datapath_driving', default=None,
                    help='datapath for sceneflow driving dataset')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--save_dir', default='./',
                    help='save directory')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--logging_filename', default='./train_sceneflow.log',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--run_depth', type=bool, default=False,
                    help='run Shape PatchMatch that samples depth and normal')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


# Shape PatchMatch
args.maxdisp = 50
model = MultiviewTransformer()


setup_logging(args.logging_filename)

# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath_monkaa,
# args.datapath_flying, args.datapath_driving)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, \
    all_left_cam, all_right_cam, test_left_cam, test_right_cam,\
    = dc.dataloader(filepath_monkaa=args.datapath_monkaa, filepath_driving=args.datapath_driving)


batch_size = 2

TrainImgLoader = torch.utils.data.DataLoader(
    dl.SceneflowLoader(all_left_img, all_right_img, all_left_disp, all_left_cam, all_right_cam, args.cost_aggregator_scale*8.0, True),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    dl.SceneflowLoader(test_left_img, test_right_img, test_left_disp, test_left_cam, test_right_cam, args.cost_aggregator_scale*8.0, False),
    batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

# if args.run_depth:
#
# else:
#
writer = SummaryWriter()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()


if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=True)
    logging.info('model loaded from {}'.format(args.loadmodel))

logging.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def disp_to_depth_L(disp_L, right_cam, left_cam):
    depth_L = torch.zeros_like(disp_L)
    for i in range(disp_L.shape[0]):
        baseline = torch.norm(right_cam["pos"][i] - left_cam["pos"][i])
        depth_L[i] = left_cam["intrinsics"]["fx"][i] * baseline / disp_L[i]
    return depth_L


def depth_to_disp_L(depth_L, right_cam, left_cam):
    disp_L = torch.zeros_like(depth_L)
    for i in range(depth_L.shape[0]):
        baseline = torch.norm(right_cam["pos"][i] - left_cam["pos"][i])
        disp_L[i] = left_cam["intrinsics"]["fx"][i] * baseline / depth_L[i]
    return disp_L


def train(imgL, imgR, disp_L, left_cam, right_cam, imgL_crop_fn, imgR_crop_fn, iteration):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    mask = disp_true < args.maxdisp
    mask.detach_()

    optimizer.zero_grad()
    result = model(imgL, imgR, left_cam, right_cam, imgL_crop_fn, imgR_crop_fn)

    loss, _ = loss_evaluation(result, disp_true, mask, args.cost_aggregator_scale)

    loss.backward()
    optimizer.step()

    return loss.item()


def custom_test_helper():
    # orig image
    imgL_fn = '/home/mike/Desktop/research/DeepPruner/eval_images/depth_sampling/trained_w_depth/test_human/M1000_H_L.jpg'
    imgR_fn = '/home/mike/Desktop/research/DeepPruner/eval_images/depth_sampling/trained_w_depth/test_human/M1000_H_R.jpg'
    downsample = 4

    left_img = dl.default_loader(imgL_fn)
    right_img = dl.default_loader(imgR_fn)

    if downsample > 1:
        width, height = left_img.size
        left_img = left_img.resize((width/downsample, height/downsample), PIL.Image.LANCZOS)
        right_img = left_img.resize((width / downsample, height / downsample), PIL.Image.LANCZOS)

    # intrinsics
    fx = 1.680629915304855e+03/downsample
    fy = 1.680629915304855e+03/downsample
    cx = 800/downsample
    cy = 600/downsample

    R_l = torch.eye(3).unsqueeze(0)
    t_l = torch.zeros(1,3,1)
    ori_l = R_l.transpose(1,2)
    pos_l = t_l
    left_cam = {
        'intrinsics': {'fx': torch.tensor([fx]), 'fy': torch.tensor([fy]), 'cx': torch.tensor([cx]), 'cy': torch.tensor([cy])},
        'R': R_l, 't': t_l, 'ori': ori_l, 'pos': pos_l}

    R_r = torch.tensor([[0.999780557587180,  -0.007771339686595,  -0.019453610208482],
                        [0.007797753502200,   0.999968774889021,   0.001282297652614],
                        [0.019443037596707,  -0.001433710719257,   0.999809938319572]])
    t_r = torch.tensor([[-0.999048251840356],
                        [-0.041388820515615],
                        [0.013767934887067]])
    pos_r = torch.matmul(-R_r.t(), t_r).unsqueeze(0)
    ori_r = R_r.t().unsqueeze(0)
    R_r = R_r.unsqueeze(0)
    t_r = t_r.unsqueeze(0)
    right_cam = {
        'intrinsics': {'fx': torch.tensor([fx]), 'fy': torch.tensor([fy]), 'cx': torch.tensor([cx]), 'cy': torch.tensor([cy])},
        'R': R_r, 't': t_r, 'ori': ori_r, 'pos': pos_r}

    from dataloader import preprocess
    w, h = left_img.size
    downsample_scale = args.cost_aggregator_scale*8.0
    dw = w + (downsample_scale - (
                w % downsample_scale + (w % downsample_scale == 0) * downsample_scale))
    dh = h + (downsample_scale - (
                h % downsample_scale + (h % downsample_scale == 0) * downsample_scale))


    # if w-dw < 0, crop() will pad with black pixels
    pad_w = dw - w
    pad_h = dh - h
    left_img = left_img.crop((w - dw, h - dh, w, h))
    right_img = right_img.crop((w - dw, h - dh, w, h))
    left_cam['intrinsics']['cx'] += pad_w
    left_cam['intrinsics']['cy'] += pad_h
    right_cam['intrinsics']['cx'] += pad_w
    right_cam['intrinsics']['cy'] += pad_h


    processed = preprocess.get_transform()
    left_img = processed(left_img)
    right_img = processed(right_img)

    imgL = Variable(torch.FloatTensor(left_img)).unsqueeze(0)
    imgR = Variable(torch.FloatTensor(right_img)).unsqueeze(0)
    return left_img, right_img, left_cam, right_cam, imgL, imgR, imgL_fn, imgR_fn, int(pad_w), int(pad_h)


def test(imgL, imgR, disp_L, left_cam, right_cam, imgL_fn, imgR_fn, iteration, pad_w, pad_h):

    model.eval()
    with torch.no_grad():
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        disp_L = Variable(torch.FloatTensor(disp_L))

        # visualize depth map
        # import matplotlib.pyplot as plt
        # plt.imshow(output_depth[0, :, :].cpu())
        # plt.show()

        if args.run_depth:
            depth_L = disp_to_depth_L(disp_L, right_cam, left_cam)

            if args.cuda:
                imgL, imgR, depth_true = imgL.cuda(), imgR.cuda(), depth_L.cuda()

            mask = depth_true < args.maxdepth
            mask.detach_()

            if len(depth_true[mask]) == 0:
                logging.info("invalid GT disaprity...")
                return 0, 0

            optimizer.zero_grad()

            # test on ESRC 3D
            # left_img, right_img, left_cam, right_cam, imgL, imgR, imgL_fn, imgR_fn, pad_w, pad_h = custom_test_helper()

            result = model(imgL, imgR, left_cam, right_cam, imgL_fn, imgR_fn)

            # test on ESRC 3D
            # plt.imshow(result[-4][0, :, :].cpu());
            # plt.show()

            output = []
            for ind in range(len(result)):
                output.append(result[ind][:, pad_h:, pad_w:])
            result = output

            loss, output_depth = loss_evaluation_depth(result, depth_true, mask, args.cost_aggregator_scale)

            # visualize
            # output_disp = depth_to_disp_L(output_depth, right_cam, left_cam)
            # import matplotlib.pyplot as plt
            # import matplotlib.gridspec as gridspec
            # fig = plt.figure(figsize=(17.195, 13.841), dpi=100)
            # gs1 = gridspec.GridSpec(2, 2)
            # gs1.update(wspace=0.025, hspace=0)
            # ax = fig.add_subplot(gs1[0])
            # ax.imshow((output_depth * mask.float())[0, :, :].cpu(), vmin=0, vmax=50)
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # plt.subplots_adjust(wspace=0, hspace=0)
            # plt.axis('off')
            # ax2 = fig.add_subplot(gs1[1])
            # ax2.imshow((depth_L * mask.cpu().float())[0, :, :].cpu(), vmin=0, vmax=50)
            # ax2.set_xticklabels([])
            # ax2.set_yticklabels([])
            # plt.subplots_adjust(wspace=0, hspace=0)
            # plt.axis('off')
            # ax3 = fig.add_subplot(gs1[2])
            # ax3.imshow(torch.abs(output_depth[0, :, :].cpu() - depth_L[0, :, :].cpu()) * mask[0, :, :].float().cpu(),
            #            vmin=0,
            #            vmax=50)
            # ax3.set_xticklabels([])
            # ax3.set_yticklabels([])
            # plt.subplots_adjust(wspace=0, hspace=0)
            # plt.axis('off')
            # ax = [plt.subplot(b) for b in gs1]
            #
            # for a in ax:
            #     a.set_xticklabels([])
            #     a.set_yticklabels([])
            #
            # plt.subplots_adjust(wspace=0, hspace=0)
            # plt.show()

            epe_loss = torch.mean(torch.abs(output_depth[mask] - depth_true[mask]))
        else:
            if args.cuda:
                imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

            mask = disp_true < args.maxdisp
            mask.detach_()

            if len(disp_true[mask]) == 0:
                logging.info("invalid GT disaprity...")
                return 0, 0

            optimizer.zero_grad()
            result = model(imgL, imgR)

            output = []
            for ind in range(len(result)):
                output.append(result[ind][:, pad_h:, pad_w:])
            result = output

            loss, output_disparity = loss_evaluation(result, disp_true, mask, args.cost_aggregator_scale)
            epe_loss = torch.mean(torch.abs(output_disparity[mask] - disp_true[mask]))

    return loss.item(), epe_loss.item()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
        lr = 0.001
    elif epoch <= 40:
        lr = 0.0007
    elif epoch <= 60:
        lr = 0.0003
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    for epoch in range(99, args.epochs):
        total_train_loss = 0
        total_test_loss = 0
        total_epe_loss = 0
        adjust_learning_rate(optimizer, epoch)

        if epoch % 1 == 0 and epoch != 0:
        # if epoch % 1 == 0:
            logging.info("testing...")
            for batch_idx, (imgL, imgR, disp_L, pad_w, pad_h, left_cam, right_cam, imgL_fn, imgR_fn) in enumerate(TestImgLoader):
                start_time = time.time()
                test_loss, epe_loss = test(imgL, imgR, disp_L, left_cam, right_cam, imgL_fn, imgR_fn,
                                           batch_idx, int(pad_w[0].item()), int(pad_h[0].item()))
                total_test_loss += test_loss
                total_epe_loss += epe_loss

                logging.info('Iter %d 3-px error in val = %.3f, time = %.2f \n' %
                      (batch_idx, epe_loss, time.time() - start_time))

                writer.add_scalar("val-loss-iter", test_loss, epoch * 4370 + batch_idx)
                writer.add_scalar("val-epe-loss-iter", epe_loss, epoch * 4370 + batch_idx)

            logging.info('epoch %d total test loss = %.3f' % (epoch, total_test_loss / len(TestImgLoader)))
            writer.add_scalar("val-loss", total_test_loss / len(TestImgLoader), epoch)
            logging.info('epoch %d total epe loss = %.3f' % (epoch, total_epe_loss / len(TestImgLoader)))
            writer.add_scalar("epe-loss", total_epe_loss / len(TestImgLoader), epoch)

        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, left_cam_crop, right_cam_crop, imgL_crop_fn, imgR_crop_fn) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L, left_cam_crop, right_cam_crop, imgL_crop_fn, imgR_crop_fn, batch_idx)
            total_train_loss += loss

            writer.add_scalar("loss-iter", loss, batch_idx + 35454 * epoch)
            logging.info('Iter %d training loss = %.3f , time = %.2f \n' % (batch_idx, loss, time.time() - start_time))

        logging.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        writer.add_scalar("loss", total_train_loss / len(TrainImgLoader), epoch)

        # SAVE
        if epoch % 1 == 0:
            savefilename = args.savemodel + 'finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss,
                'test_loss': total_test_loss,
            }, savefilename)


if __name__ == '__main__':
    main()