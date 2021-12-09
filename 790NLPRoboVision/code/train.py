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
from model import MultiviewTransformer
from loss_evaluation.loss_evaluation import calc_loss
from tensorboardX import SummaryWriter
# import skimage
import time
import logging
from setup_logging import setup_logging

import matplotlib.pyplot as plt
import PIL

SAVE_ITER = 1

parser = argparse.ArgumentParser(description='DeepPruner')
parser.add_argument('--datapath_monkaa', default='/evo970/sceneflow/monkaa',
                    help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_flying', default='/evo970/sceneflow/flying',
                    help='datapath for sceneflow flying dataset')
parser.add_argument('--datapath_driving', default='/evo970/sceneflow/driving',
                    help='datapath for sceneflow driving dataset')
parser.add_argument('--epochs', type=int, default=15,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./finetune_14.tar',
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
    = dc.dataloader(filepath_monkaa=args.datapath_monkaa, filepath_driving=args.datapath_driving,
                    filepath_flying=args.datapath_flying)


batch_size = 1

TrainImgLoader = torch.utils.data.DataLoader(
    dl.SceneflowLoader(all_left_img, all_right_img, all_left_disp, all_left_cam, all_right_cam, 32, True),
    batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    dl.SceneflowLoader(test_left_img, test_right_img, test_left_disp, test_left_cam, test_right_cam, 32, False),
    batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

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

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


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
    result = model(imgL, imgR)

    loss = calc_loss(result, disp_true, mask)

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


def test(imgL, imgR, disp_L, iteration):

    model.eval()
    with torch.no_grad():
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        disp_L = Variable(torch.FloatTensor(disp_L))

        # visualize depth map
        # import matplotlib.pyplot as plt
        # plt.imshow(output_depth[0, :, :].cpu())
        # plt.show()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        mask = disp_true < args.maxdisp
        mask.detach_()

        if len(disp_true[mask]) == 0:
            logging.info("invalid GT disaprity...")
            return 0, 0

        result = model(imgL, imgR)

        loss = calc_loss(result, disp_true, mask)
        epe_loss = torch.mean(torch.abs(result[mask] - disp_true[mask]))

        # print(f'{epe_loss}')
        # plt.figure(f'err');
        # plt.imshow(result[0].detach().cpu(), vmin=disp_true.min(), vmax=disp_true.max())
        # plt.figure(f'gt');
        # plt.imshow(disp_true[0].detach().cpu(), vmin=disp_true.min(), vmax=disp_true.max())
        # plt.figure(f'diff');
        # plt.imshow((disp_true[0]-result[0]).abs().detach().cpu(), vmin=disp_true.min(), vmax=disp_true.max())
        # plt.figure(f'L R input');
        # plt.subplot(1,2,1); plt.imshow(imgL[0].permute(1,2,0).detach().cpu()/2+0.5)
        # plt.subplot(1,2,2); plt.imshow(imgR[0].permute(1,2,0).detach().cpu()/2+0.5)

    return loss.item(), epe_loss.item()
    # return 0., epe_loss.item()

# def adjust_learning_rate(optimizer, epoch):
#     if epoch <= 20:
#         lr = 0.001
#     elif epoch <= 40:
#         lr = 0.0007
#     elif epoch <= 60:
#         lr = 0.0003
#     else:
#         lr = 0.0001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def main():
    for epoch in range(1, args.epochs):
        total_train_loss = 0
        total_test_loss = 0
        total_epe_loss = 0
        # adjust_learning_rate(optimizer, epoch)

        if epoch % 1 == 0 and epoch != 0:
        # if epoch % 1 == 0:
            logging.info("testing...")
            # for batch_idx, (imgL, imgR, disp_L, pad_w, pad_h, left_cam, right_cam, imgL_fn, imgR_fn) in enumerate(TestImgLoader):
            for batch_idx, (imgL, imgR, disp_L, left_cam_crop, right_cam_crop, imgL_crop_fn, imgR_crop_fn) in enumerate(
                    TestImgLoader):
                start_time = time.time()
                test_loss, epe_loss = test(imgL, imgR, disp_L, batch_idx)
                total_test_loss += test_loss
                total_epe_loss += epe_loss

                logging.info('Iter %d 3-px error in val = %.3f, time = %.2f \n' %
                      (batch_idx, epe_loss, time.time() - start_time))

                writer.add_scalar("val-loss-iter", test_loss, epoch * 4370 + batch_idx)
                writer.add_scalar("val-epe-loss-iter", epe_loss, epoch * 4370 + batch_idx)
                # if batch_idx == 3:
                #     break

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

            # if batch_idx == 3:
            #     break

        logging.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        writer.add_scalar("loss", total_train_loss / len(TrainImgLoader), epoch)

        # SAVE
        if epoch % SAVE_ITER == 0:
            savefilename = args.savemodel + 'finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss,
                'test_loss': total_test_loss,
            }, savefilename)


if __name__ == '__main__':
    main()