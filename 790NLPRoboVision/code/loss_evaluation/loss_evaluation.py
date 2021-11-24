from __future__ import print_function
import torch.nn.functional as F
from collections import namedtuple
import logging
import math

import torch

from helpers import angle_between_vectors
from models.config_depth import config

import matplotlib.pyplot as plt

loss_configs = {
    'loss_margin': 1.,

    'gt_loss_weight': 1.,

    'depth_far_thresh': 0.,  # distance
    'far_loss_weight': 0.125,
    "far_target_loss": 5.,
    'depth_too_far_thresh': 5,  # distance
    'too_far_loss_weight': 1.25,
    "too_far_target_loss": 10.,

    "ambiguous_thresh": 2.,
    "ambiguous_penalty": 0.3,

    'normal_near_thresh': 5,  # degrees
}

loss_configs = namedtuple('loss_weights_depth', loss_configs.keys())(*loss_configs.values())


def loss_evaluation(result, disp_true, mask, cost_aggregator_scale=4):

    # forces min_disparity to be equal or slightly lower than the true disparity
    quantile_mask1 = ((disp_true[mask] - result[-1][mask]) < 0).float()
    quantile_loss1 = (disp_true[mask] - result[-1][mask]) * (0.05 - quantile_mask1)
    quantile_min_disparity_loss = quantile_loss1.mean()

    # forces max_disparity to be equal or slightly larger than the true disparity
    quantile_mask2 = ((disp_true[mask] - result[-2][mask]) < 0).float()
    quantile_loss2 = (disp_true[mask] - result[-2][mask]) * (0.95 - quantile_mask2)
    quantile_max_disparity_loss = quantile_loss2.mean()

    min_disparity_loss = F.smooth_l1_loss(result[-1][mask], disp_true[mask], size_average=True)
    max_disparity_loss = F.smooth_l1_loss(result[-2][mask], disp_true[mask], size_average=True)
    ca_depth_loss = F.smooth_l1_loss(result[-3][mask], disp_true[mask], size_average=True)
    refined_depth_loss = F.smooth_l1_loss(result[-4][mask], disp_true[mask], size_average=True)

    logging.info("============== evaluated losses ==================")
    if cost_aggregator_scale == 8:
        refined_depth_loss_1 = F.smooth_l1_loss(result[-5][mask], disp_true[mask], size_average=True)
        loss = (loss_weights.alpha_super_refined * refined_depth_loss_1)
        output_disparity = result[-5]
        logging.info('refined_depth_loss_1: %.6f', refined_depth_loss_1)
    else:
        loss = 0
        output_disparity = result[-4]

    loss += (loss_weights.alpha_refined * refined_depth_loss) + \
            (loss_weights.alpha_ca * ca_depth_loss) + \
            (loss_weights.alpha_quantile * (quantile_max_disparity_loss + quantile_min_disparity_loss)) + \
            (loss_weights.alpha_min_max * (min_disparity_loss + max_disparity_loss))

    logging.info('refined_disparity_loss: %.6f' % refined_depth_loss)
    logging.info('ca_disparity_loss: %.6f' % ca_depth_loss)
    logging.info('quantile_loss_max_disparity: %.6f' % quantile_max_disparity_loss)
    logging.info('quantile_loss_min_disparity: %.6f' % quantile_min_disparity_loss)
    logging.info('max_disparity_loss: %.6f' % max_disparity_loss)
    logging.info('min_disparity_loss: %.6f' % min_disparity_loss)
    logging.info("==================================================\n")

    return loss, output_disparity
