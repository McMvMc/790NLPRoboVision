from __future__ import print_function
import torch.nn.functional as F
from collections import namedtuple
import logging
import math

import torch

from helpers import angle_between_vectors
from models.config_depth import config

import matplotlib.pyplot as plt

loss_weights = {
    'alpha_super_refined': 1.6,
    'alpha_refined': 1.3,
    'alpha_ca': 1.0,
    'alpha_quantile': 1.0,
    'alpha_min_max': 0.7
}

loss_weights = namedtuple('loss_weights', loss_weights.keys())(*loss_weights.values())


def calc_loss(result, disp_true, mask):
    disp_loss = F.smooth_l1_loss(result[mask], disp_true[mask], size_average=True)

    logging.info("============== evaluated losses ==================")

    logging.info('disp_loss: %.6f' % disp_loss)
    logging.info("==================================================\n")

    return disp_loss