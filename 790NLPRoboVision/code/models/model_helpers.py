from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F


def stack_pyramid_feature(pyramid_feat_list, start_lvl, end_lvl, device):
    stacked_feat = torch.zeros_like(pyramid_feat_list[start_lvl], device=device)

    # for level in range(1,len(pyramid_feat_list)):
    for level in range(start_lvl, end_lvl+1):
        if level == start_lvl:
            stacked_feat = pyramid_feat_list[level]
        else:
            # TODO: compare "nearest" against "bilinear",
            #  and align_corner=True/False, but I don't really care about corners, so False (default) might be better
            if level == 1:
                stacked_feat = torch.cat((stacked_feat, pyramid_feat_list[level]), dim=1)
            else:
                upsampler = nn.Upsample(scale_factor=pow(2, level-max(start_lvl, 1)), mode='bilinear')
                upsampled_feat = upsampler(pyramid_feat_list[level])
                stacked_feat = torch.cat((stacked_feat, upsampled_feat), dim=1)

    return stacked_feat


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


def normalize_vector(vec, norm_dim, keepdim):
    vec = vec / vec.norm(dim=norm_dim, keepdim=keepdim)
    return vec


def spherical_coord_world_to_xyz_cam_coord(d, theta, phi):
    y_r = -(d * torch.cos(theta))  # spherical coord -z is y in left cam coord
    z_r = -(d * torch.sin(theta) * torch.cos(phi))  # norm on xy plane is d*sin(theta), spherical coord y is -z in left cam coord
    x_r = (d * torch.sin(theta)) * torch.sin(phi)
    return x_r, y_r, z_r


def angle_between_vectors(a, b):
    inner_product = (a * b).sum(dim=1)
    a_norm = a.pow(2).sum(dim=1).pow(0.5)
    b_norm = b.pow(2).sum(dim=1).pow(0.5)
    cos = inner_product / (a_norm * b_norm)
    angle = torch.acos(cos)
    return angle


def get_rot_axis(a, b):
    rotation_axis = torch.cross(a, b)
    rotation_axis = rotation_axis / rotation_axis.norm(dim=1, keepdim=True).expand(-1,3,-1)
    return rotation_axis


def KRT_from(cam, scale, device=None):
    batch_size = len(cam["intrinsics"]["cy"])
    cy = cam["intrinsics"]["cy"].float() * scale
    cx = cam["intrinsics"]["cx"].float() * scale
    fx = cam["intrinsics"]["fx"].float() * scale
    fy = cam["intrinsics"]["fy"].float() * scale
    R = cam["R"].float()
    t = cam["t"].float()
    # cx = cx.view(batch_size, 1, 1)
    # cy = cy.view(batch_size, 1, 1)
    # fx = fx.view(batch_size, 1, 1)
    # fy = fy.view(batch_size, 1, 1)
    K = torch.zeros_like(R)
    K[:, 0, 0] = fx
    K[:, 0, 2] = cx
    K[:, 1, 1] = fy
    K[:, 1, 2] = cy
    K[:, 2, 2] = torch.ones(cy.shape)
    K, R, t, cx, cy, fx, fy = K.cuda(device).detach(), R.cuda(device).detach(), t.cuda(device).detach(), \
                              cx.cuda(device).detach(), cy.cuda(device).detach(), \
                              fx.cuda(device).detach(), fy.cuda(device).detach()
    return K, R, t, cx, cy, fx, fy


def reproject_deeppruner(left_input, depth_samples, left_cam, right_cam):
    """

    :param left_input: [batch size, channels, height, width]
    :param depth_samples: [batch size, depth samples, height, width]
    :param left_cam: ['R':[batch size, 3, 3], 'pos':[batch size, 3, 1], 't':[batch size, 3, 1], 'intrinsics', 'ori':[batch size, 3, 3]]
    :param right_cam: ['R', 'pos', 't', 'intrinsics', 'ori']
    :return:
    """
    device = left_input.get_device()
    batch_size = depth_samples.shape[0]
    n_samples = depth_samples.shape[1]
    height = depth_samples.shape[2]
    width = depth_samples.shape[3]

    # load K R t
    K_l, R_l, t_l, cx_l, cy_l, fx_l, fy_l = KRT_from(left_cam)
    K_r, R_r, t_r, cx_r, cy_r, fx_r, fy_r = KRT_from(right_cam)

    # convert to (batch size, samples, height, width)
    # cx_l = cx_l.view(batch_size, 1, 1, 1).repeat(1, n_samples, height, width)
    # cy_l = cy_l.view(batch_size, 1, 1, 1).repeat(1, n_samples, height, width)
    # fx_l = fx_l.view(batch_size, 1, 1, 1).repeat(1, n_samples, height, width)
    # fy_l = fy_l.view(batch_size, 1, 1, 1).repeat(1, n_samples, height, width)

    # R_l = R_l.view(batch_size, 1, 3, 3).repeat(1, n_samples, 1, 1)
    t_l = t_l.view(batch_size, 1, 1, 1, 3).repeat(1, n_samples, height, width, 1)
    # R_r = R_r.view(batch_size, 1, 3, 3).repeat(1, n_samples, 1, 1)
    t_r = t_r.view(batch_size, 1, 1, 1, 3).repeat(1, n_samples, height, width, 1)

    # convert u and v indexing in (batch size, 3*samples, height, width)
    left_u = torch.arange(0.0, left_input.size()[3], device=device).view(1, 1, 1, -1) \
        .repeat(batch_size, n_samples, height, 1)
    left_v = torch.arange(0.0, left_input.size()[2], device=device).view(1, 1, -1, 1) \
        .repeat(batch_size, n_samples, 1, width)
    x_3d_left = (left_u - cx_l) * depth_samples / fx_l
    y_3d_left = (left_v - cy_l) * depth_samples / fy_l

    # (batch, depth samples, u, v, xyz)
    P_left = torch.stack([x_3d_left, y_3d_left, depth_samples], dim=4)

    uvz_right = torch.zeros_like(P_left)
    P_right = torch.zeros_like(P_left)
    for i in range(batch_size):
        P_right[i] = torch.matmul(R_r[i], torch.matmul(R_l[i].transpose(0, 1),
                                                       (P_left[i, :] - t_l[i]).reshape(-1, 3, 1))).reshape(n_samples,
                                                                                                           height,
                                                                                                           width, 3) \
                     + t_r[i]
        uvz_right[i, :] = torch.matmul(K_r[i], P_right[i].view(-1, 3, 1)).view(n_samples, height, width, 3)

    z_right = uvz_right[:, :, :, :, 2].view(batch_size, n_samples, height, width, 1).repeat(1, 1, 1, 1, 3)
    uvz_right = uvz_right / z_right
    return uvz_right


def reproject(cur_coord, d_vec, cam_param_l, cam_param_r):
    batch_size, c, h, w = cur_coord.shape

    K_l, R_l, t_l, cx_l, cy_l, fx_l, fy_l = cam_param_l
    K_r, R_r, t_r, cx_r, cy_r, fx_r, fy_r = cam_param_r

    # # stereo test case passes
    # R_r = R_l.clone()
    # t_r = t_r.clone()
    # t_r[0,0] = t_r[0,0] + 10

    cx_l = cx_l.view(batch_size, 1, 1).expand(-1, h, w)
    cy_l = cy_l.view(batch_size, 1, 1).expand(-1, h, w)
    fx_l = fx_l.view(batch_size, 1, 1).expand(-1, h, w)
    fy_l = fy_l.view(batch_size, 1, 1).expand(-1, h, w)
    t_l = t_l.view(batch_size, 3, 1, 1).expand(-1, -1, h, w)
    t_r = t_r.view(batch_size, 3, 1, 1).expand(-1, -1, h, w)

    x_3d_left = (cur_coord[:, 0, :, :] - cx_l) * d_vec[:,0,:,:] / fx_l
    y_3d_left = (cur_coord[:, 1, :, :] - cy_l) * d_vec[:,0,:,:] / fy_l

    # (batch, depth samples, u, v, xyz)
    P_left = torch.stack([x_3d_left, y_3d_left, d_vec[:,0,:,:]], dim=1)

    uvz_right = torch.zeros_like(P_left)
    P_right = torch.zeros_like(P_left)
    for i in range(batch_size):
        P_right[i] = (R_r[i] @
                      (R_l[i].transpose(0, 1) @ (P_left[i, :] - t_l[i]).view(3, -1))).view(3, h, w) \
                     + t_r[i]
        uvz_right[i, :] = (K_r[i] @ P_right[i].view(3, -1)).view(3, h, w)

    # TEST: plot pointcloud
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_3d_left.reshape(-1)[::10].detach().cpu().numpy(),
    #            y_3d_left.reshape(-1)[::10].detach().cpu().numpy(),
    #            d_vec[:, 0, :, :].reshape(-1)[::10].detach().cpu().numpy(), s=1)
    # fig.show()

    z_right = uvz_right[:, 2, :, :].unsqueeze(1).expand(-1,3,-1,-1)
    reproj_left_coord = (uvz_right / z_right)[:,:2,:,:].detach()

    return reproj_left_coord


def warp_right_stereo(left_input, right_input, left_cam, right_cam, depth_samples, device):
    left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(
        left_input.size()[2]).view(left_input.size()[2], left_input.size()[3])

    left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
    left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

    right_feature_map = right_input.expand(depth_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
    left_feature_map = left_input.expand(depth_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

    # uvz_right: (batch, depth samples, v, u, xyz)
    # note: it's (V,U)-> (u', v', 1), so height width switched in key and value
    uvz_right = reproject(left_input, depth_samples, left_cam, right_cam)
    right_x_coordinate = torch.clamp(uvz_right[:, :, :, :, 0].round(), min=0, max=right_input.size()[3] - 1)

    # visualize disparity:
    # import matplotlib.pyplot as plt
    # plt.plot(right_y_coordinate)

    # indices are floored, not interpolated
    warped_right_feature_map = torch.gather(right_feature_map,
                                            dim=4,
                                            index=right_x_coordinate.expand(
                                                right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())
    return left_feature_map, warped_right_feature_map


def squared_dist(left_feat, right_feat):
    diff = torch.pow(left_feat - right_feat, 2).sum(dim=1)
    # left_feat.register_hook(lambda grad: print(torch.isnan(grad).sum()))
    # print("diff {}".format(torch.isnan(diff).sum()))
    # print("diff min {}".format(diff.min()))
    # print("diff max {}".format(diff.max()))
    # print("diff mean {}".format(diff.mean()))
    return diff

