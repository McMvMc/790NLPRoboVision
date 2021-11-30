import numpy as np
import scipy.spatial.transform as transform
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn.functional as F

from models.model_helpers import *

MAX_PLANE_DEPTH = 300
MIN_PLANE_DEPTH = 200


def visualize_against_gt(estimated_d_n, gt_depth, batch_id):
    fig = plt.figure()
    ax1 = plt.subplot(311)
    ax1.title.set_text('PM estimate no training no warper match')
    im1 = ax1.imshow((estimated_d_n[batch_id,0]/ min(gt_depth[batch_id].max(), 400.)).detach().cpu().numpy(), vmin=0, vmax=1)
    fig.colorbar(im1)
    ax2 = plt.subplot(312)
    ax2.title.set_text('GT')
    im2 = ax2.imshow((gt_depth[batch_id]/ min(gt_depth[batch_id].max(), 400.)).detach().cpu().numpy(), vmin=0, vmax=1)
    fig.colorbar(im2)
    ax3 = plt.subplot(313)
    ax3.title.set_text('GT & esti diff')
    im3 = ax3.imshow(((gt_depth[batch_id]-estimated_d_n[batch_id,0]).abs()).detach().cpu().numpy())
    fig.colorbar(im3)
    fig.suptitle('depth')
    plt.colorbar()
    plt.show()
    print("median diff: {}".format((gt_depth[batch_id]-estimated_d_n[batch_id,0]).abs().median()))


def load_warper_weights(model, warper_weight_path):
    warper_state_dict = torch.load(warper_weight_path)
    warper_weights = {}
    for key in warper_state_dict['state_dict'].keys():
        if "warper" in key:
            warper_key = key.replace('module.warper.','')
            warper_weights[warper_key] = warper_state_dict['state_dict'][key]
    model.patch_match_depth.warper.load_state_dict(warper_weights, strict=True)
    return model


def angle_between_vectors(a, b):
    inner_product = (a * b).sum(dim=1)
    a_norm = a.pow(2).sum(dim=1).pow(0.5)
    b_norm = b.pow(2).sum(dim=1).pow(0.5)
    cos = inner_product / (a_norm * b_norm)
    angle = torch.acos(cos)
    return angle
#
#
# def get_rot_axis(a, b):
#     rotation_axis = torch.cross(a, b)
#     rotation_axis = rotation_axis / rotation_axis.norm(dim=1).unsqueeze(1).expand(-1,3,-1)
#     return rotation_axis


def gen_normal_map(depth_map, left_cam):
    K_l, R_l, t_l, cx_l, cy_l, fx_l, fy_l = KRT_from(left_cam, 1.0)
    device = K_l.device
    batch_size, h, w = depth_map.shape
    pts_3d = unproject(K_l, depth_map.double()).float()
    h_filter = torch.tensor([[-1., 0., 1.]], device=device).view(1, 1, 1, 3)
    v_filter = torch.tensor([[-1.], [0.], [1.]], device=device).view(1, 1, 3, 1)

    dzdx = F.conv2d(pts_3d[:, 2].unsqueeze(1), h_filter, padding=(0, 1)) / 2.
    dzdx[:, :, :, 0] = (pts_3d[:, 2, :, 1] - pts_3d[:, 2, :, 0]).unsqueeze(1)
    dzdx[:, :, :, w-1] = (pts_3d[:, 2, :, w-1] - pts_3d[:, 2, :, w-2]).unsqueeze(1)
    dzdy = F.conv2d(pts_3d[:, 2].unsqueeze(1), v_filter, padding=(1, 0)) / 2.
    dzdy[:, :, 0, :] = (pts_3d[:, 2, 1, :] - pts_3d[:, 2, 0, :]).unsqueeze(1)
    dzdy[:, :, h-1, :] = (pts_3d[:, 2, h-1, :] - pts_3d[:, 2, h-2, :]).unsqueeze(1)

    # x vector
    dzdx_vec_3d = torch.zeros_like((pts_3d), device=device)
    dzdx_vec_3d[:, 0, :, :-1] = pts_3d[:, 0, :, 1:] - pts_3d[:, 0, :, :-1]
    dzdx_vec_3d[:, 0, :, -1] = pts_3d[:, 0, :, -1] - pts_3d[:, 0, :, -2]
    dzdx_vec_3d[:, 2] = dzdx[:,0]

    # y vector
    dzdy_vec_3d = torch.zeros_like((pts_3d), device=device)
    dzdy_vec_3d[:, 1, :-1, :] = pts_3d[:, 1, 1:, :] - pts_3d[:, 1, :-1, :]
    dzdy_vec_3d[:, 1, -1, :] = pts_3d[:, 1, -1, :] - pts_3d[:, 1, -2, :]
    dzdy_vec_3d[:, 2] = dzdy[:,0]

    normal_map = torch.cross(dzdy_vec_3d, dzdx_vec_3d)
    normal_map = normalize_vector(normal_map, 1, keepdim=True)

    # # visualize
    # plt.imshow(normal_map[0].permute(1, 2, 0).detach().cpu().numpy()); plt.show()
    # # visualize absolute val
    # plt.imshow(normal_map[0].permute(1, 2, 0).detach().cpu().abs().numpy()); plt.show()

    return normal_map


def gen_stereo_homography(R_l, t_l, R_r, t_r, K, plane_param):
    """
    moves the 'right' camera left right of the 'left' camera while looking at the center of the image plane
    :param relative_x_position_to_left_cam:  positive is right to the left cam
    :param relative_z_position_to_left_cam:  positive is behind the left cam
    :return:
    """
    batch_size, _, _ = plane_param.shape

    # homography: p_r = (z_l/z_r * K_r * H_l_to_r * K_l^{-1}) * p_l, H_l_to_r = R-tn^T/d
    # NOTE: not sure if z_l and z_r are needed, check result: not needed

    R_lr = (R_r @ R_l.transpose(1, 2)).double()
    t_lr = (R_r @ (-R_l.transpose(1, 2) @ t_l) + t_r).double()

    H_lr = (R_lr - t_lr @ plane_param[:,:3].transpose(1,2) / plane_param[:,3].unsqueeze(2).expand(-1,3,3)).double()

    return (K @ H_lr @ K.inverse()).double(), R_lr, t_lr


def gen_random_angle(min_angle, max_angle, batch_size):
    rotation_angle = (torch.rand([batch_size, 1]) * (max_angle - min_angle) + min_angle) \
                     / 180 * math.pi
    return rotation_angle


def rotate_to_lookat(plane_cent, O, P):
    batch_size = plane_cent.shape[0]
    lookat_vec = plane_cent - P
    lookat_vec = lookat_vec / lookat_vec.norm(dim=1).expand(-1,3).unsqueeze(2)
    normal = torch.tensor(np.array([[0, 0, 1.]]).T).unsqueeze(0).expand(batch_size,3,1)
    rotation_angle = angle_between_vectors(normal, lookat_vec)
    # rotation_axis = torch.cross(O[:,:,2].unsqueeze(2), lookat_vec)
    # rotation_axis = rotation_axis / rotation_axis.norm(dim=1).expand(-1,3).unsqueeze(2)
    rotation_axis = get_rot_axis(O[:,:,2].unsqueeze(2), lookat_vec)
    angle_axis = rotation_angle.expand(-1,3).unsqueeze(2) * rotation_axis
    R_rot = transform.Rotation.from_rotvec(angle_axis.squeeze())
    return R_rot, rotation_angle


def flowfield_from_homography(H, left_img):
    batch_size, c, h, w = left_img.shape
    flow_field = torch.ones(batch_size, h, w, 3).double().cuda()
    flow_field[:, :, :, 0] = torch.arange(0, w).repeat(h).view(h, w)
    flow_field[:, :, :, 1] = torch.arange(0, h).repeat(w).view(w, h).transpose(1, 0)
    flow_field = flow_field.permute(0, 3, 1, 2).reshape(batch_size, 3, -1)
    # from warped to original pixels
    flow_field = H.inverse().cuda() @ flow_field
    flow_field = flow_field.reshape(batch_size, 3, h, w)
    flow_field[:, 0, :, :] = flow_field[:, 0, :, :] / flow_field[:, 2, :, :]
    flow_field[:, 1, :, :] = flow_field[:, 1, :, :] / flow_field[:, 2, :, :]
    flow_field[:, 2, :, :] = flow_field[:, 2, :, :] / flow_field[:, 2, :, :]

    # normalize for grid_sample()
    half_w = torch.tensor([w / 2.]).expand(batch_size, h, w).double().cuda()
    half_h = torch.tensor([h / 2.]).expand(batch_size, h, w).double().cuda()
    flow_field[:, 0, :, :] = (flow_field[:, 0, :, :] - half_w) / half_w
    flow_field[:, 1, :, :] = (flow_field[:, 1, :, :] - half_h) / half_h
    flow_field = flow_field.permute(0, 2, 3, 1)[:, :, :, :2]
    return flow_field


def warp_image_homography(H_lr, left_img):
    # grid sample grabs each pixel in right img from left, so its flowfield is actually H_right_to_left
    flow_field_l_to_r = flowfield_from_homography(H_lr.inverse(), left_img)

    # grid sample grabs each pixel in left img from right, so flowfield is H_left_to_right
    flow_field_r_to_l = flowfield_from_homography(H_lr, left_img)

    # pytorch 1.2 has align_corner=False
    warped_left = F.grid_sample(left_img.double(), flow_field_l_to_r, mode='bilinear')

    return warped_left, flow_field_l_to_r, flow_field_r_to_l


# def xyz_to_world_spherical_coord_cam_coord(d, theta, phi):
#
#     return x_r, y_r, z_r


# def RT_mat_from_R_t(R, t):
#     batch_size = t.shape[0]
#     trans_matx = torch.zeros(batch_size, 3, 4)
#     trans_matx[:, :3, :3] = R
#     trans_matx[:, :3, 3] = t
#     trans_matx[:, 3, 3] = 1.
#     return trans_matx


def unproject(K, dmap):
    batch_size, h, w = dmap.shape
    device = dmap.device

    left_u = torch.arange(0.0, w, device=device).view(1, 1, -1).repeat(batch_size, h, 1).double()
    left_v = torch.arange(0.0, h, device=device).view(1, -1, 1).repeat(batch_size, 1, w).double()
    fx = K[:, 0, 0].view(-1, 1, 1).expand(-1, h, w).double()
    cx = K[:, 0, 2].view(-1, 1, 1).expand(-1, h, w).double()
    fy = K[:, 1, 1].view(-1, 1, 1).expand(-1, h, w).double()
    cy = K[:, 1, 2].view(-1, 1, 1).expand(-1, h, w).double()
    x = (left_u - cx) * dmap / fx
    y = (left_v - cy) * dmap / fy

    pt_3d = torch.zeros((batch_size, 3, h, w), device=device).double()
    pt_3d[:, 0, :, :] = x
    pt_3d[:, 1, :, :] = y
    pt_3d[:, 2, :, :] = dmap
    return pt_3d


# def gen_warp_param_map(left_img, plane_param, K, R, t):
#     # NOTE: this is wrong, the params are actually the same, should be the change in normals
#
#     batch_size, _, h, w = left_img.shape
#     # NOTE: the warp params for points on a 3D plane are not the same. Need to [Rt] transform the 3D pixel
#     #  and then calculate the angle between orig and new lookat vecs for each pixel
#
#     # TODO: need to verify that the transformed plane look the same as the one from H
#     dmap = plane_param[:,3].unsqueeze(2).expand(-1, h, w)
#     pixel_3d_r = unproject(K, dmap)
#     new_lookat = -pixel_3d_r
#     new_lookat = normalize_vector(new_lookat, norm_dim=1, keepdim=True).expand(-1,3,-1,-1)
#     # visualize 3D
#     # fig = plt.figure()
#     # ax = fig.add_subplot(121, projection='3d')
#     # ax.scatter(pixel_3d[0, 0, :].view(-1).numpy(),
#     #            pixel_3d[0, 1, :].view(-1).numpy(),
#     #            pixel_3d[0, 2, :].view(-1).numpy(), marker='o')
#     # ax.set_xlabel('X Label')
#     # ax.set_ylabel('Y Label')
#     # ax.set_zlabel('Z Label')
#     # ax = fig.add_subplot(122, projection='3d')
#     # ax.scatter(pixel_3d[1, 0, :].view(-1).numpy(),
#     #            pixel_3d[1, 1, :].view(-1).numpy(),
#     #            pixel_3d[1, 2, :].view(-1).numpy(), marker='o')
#     # ax.set_xlabel('X Label')
#     # ax.set_ylabel('Y Label')
#     # ax.set_zlabel('Z Label')
#     # ax.view_init(elev=-90., azim=-90.)
#     # plt.show()
#
#     # trans_matx = RT_mat_from_R_t(R, t)
#     P_l = t.unsqueeze(3).expand(-1,-1, h, w)
#     pixel_3d_l = (R @ pixel_3d_r.view(batch_size, 3, -1) + t.expand(-1,-1,h*w)).view(batch_size, 3, h, w)
#     orig_lookat = P_l-pixel_3d_l
#     orig_lookat = normalize_vector(orig_lookat, norm_dim=1, keepdim=True).expand(-1, 3, -1, -1)
#
#     lookat_rot_angle = angle_between_vectors(orig_lookat, new_lookat)
#     lookat_rot_axis = get_rot_axis(orig_lookat.permute(0,2,3,1).reshape(-1, 3, 1),
#                                    new_lookat.permute(0,2,3,1).reshape(-1, 3, 1)).reshape(batch_size, h, w, 3).permute(0,3,1,2)
#
#     # from orig to new in new cam view
#     orig_to_new_rot_angle_axis_cam_r = lookat_rot_angle.unsqueeze(1).expand(batch_size, 3, h, w) * lookat_rot_axis
#
#     # unit test: from orig to new in orig cam view, NOTE: not sure if I can rotate the rotation vector this way
#     orig_to_new_rot_angle_axis_cam_r_warped_to_l = (R.transpose(1,2) @ orig_to_new_rot_angle_axis_cam_r.view(batch_size, 3, -1))\
#         .view(batch_size, 3, h, w)
#
#     # unit test: make sure that rot vecs in cam left and cam right are the same when calc in right cam coord
#     # pixel_3d_l_test = unproject(K, dmap)
#     # orig_lookat_test = -pixel_3d_l_test
#     # orig_lookat_test = normalize_vector(orig_lookat_test, norm_dim=1, keepdim=True).expand(-1,3,-1,-1)
#     #
#     # # trans_matx = RT_mat_from_R_t(R, t)
#     # P_l = (-R.transpose(1,2) @ t).unsqueeze(3).expand(-1,-1, h, w)
#     # pixel_3d_l = (R.transpose(1,2) @ (pixel_3d_r.view(batch_size, 3, -1) - t.expand(-1,-1,h*w))).view(batch_size, 3, h, w)
#     # orig_lookat = P_l-pixel_3d_l
#     # orig_lookat = normalize_vector(orig_lookat, norm_dim=1, keepdim=True).expand(-1, 3, -1, -1)
#     #
#     # lookat_rot_angle = angle_between_vectors(orig_lookat, new_lookat)
#     # lookat_rot_axis = get_rot_axis(orig_lookat.permute(0,2,3,1).reshape(-1, 3, 1),
#     #                                new_lookat.permute(0,2,3,1).reshape(-1, 3, 1)).reshape(batch_size, h, w, 3).permute(0,3,1,2)
#     #
#     # # from orig to new in new cam view
#     # orig_to_new_rot_angle_axis_cam_r = lookat_rot_angle.unsqueeze(1).expand(batch_size, 3, h, w) * lookat_rot_axis
#
#     # unit test: verify params by rotating orig lookat vecs to new vecs using warp params and take diff
#     # max_diff = 0
#     # max_row = None
#     # max_r = 0
#     # for r in range(h):
#     #     tmp_o = orig_lookat.permute(0, 2, 3, 1)[0, r, :, :].reshape(-1, 3)
#     #     tmp_rot = orig_to_new_rot_angle_axis_cam_r[0,:,r,:].t()
#     #     rot_m = torch.tensor(transform.Rotation.from_rotvec(tmp_rot).as_matrix())
#     #     tmp_new_lookat = rot_m @ tmp_o.unsqueeze(2)
#     #     diff = tmp_new_lookat.squeeze() - new_lookat[0, :, r, :].t()
#     #     cur_max_diff = (diff.norm(dim=1).abs()).max()
#     #     if cur_max_diff > max_diff:
#     #         max_diff = cur_max_diff
#     #         max_row = diff
#     #         max_r = r
#     #         print(max_r)
#     #     if r%100 == 0:
#     #         plt.plot(diff.norm(dim=1)); plt.show()
#     #         plt.plot(tmp_new_lookat.norm(dim=1)); plt.show()
#     #         plt.plot(new_lookat[0, :, r, :].t().norm(dim=1)); plt.show()
#     # plt.plot(max_row.norm(dim=1)); plt.show()
#     # print("max diff row: {}".format(max_r))
#
#     return orig_to_new_rot_angle_axis_cam_r


def gen_warp_param_map(left_img, R):
    batch_size, c, h, w = left_img.shape
    orig_norm = torch.tensor([[0.,0.,-1.]]).t().unsqueeze(0).expand(batch_size, -1, -1).double()
    new_norm = R.transpose(1,2) @ orig_norm.clone()

    norm_rot_angle = angle_between_vectors(new_norm, orig_norm)
    norm_rot_axis = get_rot_axis(new_norm, orig_norm)

    norm_rot_angle_axis = norm_rot_angle.unsqueeze(1).expand(-1,3,-1) * norm_rot_axis

    # dmap = plane_param[:, 3].unsqueeze(2).expand(-1, h, w)
    # pixel_3d_r = unproject(K, dmap)

    return norm_rot_angle_axis.unsqueeze(3).expand(-1, -1, h, w)


def gen_warped_stereo_image(left_img, left_cam, max_surface_normal):
    batch_size, c, h, w = left_img.shape

    # TODO: need to determine plane depth range
    d = torch.rand([batch_size, 1]) * (MAX_PLANE_DEPTH - MIN_PLANE_DEPTH) + MIN_PLANE_DEPTH
    plane_param = torch.zeros([batch_size, 4, 1]).double()
    plane_param[:, 2] = -1.0
    plane_param[:, 3] = d
    plane_cent = torch.zeros([batch_size, 3, 1]).double()
    plane_cent[:,2] = plane_param[:,3]

    cy_l = left_cam["intrinsics"]["cy"].float()
    cx_l = left_cam["intrinsics"]["cx"].float()
    fx_l = left_cam["intrinsics"]["fx"].float()
    fy_l = left_cam["intrinsics"]["fy"].float()
    K = torch.zeros([batch_size, 3, 3]).double()
    K[:, 0, 0] = fx_l
    K[:, 1, 1] = fy_l
    K[:, 0, 2] = cx_l
    K[:, 1, 2] = cy_l
    K[:, 2, 2] = 1.
    R_l = torch.eye(3).double().repeat(batch_size, 1, 1)
    t_l = torch.zeros(3,1).double().repeat(batch_size, 1, 1)

    # TODO: need to determine movement range
    theta = gen_random_angle(90-max_surface_normal, 90+max_surface_normal, batch_size)  # elevation angle
    phi = gen_random_angle(-max_surface_normal, max_surface_normal, batch_size)  # azimuth angle
    # theta[0] = math.pi*3 / 4
    # phi[0] = 0.
    # print("theta: {}".format(theta / math.pi * 180.))
    # print("phi: {}".format(phi / math.pi * 180.))
    x_r, y_r, z_r = spherical_coord_world_to_xyz_cam_coord(d, theta, phi)
    z_r += d  # plane is d away from origin
    P_r = torch.cat((x_r, y_r, z_r), dim=1).view(batch_size, 3, 1).double()
    O_r, rotation_angle = rotate_to_lookat(plane_cent, R_l, P_r)
    print("surface normal change by {} degrees".format(rotation_angle/math.pi * 180))
    print("mean surface normal change by {} degrees".format(rotation_angle.mean() / math.pi * 180))

    # TODO: worry about in plane rotation later? cuz most of the data have upright cameras
    # in_plane_rot = gen_random_angle(90-max_surface_normal, 90+max_surface_normal, batch_size)  # elevation angle

    R_r = torch.tensor(O_r.as_matrix()).transpose(1,2).double()
    t_r = (-R_r @ P_r).double()
    H_lr, _, _ = gen_stereo_homography(R_l, t_l, R_r, t_r, K, plane_param)

    right_image, flow_field_l_to_r, flow_field_r_to_l = warp_image_homography(H_lr, left_img)

    # generate warp parameters for each pixel
    # NOTE: the angle axis params are in right cam coord
    warp_r_to_l_param_map_cam_l = gen_warp_param_map(left_img, R_r)

    # generate mask
    # mask = torch.ones_like(flow_field[:,:3,:,:])

    # visualize
    # plt.imshow(warped_left[0].permute(1,2,0).cpu().numpy()); plt.show()

    # # get left to right flow field
    # inverse_flow_field = torch.ones(1, h, w, 3).double()
    # inverse_flow_field[:, :, :, 0] = torch.arange(0, w).repeat(h).view(h, w)
    # inverse_flow_field[:, :, :, 1] = torch.arange(0, h).repeat(w).view(w, h).transpose(1, 0)
    # inverse_flow_field = inverse_flow_field.permute(3, 0, 1, 2).reshape(3, -1)
    # # from warped to original pixels
    # inverse_flow_field = H_lr @ inverse_flow_field
    # inverse_flow_field = inverse_flow_field.reshape(3, 1, h, w)
    # inverse_flow_field[0, :, :, :] = inverse_flow_field[0, :, :, :] / inverse_flow_field[2, :, :, :]
    # inverse_flow_field[1, :, :, :] = inverse_flow_field[1, :, :, :] / inverse_flow_field[2, :, :, :]
    # inverse_flow_field[2, :, :, :] = inverse_flow_field[2, :, :, :] / inverse_flow_field[2, :, :, :]
    #
    # # normalize for grid_sample()
    # inverse_flow_field[0, :, :, :] = (inverse_flow_field[0, :, :, :] - w / 2.) / (w / 2.)
    # inverse_flow_field[1, :, :, :] = (inverse_flow_field[1, :, :, :] - h / 2.) / (h / 2.)
    # inverse_flow_field = inverse_flow_field.squeeze(1).permute(1, 2, 0)[:, :, :2].unsqueeze(0)

    return right_image, flow_field_l_to_r, flow_field_r_to_l, H_lr, warp_r_to_l_param_map_cam_l
