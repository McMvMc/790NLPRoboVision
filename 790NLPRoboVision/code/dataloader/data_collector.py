rom __future__ import print_function
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import os.path
import logging

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# DEFAULT is 32MM
CAM_INTRINSICS_32MM = {'fx': 1050.0, 'fy': 1050.0, 'cx': 479.5, 'cy': 269.5}

# driving has 15MM dataset
CAM_INTRINSICS_15MM = {'fx': 450.0, 'fy': 450.0, 'cx': 479.5, 'cy': 269.5}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_cam_params(cam_path, cam_model):
    # gt is camera to world (Orientation, Position), convert to (Rotation, Translation)
    def read_Rt(cur_str):
        s = list(map(float, cur_str.split()[1:]))
        ori = np.array([[s[0], s[1], s[2]],
                       [s[4], s[5], s[6]],
                       [s[8], s[9], s[10]]])
        pos = np.array([[s[3], s[7], s[11]]]).T

        R = ori.T
        t = -np.matmul(R, pos)

        return R, t, ori, pos

    cam_file = open(cam_path, 'r')
    lines = cam_file.readlines()

    all_left_cam = {}
    all_right_cam = {}
    # all_cam_idx = []
    i = 0
    while i < len(lines):
        cur_line = lines[i]
        cur_cam_idx = int(cur_line.split()[1])
        # all_cam_idx.append(int(cur_line.split()[1]))

        i += 1
        try:
            cur_line = lines[i]
            R_l, t_l, ori_l, pos_l = read_Rt(cur_line)
            left_cam = {'intrinsics':{'fx':cam_model['fx'], 'fy':cam_model['fy'], 'cx':cam_model['cx'], 'cy':cam_model['cy']},
                        'R':R_l, 't':t_l, 'ori': ori_l, 'pos': pos_l}
            all_left_cam[cur_cam_idx] = left_cam
        except:
            print("error loading left camera param at line: {}".format(i))

        i += 1
        try:
            cur_line = lines[i]
            R_r, t_r, ori_r, pos_r = read_Rt(cur_line)
            right_cam = {'intrinsics': {'fx': cam_model['fx'], 'fy': cam_model['fy'], 'cx': cam_model['cx'],
                                       'cy': cam_model['cy']},
                        'R':R_r, 't':t_r, 'ori': ori_r, 'pos': pos_r}
            all_right_cam[cur_cam_idx] = right_cam
        except:
            print("error loading right camera param at line: {}".format(i))

        # skip the empty line
        i += 2

    return all_left_cam, all_right_cam


def dataloader(filepath_monkaa=None, filepath_flying=None, filepath_driving=None):
    all_left_cam = []
    all_right_cam = []
    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_left_cam = []
    test_right_cam = []

    if filepath_monkaa:
        try:
            monkaa_path = os.path.join(filepath_monkaa, 'monkaa_frames_cleanpass')
            monkaa_disp = os.path.join(filepath_monkaa, 'monkaa_disparity')
            monkaa_cam = os.path.join(filepath_monkaa, 'camera_data')
            monkaa_dir = os.listdir(monkaa_path)

            for dd in monkaa_dir:
                left_cam_dict, right_cam_dict = read_cam_params(os.path.join(monkaa_cam, dd, 'camera_data.txt'),
                                                      CAM_INTRINSICS_32MM)
                for im in os.listdir(os.path.join(monkaa_path, dd, 'left')):
                    im_idx = int(im.split('.')[0])
                    if is_image_file(os.path.join(monkaa_path, dd, 'left', im)) and is_image_file(
                            os.path.join(monkaa_path, dd, 'right', im)):
                        all_left_img.append(os.path.join(monkaa_path, dd, 'left', im))
                        all_left_disp.append(os.path.join(monkaa_disp, dd, 'left', im.split(".")[0] + '.pfm'))
                        all_left_cam.append(left_cam_dict[im_idx])
                        all_right_img.append(os.path.join(monkaa_path, dd, 'right', im))
                        all_right_cam.append(right_cam_dict[im_idx])

        except:
            logging.error("Some error in Monkaa, Monkaa might not be loaded correctly in this case...")
            raise Exception('Monkaa dataset couldn\'t be loaded correctly.')

    if filepath_flying:    
        # try:
        flying_path = os.path.join(filepath_flying, 'frames_cleanpass')
        flying_disp = os.path.join(filepath_flying, 'disparity')
        flying_cam = os.path.join(filepath_flying, 'camera_data')
        flying_dir = flying_path + '/TRAIN/'
        subdir = ['A', 'B', 'C']

        missing_cam_imgs = []
        for ss in subdir:
            flying = os.listdir(os.path.join(flying_dir, ss))
            for ff in flying:
                left_cam_dict, right_cam_dict = read_cam_params(
                    os.path.join(flying_cam, 'TRAIN', ss, ff, 'camera_data.txt'),
                    CAM_INTRINSICS_32MM)
                imm_l = os.listdir(os.path.join(flying_dir, ss, ff, 'left'))
                for im in imm_l:
                    im_idx = int(im.split('.')[0])
                    if im_idx in left_cam_dict and im_idx in right_cam_dict:
                        all_left_cam.append(left_cam_dict[im_idx])
                        if is_image_file(os.path.join(flying_dir, ss, ff, 'left', im)):
                            all_left_img.append(os.path.join(flying_dir, ss, ff, 'left', im))
                        all_left_disp.append(os.path.join(flying_disp, 'TRAIN', ss, ff, 'left', im.split(".")[0] + '.pfm'))

                        if is_image_file(os.path.join(flying_dir, ss, ff, 'right', im)):
                            all_right_img.append(os.path.join(flying_dir, ss, ff, 'right', im))
                        all_right_cam.append(right_cam_dict[im_idx])
                    else:
                        missing_cam_imgs.append(os.path.join(flying_dir, ss, ff, 'left', im))

        flying_dir = flying_path + '/TEST/'
        subdir = ['A', 'B', 'C']

        for ss in subdir:
            flying = os.listdir(os.path.join(flying_dir, ss))
            for ff in flying:
                left_cam_dict, right_cam_dict = read_cam_params(
                    os.path.join(flying_cam, 'TEST', ss, ff, 'camera_data.txt'),
                    CAM_INTRINSICS_32MM)
                imm_l = os.listdir(os.path.join(flying_dir, ss, ff, 'left'))
                for im in imm_l:
                    im_idx = int(im.split('.')[0])
                    if im_idx in left_cam_dict and im_idx in right_cam_dict:
                        if is_image_file(os.path.join(flying_dir, ss, ff, 'left', im)):
                            test_left_img.append(os.path.join(flying_dir, ss, ff, 'left', im))
                        test_left_cam.append(left_cam_dict[im_idx])
                        test_left_disp.append(os.path.join(flying_disp, 'TEST', ss, ff, 'left', im.split(".")[0] + '.pfm'))

                        if is_image_file(os.path.join(flying_dir, ss, ff, 'right', im)):
                            test_right_img.append(os.path.join(flying_dir, ss, ff, 'right', im))
                        test_right_cam.append(right_cam_dict[im_idx])
                    else:
                        missing_cam_imgs.append(os.path.join(flying_dir, ss, ff, 'left', im))
        # except:
        #     logging.error("Some error in Flying Things, Flying Things might not be loaded correctly in this case...")
        #     raise Exception('Flying Things dataset couldn\'t be loaded correctly.')

    if filepath_driving:
        try:
            driving_dir = os.path.join(filepath_driving, 'driving_frames_cleanpass/')
            driving_disp = os.path.join(filepath_driving, 'driving_disparity/')
            driving_cam = os.path.join(filepath_driving, 'camera_data')

            subdir1 = ['35mm_focallength', '15mm_focallength']
            subdir2 = ['scene_backwards', 'scene_forwards']
            subdir3 = ['fast', 'slow']

            for i in subdir1:
                for j in subdir2:
                    for k in subdir3:
                        left_cam_dict, right_cam_dict = read_cam_params(os.path.join(driving_cam, i, j, k, 'camera_data.txt'),
                                                                        CAM_INTRINSICS_32MM if '32' in subdir1
                                                                        else CAM_INTRINSICS_15MM)
                        imm_l = os.listdir(os.path.join(driving_dir, i, j, k, 'left'))
                        for im in imm_l:
                            im_idx = int(im.split('.')[0])
                            if is_image_file(os.path.join(driving_dir, i, j, k, 'left', im)):
                                all_left_img.append(os.path.join(driving_dir, i, j, k, 'left', im))
                            all_left_disp.append(os.path.join(driving_disp, i, j, k, 'left', im.split(".")[0] + '.pfm'))
                            all_left_cam.append(left_cam_dict[im_idx])
                            if is_image_file(os.path.join(driving_dir, i, j, k, 'right', im)):
                                all_right_img.append(os.path.join(driving_dir, i, j, k, 'right', im))
                            all_right_cam.append(right_cam_dict[im_idx])

            # driving_test_dir = driving_dir + '/TEST/'
            #
            # imm_l = os.listdir(os.path.join(driving_test_dir, 'left'))
            # for im in imm_l:
            #     if is_image_file(os.path.join(driving_test_dir, 'left', im)):
            #         test_left_img.append(os.path.join(driving_test_dir, 'left', im))
            #
            #     test_left_disp.append(os.path.join(driving_disp, 'TEST', 'left', im.split(".")[0] + '.pfm'))
            #
            #     if is_image_file(os.path.join(driving_test_dir, 'right', im)):
            #         test_right_img.append(os.path.join(driving_test_dir, 'right', im))
        except:
            logging.error("Some error in Driving, Driving might not be loaded correctly in this case...")
            raise Exception('Driving dataset couldn\'t be loaded correctly.')

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, \
           all_left_cam, all_right_cam, test_left_cam, test_right_cam