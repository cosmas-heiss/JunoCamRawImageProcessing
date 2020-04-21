import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from PIL import Image
from mayavi import mlab
import cv2

import sys
sys.path.append("..")
from core.Util import *
from core.Framelets import *
from core.JitterCorrection import *
from core.Vis3D import *
from core.ColorCorrection import *


image0 = dir_path + '/../RawImageData/PJ21/RawImages/JNCE_2019202_21C00033_V01-raw.png'
im_info0 = dir_path + '/../RawImageData/PJ21/RawImageData/JNCE_2019202_21C00033_V01.json'
image1 = dir_path + '/../RawImageData/PJ21/RawImages/JNCE_2019202_21C00036_V01-raw.png'
im_info1 = dir_path + '/../RawImageData/PJ21/RawImageData/JNCE_2019202_21C00036_V01.json'
image2 = dir_path + '/../RawImageData/PJ21/RawImages/JNCE_2019202_21C00039_V01-raw.png'
im_info2 = dir_path + '/../RawImageData/PJ21/RawImageData/JNCE_2019202_21C00039_V01.json'


pos = np.array([0.5,0.7,-0.3])
pos *= JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos)
pos[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
FRAMELET_RES_X = 2048
FRAMELET_RES_Y = 1024
raster = project_tangential_plane(pos, 40000, 20000, FRAMELET_RES_X, FRAMELET_RES_Y, 0)

def get_image_gradients_and_data(image_path, im_info_path):
    img = Image.open(image_path)
    im_ar = np.array(img)
    im_ar = remove_bad_pixels(im_ar)

    with open(im_info_path, 'rb') as json_file:
        im_info_dict = json.load(json_file)

    start_time = im_info_dict["START_TIME"]
    frame_delay = float(im_info_dict["INTERFRAME_DELAY"].split()[0]) + 0.001

    start_correction, frame_delay = correct_image_start_time_and_frame_delay(im_ar, start_time, frame_delay)

    framelets = generate_framelets(revert_square_root_encoding(im_ar), start_time, start_correction, frame_delay)

    gradient_fields = [[], [], []]
    colors = np.zeros((FRAMELET_RES_Y, FRAMELET_RES_X, 3))
    color_counts = np.zeros((FRAMELET_RES_Y, FRAMELET_RES_X, 3))
    for k, framelet in enumerate(framelets):
        print_progress_bar(k+1, len(framelets),
                           'Processing framelet {} of {}:'.format(k + 1, len(framelets)),
                           length=18)

        brightnesses, y_coords, valid = framelet.get_pixel_val_at_surf_point(raster, sun_brightness_correction=True,
                                                                             return_y_indices=True)
        x_gradientx = (np.gradient(y_coords, axis=0) * valid)[..., None]
        x_gradienty = (np.gradient(y_coords, axis=1) * valid)[..., None]
        x_tangential_field = np.concatenate((x_gradientx, x_gradienty), axis=-1)
        x_tangential_field_norm_sq = np.sum(x_tangential_field ** 2, axis=-1)
        x_tangential_field_norm_sq = np.where(x_tangential_field_norm_sq != 0, x_tangential_field_norm_sq, 1)

        gradient_fields[2 - framelet.color].append(x_tangential_field / x_tangential_field_norm_sq[..., None])
        colors[..., 2 - framelet.color] += brightnesses
        color_counts[..., 2 - framelet.color] += valid
    colors = colors / np.maximum(color_counts, 1)
    mask = np.all(color_counts > 0, axis=-1)

    return colors, gradient_fields, mask



img0, gradient_fields0, mask0 = get_image_gradients_and_data(image0, im_info0)
np.save('example10_qs_img_mask0', {'img0':img0, 'm0':mask0})
np.save('example10_qs_grad_fields0', {'g0':gradient_fields0})
del img0, gradient_fields0, mask0

img1, gradient_fields1, mask1 = get_image_gradients_and_data(image1, im_info1)
np.save('example10_qs_img_mask1', {'img1':img1, 'm1':mask1})
np.save('example10_qs_grad_fields1', {'g1':gradient_fields1})
del img1, gradient_fields1, mask1

img2, gradient_fields2, mask2 = get_image_gradients_and_data(image2, im_info2)
np.save('example10_qs_img_mask2', {'img2':img2, 'm2':mask2})
np.save('example10_qs_grad_fields2', {'g2':gradient_fields2})
del img2, gradient_fields2, mask2


with open(im_info0, 'rb') as json_file:
    im_info_dict = json.load(json_file)
    second_count0 = int(im_info_dict["SPACECRAFT_CLOCK_START_COUNT"].split(':')[0])
with open(im_info1, 'rb') as json_file:
    im_info_dict = json.load(json_file)
    second_count1 = int(im_info_dict["SPACECRAFT_CLOCK_START_COUNT"].split(':')[0])
with open(im_info2, 'rb') as json_file:
    im_info_dict = json.load(json_file)
    second_count2 = int(im_info_dict["SPACECRAFT_CLOCK_START_COUNT"].split(':')[0])

time_delay01 = second_count1 - second_count0
time_delay12 = second_count2 - second_count1
pixel_in_meter = 20000000 / FRAMELET_RES_Y



img_mask_dict0 = np.load('example10_qs_img_mask0.npy', allow_pickle=True).item()
#grad_field_dict0 = np.load('example10_qs_grad_fields0.npy', allow_pickle=True).item()
img_mask_dict1 = np.load('example10_qs_img_mask1.npy', allow_pickle=True).item()
#grad_field_dict1 = np.load('example10_qs_grad_fields1.npy', allow_pickle=True).item()
#all_gradient_fields01 = [x+y for x,y in zip(grad_field_dict0['g0'], grad_field_dict1['g1'])]
mask01 = np.logical_and(img_mask_dict0['m0'], img_mask_dict1['m1'])
img0, img1 = img_mask_dict0['img0'], img_mask_dict1['img1']
#del grad_field_dict0

vel01 = compute_optical_flow(img0, img1, mask=mask01)#, error_fields=all_gradient_fields01)
vel01 = vel01 * pixel_in_meter / time_delay01
#del all_gradient_fields01


img_mask_dict2 = np.load('example10_qs_img_mask2.npy', allow_pickle=True).item()
#grad_field_dict2 = np.load('example10_qs_grad_fields2.npy', allow_pickle=True).item()
img_mask_dict1 = np.load('example10_qs_img_mask1.npy', allow_pickle=True).item()
#grad_field_dict1 = np.load('example10_qs_grad_fields1.npy', allow_pickle=True).item()
#all_gradient_fields12 = [x+y for x,y in zip(grad_field_dict1['g1'], grad_field_dict2['g2'])]
mask12 = np.logical_and(img_mask_dict1['m1'], img_mask_dict2['m2'])
img1, img2 = img_mask_dict1['img1'], img_mask_dict2['img2']
#del grad_field_dict1, grad_field_dict2

vel12 = compute_optical_flow(img1, img2, mask=mask12)#, error_fields=all_gradient_fields12)
vel12 = vel12 * pixel_in_meter / time_delay12
#del all_gradient_fields12


vel = vel01 + vel12 / np.maximum(mask01 + mask12, 1)[..., None]

np.save('example10_quicksave3', {'vel01':vel01,
                                 'vel12':vel12,
                                 'mask01':mask01,
                                 'mask12':mask12,
                                 'vel':vel})

vx, vy = vel[..., 0], vel[..., 1]


