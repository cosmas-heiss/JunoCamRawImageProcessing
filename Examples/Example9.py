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


pos = np.array([0.2,0.6,-0.5])
pos *= JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos)
pos[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
FRAMELET_RES = 512

image1 = dir_path + '/../RawImageData/PJ16/RawImages/JNCE_2018302_16C00024_V01-raw.png'
im_info1 = dir_path + '/../RawImageData/PJ16/RawImageData/JNCE_2018302_16C00024_V01.json'
image2 = dir_path + '/../RawImageData/PJ16/RawImages/JNCE_2018302_16C00025_V01-raw.png'
im_info2 = dir_path + '/../RawImageData/PJ16/RawImageData/JNCE_2018302_16C00025_V01.json'
raster = project_tangential_plane(pos, 20000, 20000, FRAMELET_RES, FRAMELET_RES, np.pi)

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
    colors = np.zeros((FRAMELET_RES, FRAMELET_RES, 3))
    color_counts = np.zeros((FRAMELET_RES, FRAMELET_RES, 3))
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

    return colors, gradient_fields


img1, gradient_fields1 = get_image_gradients_and_data(image1, im_info1)
img2, gradient_fields2 = get_image_gradients_and_data(image2, im_info2)
all_gradient_fields = [x+y for x,y in zip(gradient_fields1, gradient_fields2)]

with open(im_info1, 'rb') as json_file:
    im_info_dict = json.load(json_file)
    second_count1 = int(im_info_dict["SPACECRAFT_CLOCK_START_COUNT"].split(':')[0])
with open(im_info2, 'rb') as json_file:
    im_info_dict = json.load(json_file)
    second_count2 = int(im_info_dict["SPACECRAFT_CLOCK_START_COUNT"].split(':')[0])
time_delay = second_count2 - second_count1
pixel_in_meter = 20000000 / FRAMELET_RES

vel = compute_optical_flow(img1, img2, error_fields=all_gradient_fields) * pixel_in_meter / time_delay
vx, vy = vel[..., 0], vel[..., 1]

x_ind, y_ind = np.indices((FRAMELET_RES, FRAMELET_RES))
plt.figure(figsize=(16,16))
plt.xlim((0,FRAMELET_RES))
plt.ylim((FRAMELET_RES,0))
plt.streamplot(y_ind, x_ind, vy, vx, linewidth=1.5, color=np.sqrt(vx ** 2 + vy ** 2), cmap='plasma', density=6)
clb = plt.colorbar()
clb.ax.set_title('m/s')
plt.savefig('Optical_flow_corrected.png')