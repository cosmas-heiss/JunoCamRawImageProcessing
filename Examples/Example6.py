import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from PIL import Image
from mayavi import mlab
import cv2

import os
dir_path = os.path.dirname(__file__)

import sys
sys.path.append("..")
from core.Util import *
from core.Framelets import *
from core.JitterCorrection import *
from core.Vis3D import *
from core.ColorCorrection import *

# generating a specific position on the surface
pos = np.array([4756., 37588., -50225.])
pos[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
pos *= JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos)
pos[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
FRAMELET_RES = 256

image1 = dir_path + '/../RawImageData/PJ16/RawImages/JNCE_2018302_16C00028_V01-raw.png'
im_info1 = dir_path + '/../RawImageData/PJ16/RawImageData/JNCE_2018302_16C00028_V01.json'
image2 = dir_path + '/../RawImageData/PJ16/RawImages/JNCE_2018302_16C00030_V01-raw.png'
im_info2 = dir_path + '/../RawImageData/PJ16/RawImageData/JNCE_2018302_16C00030_V01.json'
raster = project_tangential_plane(pos, 20000, 20000, FRAMELET_RES, FRAMELET_RES, np.pi)

def get_image_and_data(image_path, im_info_path):
    img = Image.open(image_path)
    im_ar = np.array(img)
    im_ar = remove_bad_pixels(im_ar)

    with open(im_info_path, 'rb') as json_file:
        im_info_dict = json.load(json_file)

    start_time = im_info_dict["START_TIME"]
    frame_delay = float(im_info_dict["INTERFRAME_DELAY"].split()[0]) + 0.001

    start_correction, frame_delay = correct_image_start_time_and_frame_delay(im_ar, start_time, frame_delay)

    framelets = generate_framelets(revert_square_root_encoding(im_ar), start_time, start_correction, frame_delay)

    colors = np.zeros((FRAMELET_RES, FRAMELET_RES, 3))
    color_counts = np.zeros((FRAMELET_RES, FRAMELET_RES, 3))
    for k, framelet in enumerate(framelets):
        print_progress_bar(k+1, len(framelets),
                           'Processing framelet {} of {}:'.format(k + 1, len(framelets)),
                           length=18)

        brightnesses, valid = framelet.get_pixel_val_at_surf_point(raster, sun_brightness_correction=True)

        colors[..., 2 - framelet.color] += brightnesses
        color_counts[..., 2 - framelet.color] += valid
    colors = colors / np.maximum(color_counts, 1)

    return colors

img1 = get_image_and_data(image1, im_info1)
img2 = get_image_and_data(image2, im_info2)
img1 = (255 * img1 / np.max(img1)).astype(np.uint8)
img2 = (255 * img2 / np.max(img2)).astype(np.uint8)

base_image = np.concatenate((img1, img2), axis=1)
Image.fromarray(base_image).save("Example6_pic1.png")

new_img1 = np.zeros(img1.shape)
new_img2 = np.zeros(img2.shape)
for color_channel in range(3):
    ch1, ch2 = img1[..., color_channel], img2[..., color_channel]
    ch1, ch2 = preprocess_images(ch1, ch2, np.ones(ch1.shape, dtype=np.bool))

    new_img1[..., color_channel], new_img2[..., color_channel] = ch1, ch2


max_val = max(np.max(new_img1), np.max(new_img2))
new_img1 = (255 * new_img1 / max_val).astype(np.uint8)
new_img2 = (255 * new_img2 / max_val).astype(np.uint8)

new_image = np.concatenate((new_img1, new_img2), axis=1)
Image.fromarray(new_image).save("Example6_pic2.png")
