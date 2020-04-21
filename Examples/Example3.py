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


USE_CORRECTION = True

image = dir_path + '/../RawImageData/PJ20/RawImages/JNCE_2019149_20C00048_V01-raw.png'
im_info = dir_path + '/../RawImageData/PJ20/RawImageData/JNCE_2019149_20C00048_V01.json'
with open(im_info, 'rb') as json_file:
    im_info_dir = json.load(json_file)

img = Image.open(image)
im_ar = np.array(img)
im_ar = remove_bad_pixels(im_ar)
plt.figure('number2')
s1, s2 = im_ar.shape

mask1 = get_raw_image_mask(im_ar)

start_time = im_info_dir["START_TIME"]
frame_delay = float(im_info_dir["INTERFRAME_DELAY"].split()[0])+0.001

if USE_CORRECTION:
    start_correction, frame_delay = correct_image_start_time_and_frame_delay(im_ar, start_time, frame_delay)
else:
    start_correction = 0

framelets = generate_framelets(revert_square_root_encoding(im_ar), start_time, start_correction, frame_delay)

new_mask = np.zeros_like(mask1)
direction_array = np.concatenate(CAMERA_STRIPE_VECTORS, axis=0)
for k in range(s1//384):
    cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(start_time,
                                                             add_seconds=start_correction + k * frame_delay)
    direction_array_new = direction_array.dot(cam_orient)
    _, jupiter_mask = project_onto_jupiter_surf(cam_pos, direction_array_new)
    new_mask[k*384:(k+1)*384] = jupiter_mask

plt.imshow(new_mask+0.5*mask1)
plt.show()