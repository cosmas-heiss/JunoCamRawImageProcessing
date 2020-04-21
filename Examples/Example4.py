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


image = dir_path + '/../RawImageData/PJ20/RawImages/JNCE_2019149_20C00048_V01-raw.png'
im_info = dir_path + '/../RawImageData/PJ20/RawImageData/JNCE_2019149_20C00048_V01.json'
with open(im_info, 'rb') as json_file:
    im_info_dir = json.load(json_file)

img = Image.open(image)
im_ar = np.array(img)
im_ar = remove_bad_pixels(im_ar)

s1, s2 = im_ar.shape

mask1 = get_raw_image_mask(im_ar)

start_time = im_info_dir["START_TIME"]
frame_delay = float(im_info_dir["INTERFRAME_DELAY"].split()[0])+0.001

start_correction, frame_delay = correct_image_start_time_and_frame_delay(im_ar, start_time, frame_delay)

framelets = generate_framelets(revert_square_root_encoding(im_ar), start_time, start_correction, frame_delay)

visualize_framelets_with_mayavi(framelets, 1024, 2048)