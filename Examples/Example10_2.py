import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from PIL import Image
from mayavi import mlab
import cv2
import imageio

import sys
sys.path.append("..")
from core.Util import *
from core.Framelets import *
from core.JitterCorrection import *
from core.Vis3D import *
from core.ColorCorrection import *

FRAMELET_RES_X = 2048
FRAMELET_RES_Y = 1024

img1_dict = np.load('example10_qs_img_mask1.npy', allow_pickle=True).item()
base_img, base_mask = img1_dict['img1'], img1_dict['m1']
img0_dict = np.load('example10_qs_img_mask0.npy', allow_pickle=True).item()
extra_image, extra_mask = img0_dict['img0'], img0_dict['m0']


base_mask = cv2.erode(base_mask.astype(np.float32), np.ones((121,121)))
base_mask = cv2.GaussianBlur(base_mask, (121,121), sigmaX=50, sigmaY=50)
extra_mask = np.maximum(extra_mask-base_mask, 0)

base_img = base_img * base_mask[..., None] + extra_image * extra_mask[..., None]


vel_dict = np.load('example10_quicksave3.npy', allow_pickle=True).item() # prominent velocity it 12 and secondary is 01
mask2, mask1 = vel_dict['mask01'], vel_dict['mask12']

mask1 = cv2.erode(mask1.astype(np.float32), np.ones((201,201)))
mask1 = cv2.GaussianBlur(mask1, (201,201), sigmaX=90, sigmaY=90)
mask2 = np.maximum(mask2-mask1, 0)

mask_sum = mask1 + mask2
mask_sum = np.where(mask_sum != 0, mask_sum, 1)


vel = (vel_dict['vel12'] * mask1[..., None] + vel_dict['vel01'] * mask2[..., None]) / mask_sum[..., None]
vx, vy = vel[..., 0], vel[..., 1]

x_ind, y_ind = np.indices((FRAMELET_RES_Y, FRAMELET_RES_X))
plt.figure(figsize=(36,16))
plt.imshow(base_img / np.max(base_img))
plt.xlim((0,FRAMELET_RES_X))
plt.ylim((FRAMELET_RES_Y,0))
plt.streamplot(y_ind, x_ind, vy, vx, linewidth=1.5, color=np.sqrt(vx ** 2 + vy ** 2), cmap='plasma', density=6)
clb = plt.colorbar()
clb.ax.set_title('m/s')
plt.savefig('Optical_flow_red_spot.png')


factor = 150 * 1024 / (20 * 10**6)
vx, vy = vx * factor, vy * factor


new_img = (base_img * 255 / np.max(base_img)).astype(np.uint8)
fade_frames = 20
new_vx = -vx.copy().astype(np.float32)
new_vy = -vy.copy().astype(np.float32)
images = [new_img]

for i in range(fade_frames + 40):
    tmp = cv2.remap(new_img,
                    (y_ind - new_vy).astype(np.float32),
                    (x_ind - new_vx).astype(np.float32),
                    cv2.INTER_LINEAR)
    images.append(tmp)
    new_vx += cv2.remap(-vx,
                       (y_ind + new_vy).astype(np.float32),
                       (x_ind + new_vx).astype(np.float32),
                       cv2.INTER_LINEAR)
    new_vy += cv2.remap(-vy,
                       (y_ind + new_vy).astype(np.float32),
                       (x_ind + new_vx).astype(np.float32),
                       cv2.INTER_LINEAR) 

images = images[::-1]
new_vx = vx.copy().astype(np.float32)
new_vy = vy.copy().astype(np.float32)

for i in range(41):
    tmp = cv2.remap(new_img,
                    (y_ind - new_vy).astype(np.float32),
                    (x_ind - new_vx).astype(np.float32),
                    cv2.INTER_LINEAR)
    images.append(tmp)
    new_vx += cv2.remap(vx,
                       (y_ind + new_vy).astype(np.float32),
                       (x_ind + new_vx).astype(np.float32),
                       cv2.INTER_LINEAR)
    new_vy += cv2.remap(vy,
                       (y_ind + new_vy).astype(np.float32),
                       (x_ind + new_vx).astype(np.float32),
                       cv2.INTER_LINEAR)

for i in range(1, fade_frames):
    ind = len(images) - fade_frames + i
    images[ind] = (i / fade_frames * images[i] + (1 - i / fade_frames) * images[ind]).astype(np.uint8)
images = images[fade_frames:]

for i in range(len(images)):
  images[i] = images[i][64:-64, 64:-64, :]
  #plt.imshow(images[i])
  #plt.show()

with imageio.get_writer('red_dot_movie.gif', mode='I') as writer:
    for image in images:
        writer.append_data(image)