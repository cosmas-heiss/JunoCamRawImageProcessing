import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab
from tvtk.api import tvtk

from Util import *

def visualize_framelets_with_mayavi(framelet_list, lat_res, lon_res):
    phi, theta = np.mgrid[0:np.pi:lat_res * 1j, 0:2 * np.pi:lon_res * 1j]
    x = JUPITER_EQUATORIAL_RADIUS * np.sin(phi) * np.cos(theta)
    y = JUPITER_EQUATORIAL_RADIUS * np.sin(phi) * np.sin(theta)
    z = JUPITER_POLAR_RADIUS * np.cos(phi)
    coords = np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)
    colors = np.zeros((lat_res, lon_res, 3))
    color_counts = np.zeros((lat_res, lon_res, 3))

    for k,framelet in enumerate(framelet_list):
        print('Processing framelet {} of {}..'.format(k+1, len(framelet_list)))
        col = framelet.color
        brightnesses, valid_map = framelet.get_pixel_val_at_surf_point(coords)
        colors[...,2-col] += brightnesses
        color_counts[...,2-col] += valid_map

    colors /= np.maximum(color_counts, 1)
    colors *= 255 / np.max(colors)

    unique_colors, unique_indices = np.unique(colors.reshape(-1,3), return_inverse=True, axis=0)
    unique_colors = np.concatenate([unique_colors, 255*np.ones((unique_colors.shape[0],1))], axis=-1)
    unique_colors = unique_colors.astype(np.uint8)
    unique_indices = unique_indices.reshape(colors.shape[:-1])
    if unique_colors.shape[0] == 1:
        unique_colors = unique_colors.repeat(2, axis=0)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
    mlab.clf()
    mesh = mlab.mesh(x, y, z, scalars=unique_indices)
    mesh.module_manager.scalar_lut_manager.lut.table = unique_colors
    mlab.show()
