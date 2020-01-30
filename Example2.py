import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from PIL import Image
from mayavi import mlab
import cv2

from Util import *

def set_axes_equal(ax, middle=None):
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	  ax: a matplotlib axis, e.g., as output from plt.gca().
	'''

	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	if middle is None:
		x_middle = np.mean(x_limits)
		y_middle = np.mean(y_limits)
		z_middle = np.mean(z_limits)
	else:
		x_middle = middle[0]
		y_middle = middle[1]
		z_middle = middle[2]

	x_range = max(x_limits[0]-x_middle, x_limits[1]-x_middle)
	y_range = max(y_limits[0]-y_middle, y_limits[1]-y_middle)
	z_range = max(z_limits[0]-z_middle, z_limits[1]-z_middle)


	# The plot bounding box is a sphere in the sense of the infinity
	# norm, hence I call half the max range the plot radius.
	plot_radius = max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

start_time = spice.str2et("2019-05-29T07:31:51.385")
end_time = spice.str2et("2019-05-29T17:15:21.623")

positions, lightTimes = spice.spkpos('Juno', list(np.linspace(start_time,end_time,50)), 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
juno_orient = np.array([spice.pxform("IAU_JUPITER", "JUNO_JUNOCAM_CUBE", t) for t in list(np.linspace(start_time,end_time,50))])
pos_array = np.array(positions)

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = JUPITER_EQUATORIAL_RADIUS * np.outer(np.cos(u), np.sin(v))
y = JUPITER_EQUATORIAL_RADIUS * np.outer(np.sin(u), np.sin(v))
z = JUPITER_POLAR_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='orange')

ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])
ax.quiver(pos_array[:, 0],
	pos_array[:, 1],
	pos_array[:, 2],
	juno_orient[:, 0, 0],
	juno_orient[:, 0, 1],
	juno_orient[:, 0, 2],
	color='red', length=40000, normalize=True)

ax.quiver(pos_array[:, 0],
	pos_array[:, 1],
	pos_array[:, 2],
	juno_orient[:, 1, 0],
	juno_orient[:, 1, 1],
	juno_orient[:, 1, 2],
	color='green', length=40000, normalize=True)

ax.quiver(pos_array[:, 0],
	pos_array[:, 1],
	pos_array[:, 2],
	juno_orient[:, 2, 0],
	juno_orient[:, 2, 1],
	juno_orient[:, 2, 2],
	color='blue', length=40000, normalize=True)

set_axes_equal(ax)

plt.show()