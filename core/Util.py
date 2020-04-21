import numpy as np
import spiceypy as spice
from PIL import Image
print(spice.tkvrsn("TOOLKIT"))
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
spice_data_path = dir_path+"/../SPICEdata/"
KERNEL_LIST = [spice_data_path + x for x in os.listdir(spice_data_path)]
spice.furnsh(KERNEL_LIST)

JUPITER_EQUATORIAL_RADIUS = 71492  # km
JUPITER_POLAR_RADIUS = 66854  # km
STRIPE_LENGTH = 1648  # px
STRIPE_HEIGHT = 128  # px
STANDARD_IMAGE_START_TIME_OFFSET = 0.06188  # sec

PHOTOACTIVE_PIXELS = np.ones((STRIPE_LENGTH, STRIPE_HEIGHT), dtype=np.int)
PHOTOACTIVE_PIXELS[:23+1,:] = 0
PHOTOACTIVE_PIXELS[1631-1:,:] = 0

MIN_SURFACE_ANGLE = 15*np.pi/180
MIN_SUN_ANGLE = 5*np.pi/180

STANDARD_DATASET_SAMPLING_MAP_SIZE = (1024, 2048)

BAD_PIXEL_MASK = (np.array(Image.open(dir_path+"/BAD_PIXEL_MASK.png"))==0).astype(np.uint8)

# camera distortion parameters taken from
# https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/ik/juno_junocam_v03.ti
# in lists for each color: 0: blue, 1: green, 2: red
# INS-6150#_DISTORTION_X
cx = [814.21, 814.21, 814.21]
# INS-6150#_DISTORTION_Y
cy = [158.48, 3.48, -151.52]
# INS-6150#_DISTORTION_K1
k1 = [-5.9624209455667325e-08, -5.9624209455667325e-08, -5.9624209455667325e-08]
# INS-6150#_DISTORTION_K2
k2 = [2.7381910042256151e-14, 2.7381910042256151e-14, 2.7381910042256151e-14]
# INS-6150#_FOCAL_LENGTH/INS-6150#_PIXEL_SIZE
fl = [10.95637/0.0074, 10.95637/0.0074, 10.95637/0.0074]


def get_junocam_jupiter_rel_pos_orient(time_str, add_seconds=0):
	"""
	:param time_str: str (time stamp from image specific json data file)
	:param add_seconds: int or float (seconds to add to time specified by time str for conveniency)
	:return: ndarray(3), position
		ndarray(3,3), orientation matrix
	"""
	et = spice.str2et(time_str)+add_seconds
	pos, light_time = spice.spkpos("Juno", [et], 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
	pos = np.array(pos[0])

	orient = spice.pxform("IAU_JUPITER", "JUNO_JUNOCAM", et)
	orient = np.array(orient)  # orient[0, :] is x-vec, orient[1, :] y-vec, orient[2, :] z-vec

	return pos, orient

def get_sun_jupiter_rel_pos(time_str, add_seconds=0):
	et = spice.str2et(time_str) + add_seconds
	pos, light_time = spice.spkpos("SUN", [et], 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
	return np.array(pos[0])


def undistort(cam_x, cam_y, color):
	# undistort function from juno_junocam_v03.ti changed for using numpy arrays
	cam_x_old = cam_x.copy()
	cam_y_old = cam_y.copy()
	for i in range(5):  # fixed number of iterations for simplicity
		r2 = (cam_x**2+cam_y**2)
		dr = 1+k1[color]*r2+k2[color]*r2*r2
		cam_x = cam_x_old/dr
		cam_y = cam_y_old/dr
	return cam_x, cam_y


def get_lightray_vectors(pixel_x, pixel_y, color):
	"""
	:param pixel_x: nd-ndarray of x pixel coords
	:param pixel_y: nd-ndarray of y pixel coords
	:param color: int (0-2) specifying color 0:blue, 1:green, 2:red
	:return: (n+1)d-ndarray of float 3d-lightray vectors in camera reference frame (n+1)-dimension are x,y,z components
	"""
	cam_x = pixel_x-cx[color]
	cam_y = pixel_y-cy[color]
	cam_x, cam_y = undistort(cam_x, cam_y, color)
	return np.stack([cam_x, cam_y, np.full_like(cam_x, fl[color])], axis=-1)

def distort(cam, color):
	"""
	:param cam: ndarray(..,2)
	:param color: int, color 0:blue 1:green 2:red
	:return: ndarray(..,2), undistortet values
	"""
	r2 = np.sum(cam**2, axis=-1)
	dr = 1 + k1[color] * r2 + k2[color] * r2**2
	return cam * dr[...,None]

def get_pixel_coord_from_lightray(ray_dirs, color):
	behind_sensor = (ray_dirs[...,2] <= 0)
	ray_dirs[behind_sensor, 2] = 1
	alpha = ray_dirs[...,2] / fl[color]
	cam = ray_dirs[...,:2] / alpha[...,None]
	cam = distort(cam, color)
	cam[...,0] += cx[color]
	cam[...,1] += cy[color]
	return np.where(behind_sensor[...,None], -1, cam)


x, y = np.indices((STRIPE_LENGTH, STRIPE_HEIGHT))
# list of precomputed 3d-lighray vectors in camera reference frame for 3 colors 0:blue, 1:green, 2:red
CAMERA_STRIPE_VECTORS = [get_lightray_vectors(x, y, color).transpose(1, 0, 2) for color in range(3)]


def project_onto_jupiter_surf(pos, direct):
	"""
	:param pos: ndarray(..,3), origin of the rays
	:param direct: ndarray(..,3), direction of rays in whatever order
	:return: ndarray(..,3), position of projected points closest to pos
		ndarray(..), mask of if the ray hit the sphere
	"""
	b, a = JUPITER_POLAR_RADIUS, JUPITER_EQUATORIAL_RADIUS
	# equations where the line intersects the jupiter surface
	q1 = b**2*direct[..., 0]**2+b**2*direct[..., 1]**2+a**2*direct[..., 2]**2
	q2 = 2*pos[..., 0]*direct[..., 0]*b**2+2*pos[..., 1]*direct[..., 1]*b**2+2*pos[..., 2]*direct[..., 2]*a**2
	q3 = pos[..., 0]**2*b**2+pos[..., 1]**2*b**2+pos[..., 2]**2*a**2-float(a**2*b**2)

	p, q = q2/q1, q3/q1

	tmp = (0.5*p)**2-q
	mask = tmp >= 0

	s = -p*0.5-np.sqrt(tmp*mask)

	return (pos+s[..., None]*direct)*mask[..., None], mask


def rotate_around_axis(n, v, alpha):
	n /= np.linalg.norm(n)
	return n*n.dot(v) + np.cos(alpha)*np.cross(np.cross(n, v), n) + np.sin(alpha)*np.cross(n, v)


def project_tangential_plane(center, x_extent, y_extent, x_res, y_res, orientation):
	"""
	:param center: ndarray(3) of float, center of map on jupiter surface
	:param x_extent: float, extent of the x-axis of the map in km
	:param y_extent: float, extent of the y-axis of the map in km
	:param x_res: int, number of pixels on x-axis
	:param y_res: int, number of pixels on y-axis
	:param orientation: float, rotating orientation of map around center in radiants
	:return: ndarray(x_res, y_res, 3) 3d positions for map pixels
	"""
	assert np.sqrt(x_extent**2 + y_extent**2) < JUPITER_POLAR_RADIUS
	a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS

	normal_vector = np.array([center[0]*a, center[1]*a, center[2]/a])
	if center[0] == 0 and center[1] == 0:
		east_tang_vector = np.array([0, 1, 0])
	else:
		east_tang_vector = np.array([-center[1], center[0], 0])
		east_tang_vector /= np.linalg.norm(east_tang_vector)

	north_tang_vector = np.cross(normal_vector, east_tang_vector)
	north_tang_vector /= np.linalg.norm(north_tang_vector)

	north_tang_vector = rotate_around_axis(normal_vector, north_tang_vector, orientation)
	east_tang_vector = rotate_around_axis(normal_vector, east_tang_vector, orientation)

	x_raster, y_raster = np.meshgrid(np.linspace(-x_extent/2, x_extent/2, x_res),
									 np.linspace(-y_extent/2, y_extent/2, y_res))

	raster = center[None, None, :] + x_raster[..., None] * east_tang_vector[None, None, :]\
			 + y_raster[..., None] * north_tang_vector[None, None, :]


	points_on_jup, _ = project_onto_jupiter_surf(raster, -normal_vector[None, None, :])
	return points_on_jup


def convert_surf_pos_to_spherical_coords(surf_pos):
	surf_pos = surf_pos.copy()
	surf_pos[...,2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
	phi = np.arccos(surf_pos[...,2] / np.linalg.norm(surf_pos, axis=-1))
	theta = np.arctan2(surf_pos[...,1], surf_pos[...,0])
	return phi, theta


def print_progress_bar(i, n, text, channel=None, length=36):
	if channel is None:
		percentage = 100 * i / n
		bar_progress = int(length * i / n)
		bar = bar_progress * '█' + (length - bar_progress) * '.'
	else:
		percentage = 100 * (channel * n + i) / (3 * n)
		c_blocks = ['B', 'G', 'R']
		full_bar = c_blocks[0] * (length//3) + c_blocks[1] * (length//3) + c_blocks[2] * (length-2*(length//3))
		bar_progress = int(length * (channel * n + i) / (3 * n))
		bar = full_bar[:bar_progress] + (length - bar_progress) * '.'

	print('\r'+text+' |{}| {:.2f}%'.format(bar, percentage),end='\r')
	if i == n and channel != 0 and channel != 1:
		print()


import cv2
def remove_bad_pixels(raw_image):
	s1, s2 = raw_image.shape
	new_raw_image = np.zeros_like(raw_image)
	for k in range(s1//384):
		new_raw_image[k*384:(k+1)*384] = cv2.inpaint(raw_image[k*384:(k+1)*384], BAD_PIXEL_MASK, 3, cv2.INPAINT_NS)
	return new_raw_image