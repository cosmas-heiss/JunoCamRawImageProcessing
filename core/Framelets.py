import numpy as np
from scipy.interpolate import RectBivariateSpline

from core.Util import *
from core.OpticalFlow import *


class CamFramelet:
    def __init__(self, cam_pos, cam_orient, sun_pos, pixel_val, color):
        # pixel_val is expected to be squared compared to the raw_image
        self.color = color
        self.img = pixel_val.T
        self.cam_pos = cam_pos
        self.cam_orient = cam_orient
        self.inv_cam_orient = np.linalg.inv(cam_orient)
        self.sun_pos = sun_pos

        self.interp_function = RectBivariateSpline(np.arange(STRIPE_LENGTH), np.arange(STRIPE_HEIGHT), self.img)

    @staticmethod
    def get_photoactive_pixels(pixel_coords):
        int_px_coords = np.rint(pixel_coords).astype(np.int)
        outside_stripe_mask = (int_px_coords[...,0] < STRIPE_LENGTH) & (int_px_coords[...,0] >= 0)
        outside_stripe_mask &= (int_px_coords[...,1] < STRIPE_HEIGHT) & (int_px_coords[...,1] >= 0)
        int_px_coords *= outside_stripe_mask[...,None]
        return PHOTOACTIVE_PIXELS[int_px_coords[...,0], int_px_coords[...,1]] * outside_stripe_mask

    def get_valid_angle_pixels(self, surf_pos):
        a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
        surf_normal = surf_pos.copy()
        surf_normal[...,0:2] *= a
        surf_normal[...,2] /= a

        rays = surf_pos - self.cam_pos

        scalar_prod = np.sum(rays*surf_normal, axis=-1)
        surf_normal_norm = np.linalg.norm(surf_normal, axis=-1)
        cos_angle = -scalar_prod/(np.linalg.norm(rays, axis=-1)*np.where(surf_normal_norm != 0, surf_normal_norm, 1))

        return cos_angle > np.cos(np.pi/2-MIN_SURFACE_ANGLE)

    def get_sun_correction(self, surf_pos):
        a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
        surf_normal = surf_pos.copy()
        surf_normal[..., 0:2] *= a
        surf_normal[..., 2] /= a

        sun_rays = surf_pos - self.sun_pos

        scalar_prod = np.sum(sun_rays * surf_normal, axis=-1)
        surf_normal_norm = np.linalg.norm(surf_normal, axis=-1)
        cos_angle = -scalar_prod / (np.linalg.norm(sun_rays, axis=-1) * np.where(surf_normal_norm != 0, surf_normal_norm, 1))
        mask = cos_angle > np.cos(np.pi/2-MIN_SUN_ANGLE)

        return 1 / np.where(mask, cos_angle, 1), mask

    def get_pixel_val_at_surf_point(self, pos, sun_brightness_correction=True, return_y_indices=False):
        """
        :param pos: ndarray(..,3), positions of jupiter surface points in 3d space
        :param sun_brightness_correction: if brightness should be corrected for sun illumination
        :param return_y_indices: if True, an additional return are the y coords in pixel space
        :return: ndarray(..) of float, brightness values for these points, val is set to 0 if not part of usable sensor
                 ndarray(..) of bool, mask of which surface positions are actually recorded by the sensor
        """
        rays = pos-self.cam_pos
        rays = rays.dot(self.inv_cam_orient)
        pixel_coords = get_pixel_coord_from_lightray(rays, self.color)
        valid = self.get_photoactive_pixels(pixel_coords)
        valid *= self.get_valid_angle_pixels(pos)

        if sun_brightness_correction:
            sun_correction, sun_angle_mask = self.get_sun_correction(pos)
            valid *= sun_angle_mask
        else:
            sun_correction = 1

        in_shape = pos.shape[:-1]
        pixel_coords_flattened = pixel_coords.reshape(-1,2)
        out_val = self.interp_function(pixel_coords_flattened[...,0], pixel_coords_flattened[...,1], grid=False)
        out_val = np.maximum(out_val, 0) # need to do that because the spline interpol sometimes goes negative

        if return_y_indices:
            return out_val.reshape(in_shape) * sun_correction * valid,\
                   pixel_coords[...,1],\
                   valid.astype(np.bool)
        return out_val.reshape(in_shape) * sun_correction * valid, valid.astype(np.bool)


    def get_px_per_km_on_surf_points(self, pos, eps=1e-6):
        """
        :param pos:  ndarray(n,m,3), positions of jupiter surface points in 3d space
        :return: float, minimum point distance in pixel space
        """
        rays = pos - self.cam_pos
        rays = rays.dot(self.inv_cam_orient)
        pixel_coords = get_pixel_coord_from_lightray(rays, self.color)
        valid = self.get_photoactive_pixels(pixel_coords)
        valid *= self.get_valid_angle_pixels(pos)
        _, sun_angle_mask = self.get_sun_correction(pos)
        valid *= sun_angle_mask
        valid = valid.astype(np.bool)

        px_grad_x = np.gradient(pixel_coords[:,:,0], axis=0)
        px_grad_y = np.gradient(pixel_coords[:,:,1], axis=1)

        km_grad_x = np.linalg.norm(np.gradient(pos, axis=0), axis=-1)
        km_grad_y = np.linalg.norm(np.gradient(pos, axis=1), axis=-1)

        km_grad_x = np.where(km_grad_x != 0,km_grad_x,eps)
        km_grad_y = np.where(km_grad_y != 0, km_grad_y, eps)

        return np.sqrt((px_grad_x/km_grad_x)**2+(px_grad_y/km_grad_y)**2) * valid, valid





def generate_framelets(image_array, start_time, start_correction, frame_delay):
    """
    :param image_array: ndarray(k*384,1648), raw image array
    :param start_time: float, start of imaging in et-J2000 time
    :param start_correction: float, correction of start of image time
    :param frame_delay: float, seconds between framelets
    :return:
    """
    framelets = []
    s1, s2 = image_array.shape
    for k in range(s1 // 384):
        stripe_delay = start_correction + k * frame_delay
        cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(start_time, add_seconds=stripe_delay)
        sun_pos = get_sun_jupiter_rel_pos(start_time, add_seconds=stripe_delay)

        for color in range(3):
            stripe = image_array[k*384+color*128:k*384+(color+1)*128]

            new_framelet = CamFramelet(cam_pos,
                                       cam_orient,
                                       sun_pos,
                                       stripe,
                                       color)

            framelets.append(new_framelet)

    return framelets



