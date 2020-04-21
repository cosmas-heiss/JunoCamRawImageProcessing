import numpy as np
import cv2

from core.Util import print_progress_bar

def get_lookup_table(cum_hist, reference_cum_hist):
    table = np.zeros(256)
    for i in range(256):
        table[i] = np.min((reference_cum_hist >= cum_hist[i]).nonzero()[0])
    return table


def unify_histograms(img1, img2, mask):
    cum_hist1 = np.cumsum(np.histogram(img1.ravel()[mask.ravel()], 256, [0, 256])[0])
    cum_hist2 = np.cumsum(np.histogram(img2.ravel()[mask.ravel()], 256, [0, 256])[0])
    max_cum_hist = np.maximum(cum_hist1, cum_hist2)
    table1 = get_lookup_table(cum_hist1, max_cum_hist)
    table2 = get_lookup_table(cum_hist2, max_cum_hist)
    return table1[img1], table2[img2]


def preprocess_images(img1, img2, mask):
    # compresses the histograms of both images such that values are comparable even in differen lighting conditions
    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    mask_mean = np.sum(img1 + img2) / (2 * np.sum(mask))
    sigma = 20
    ksize = sigma * 4 + 1
    img1 = img1 - cv2.GaussianBlur(img1 + (1 - mask) * mask_mean, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) * mask
    img2 = img2 - cv2.GaussianBlur(img2 + (1 - mask) * mask_mean, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) * mask
    img1_min, img1_max = np.min(img1, where=mask, initial=0), np.max(img1, where=mask, initial=0)
    img2_min, img2_max = np.min(img2, where=mask, initial=0), np.max(img2, where=mask, initial=0)
    img1 = (255 * (img1 - img1_min) / (img1_max - img1_min)).astype(np.uint8)
    img2 = (255 * (img2 - img2_min) / (img2_max - img2_min)).astype(np.uint8)
    img1, img2 = unify_histograms(img1, img2, mask=mask)
    max_val = max(np.max(img1, where=mask, initial=0), np.max(img2, where=mask, initial=0))
    img1, img2 = img1.astype(np.float32) / max_val, img2.astype(np.float32) / max_val
    return img1, img2


def get_rotation_free_part(fx, fy, mask):
    # computes the rotation free part of the vector field using the fourier transform
    # because this decomposition is in a sense global, but the effect decreases inversely proportional to distance,
    # the values are more reliable in the 'middle' of the resulting field
    if not np.all(mask):
        fx = fx * mask + (1 - mask) * np.sum(fx * mask) / np.sum(mask)
        fy = fy * mask + (1 - mask) * np.sum(fy * mask) / np.sum(mask)

    s1, s2 = fx.shape
    xx, yy = np.indices((s1, s2), dtype=np.float32)
    xx = (xx + s1 - s1 // 2) % s1 - s1 + s1 // 2
    yy = (yy + s2 - s2 // 2) % s2 - s2 + s2 // 2
    gx = np.fft.fft2(fx)
    gy = np.fft.fft2(fy)
    norm_tmp = (xx ** 2 + yy ** 2)
    h = (gx * xx + gy * yy) / np.where(norm_tmp != 0, norm_tmp, 1)
    return np.real(np.fft.ifft2(xx * h)) * mask, np.real(np.fft.ifft2(yy * h)) * mask


def compute_grayscale_optical_flow(img1, img2, mask, sigma, num_iter, step_size, channel=None, orth_vec_fields=[]):
    """
    :param img1: ndarray(1024,1024), first non-distorting image of surface (extent 20000km x 20000km)
    :param img2: ndarray(1024,1024), second non-distorting image of surface (extent 20000km x 20000km)
    :param mask: ndarray(1024,1024) of bool, mask indicating valid regions in the image
    :param sigma: num,  standard deviation of smoothing gradually (like the evolution of the heat equation) applied to
                        the velocity field
    :param num_iter: int, number of iteration done for gradient descent
    :param step_size: float,    step size for the gradient descent (can be chosen quite high due to optimization of
                                absolute error and smoothing)
    :param channel: 0 or 1 or 2, channel of the image 0:blue, 1:green, 2:red only needed for progress bar ^^
    :param orth_vector_fields: list of pairwise orthogonal vector field to which the solution should be orthogonal to
    :return: ndarray(1024,1024,2), estimated optical flow velocity field (not corrected for time yet)
    """
    img1, img2 = img1 * mask, img2 * mask
    # equalize histograms
    img1, img2 = preprocess_images(img1, img2, mask)
    import matplotlib.pyplot as plt

    # computing sobel gradients with image mean subtracted such that mask edges are not that hard
    img2_mean = np.sum(img2 * mask) / np.sum(mask)
    dimg2_dy = cv2.Sobel(img2 - img2_mean, cv2.CV_32F, 1, 0, ksize=7, scale=2 ** (-7 * 2 + 10))
    dimg2_dx = cv2.Sobel(img2 - img2_mean, cv2.CV_32F, 0, 1, ksize=7, scale=2 ** (-7 * 2 + 10))

    # initializing velocity fields
    vx, vy = np.zeros_like(img1), np.zeros_like(img1)
    x_ind, y_ind = np.indices(img1.shape, dtype=np.float32)

    # setting up the smoothing term and the correction map for gaussian blur using the mask
    sigma_iter = np.sqrt(3/4) * sigma / np.sqrt(num_iter)
    ksize = 4 * int(np.ceil(sigma_iter)) + 1
    smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigmaX=sigma_iter, sigmaY=sigma_iter)
    smoothed_mask = np.where(smoothed_mask != 0, smoothed_mask, 1)

    for i in range(num_iter):
        # remap such to know corresponding image values and gradients in the second image
        img2_interp = cv2.remap(img2, y_ind + vy, x_ind + vx, cv2.INTER_LINEAR)
        dimg2_dx_interp = cv2.remap(dimg2_dx, y_ind + vy, x_ind + vx, cv2.INTER_LINEAR)
        dimg2_dy_interp = cv2.remap(dimg2_dy, y_ind + vy, x_ind + vx, cv2.INTER_LINEAR)

        # if the pixel value at the end of the remapping vector is lower than the pixel value from the first image,
        # the remapping (velocity) vector is changed in direction of the gradient to increase sayd value
        dvx = 2 * dimg2_dx_interp * np.sign(img1 - img2_interp) * mask
        dvy = 2 * dimg2_dy_interp * np.sign(img1 - img2_interp) * mask

        # gradient descent update
        vx += step_size * dvx
        vy += step_size * dvy

        # smooth by a little amount to in the have smoothed by sigma much
        vx = cv2.GaussianBlur(vx * mask, (ksize, ksize), sigmaX=sigma_iter, sigmaY=sigma_iter)
        vy = cv2.GaussianBlur(vy * mask, (ksize, ksize), sigmaX=sigma_iter, sigmaY=sigma_iter)

        # correct smoothing for mask
        vx = mask * vx / smoothed_mask
        vy = mask * vy / smoothed_mask

        # every 40 iterations, the irrotational part of the vector field * 0.8 is subtracted
        # is not done every iteration because its slow
        if (i + 1) % 40 == 0:
            if i + 1 < num_iter:
                wx, wy = get_rotation_free_part(vx, vy, mask)
                vx -= wx * 0.8
                vy -= wy * 0.8
            print_progress_bar(i + 1, num_iter, "Estimating velocity field:", channel=channel)

    for error_field in orth_vec_fields:
        error_field_norm = np.sum(error_field**2)
        scalar_prod = np.sum(vx*error_field[...,0]+vy*error_field[...,1]) /\
                      np.where(error_field_norm!=0, error_field_norm, 1)
        vx -= scalar_prod * error_field[...,0]
        vy -= scalar_prod * error_field[...,1]

    vx = cv2.GaussianBlur(vx * mask, (ksize, ksize), sigmaX=np.sqrt(1/8) * sigma, sigmaY=np.sqrt(1/8) * sigma)
    vy = cv2.GaussianBlur(vy * mask, (ksize, ksize), sigmaX=np.sqrt(1/8) * sigma, sigmaY=np.sqrt(1/8) * sigma)
    vx = mask * vx / smoothed_mask
    vy = mask * vy / smoothed_mask

    wx, wy = get_rotation_free_part(vx, vy, mask)
    vx -= wx * 0.8
    vy -= wy * 0.8

    vx = cv2.GaussianBlur(vx * mask, (ksize, ksize), sigmaX=np.sqrt(1/8) * sigma, sigmaY=np.sqrt(1/8) * sigma)
    vy = cv2.GaussianBlur(vy * mask, (ksize, ksize), sigmaX=np.sqrt(1/8) * sigma, sigmaY=np.sqrt(1/8) * sigma)
    vx = mask * vx / smoothed_mask
    vy = mask * vy / smoothed_mask

    return np.concatenate((vx[..., None], vy[..., None]), axis=-1)

def orthogonalize_error_vector_fields(vec_list):
    new_vec_list = [vec_list[0]]
    for i in range(1, len(vec_list)):
        sum_vec = np.zeros_like(vec_list[i])
        for v in new_vec_list:
            v_norm = np.sum(v**2)
            sum_vec += v*np.sum(v*vec_list[i])/np.where(v_norm!=0, v_norm, 1)
        new_vec_list.append(vec_list[i]-sum_vec)
    return new_vec_list


def compute_optical_flow(img1, img2, sigma=250, num_iter=2000, step_size=3e-1, mask=None, error_fields=[[],[],[]]):
    """
    image is expected to be 1024px x 1024px with extent 20000km x 20000km
    :param img1: ndarray(1024,1024,3), first non-distorting image of surface (extent 20000km x 20000km)
    :param img2: ndarray(1024,1024,3), second non-distorting image of surface (extent 20000km x 20000km)
    images do not have to be of that size and surface extent, but then the default parameters haveto be changed
    :param sigma: num,  standard deviation of smoothing gradually (like the evolution of the heat equation) applied to
                        the velocity field
    :param num_iter: int, number of iteration done for gradient descent
    :param step_size: float,    step size for the gradient descent (can be chosen quite high due to optimization of
                                absolute error and smoothing)
    :param mask: ndarray(1024,1024) of bool, mask indicating valid regions in the image
    :param error_fields: error field to which the wind field solution should be orthogonal to
    :return: ndarray(1024,1024,2), estimated optical flow velocity field (not corrected for time yet)
    """
    if mask is None:
        mask = np.ones(img1.shape[:2], dtype=np.bool)
    vel = np.zeros((*img1.shape[:2], 2))
    if not np.any(mask): return vel
    for channel in range(3):
        if error_fields[channel]:
            error_fields[channel] = orthogonalize_error_vector_fields(error_fields[channel])
    for channel in range(3):
        vel += compute_grayscale_optical_flow(img1[..., channel], img2[..., channel], mask,
                                              sigma, num_iter, step_size, channel=channel,
                                              orth_vec_fields=error_fields[channel])
    return vel / 3