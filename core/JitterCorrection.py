import numpy as np
import json
from PIL import Image
import cv2  # only needed in detect_sharp_edges

from core.Util import *

BLUE_JUPITER_THRESHOLD = 35
GREEN_JUPITER_THRESHOLD = 75
RED_JUPITER_THRESHOLD = 75

LAPLACIAN_EDGE_DETECTION_THRESHOLD = 0.1


def get_raw_image_mask(raw_image_array):
    """
    :param raw_image_array: ndarray(k*384, 1648), array of image data from the raw image
    :return: ndarray of uint8, mask of where in the picture jupiter actually is for alignment
    """
    s1, s2 = raw_image_array.shape
    mask = np.zeros((s1, s2), dtype=np.uint8)
    for k in range(0, s1, 384):
        mask[k:k + 128] = (raw_image_array[k:k + 128] > BLUE_JUPITER_THRESHOLD)
        mask[k + 128:k + 256] = (raw_image_array[k + 128:k + 256] > GREEN_JUPITER_THRESHOLD)
        mask[k + 256:k + 384] = (raw_image_array[k + 256:k + 384] > RED_JUPITER_THRESHOLD)
    return mask


def detect_valid_edge_pixels(stripe):
    """
    :param stripe: ndarray(128, 1648), stripe mask
    :return: ndarray(1648) of int, edge indices
             ndarray(1648) of int, mask of which pixels can be used for edge alignment (no noise clear edge)
             ndarray(1648) of int, direction of edge, 1 for edge in up-direction, -1 for edge in down-direction
    """
    stripe = stripe.astype(np.int8)
    y_coords = np.arange(128)
    # detecting the edge
    edge = np.argmax(np.abs(stripe[1:, :]-stripe[:-1, :]), axis=0)
    edge_dir = (stripe[1:, :]-stripe[:-1, :])[edge, np.arange(STRIPE_LENGTH)]

    # finding the indices where values above edge and below edge are all the same by first filling up the rest
    # and the using np.all to detect if the not-filled up part is uniform
    valid_up = np.where(y_coords[:, None] <= edge[None, :], stripe, stripe[None, 0, :])
    valid_down = np.where(y_coords[:, None] > edge[None, :], stripe, stripe[None, -1, :])

    # valid are indices where there are different pixels at the top and bottom
    valid = stripe[None, 0, :] != stripe[None, -1, :]
    valid_up = np.all(valid_up == valid_up[None, 0, :], axis=0)
    valid_down = np.all(valid_down == valid_down[None, 0, :], axis=0)

    return edge, valid_up * valid_down * valid, edge_dir


def detect_sharp_edges(edge, image_stripe, color):
    """
    :param edge: ndarray(1648) of int, edge indices as retuned from detect_valid_edge_pixels
    :param image_stripe: ndarray(128, 1648), stripe of the raw image
    :param color: int (0-2), color of stripe introduced for different edge thresholding for the different colors
                            but not needed yet
    :return: ndarray(1648) of bool, mask on x-dimension similar to second return from detect_valid_edge_pixels
    """
    edge_sharpness = cv2.Laplacian(image_stripe, cv2.CV_32F, ksize=7) / (255 * 49)
    edge_sharpness = cv2.blur(np.abs(edge_sharpness), ksize=(7, 7))

    return edge_sharpness[edge, np.arange(STRIPE_LENGTH)] > LAPLACIAN_EDGE_DETECTION_THRESHOLD


def detect_mask_offset(raw_image_mask, jupiter_mask, raw_image):
    """
    :param raw_image_mask: ndarray of uint8, mask retuned by get_jupiter_mask on the image
    :param jupiter_mask: ndarray of uint8, jupiter mask constructed by position and orientation of spaceprobe
    :return: float, start time offset in seconds for correcting jitter
    """
    s1, s2 = raw_image_mask.shape
    assert jupiter_mask.shape == (s1, s2)

    offset_sum = 0
    offset_count = 0
    for color, k in enumerate(range(0, s1, 128)):
        raw_image_mask_stripe = raw_image_mask[k:k + 128]
        jupiter_mask_stripe = jupiter_mask[k:k + 128]

        jupiter_mask_edge, jupiter_mask_valid, jupiter_mask_egde_dir = detect_valid_edge_pixels(jupiter_mask_stripe)
        raw_image_mask_edge, raw_image_mask_valid, raw_image_egde_dir = detect_valid_edge_pixels(raw_image_mask_stripe)
        sharp_valid = detect_sharp_edges(raw_image_mask_edge, raw_image[k:k + 128], color % 3)

        valid = jupiter_mask_valid * raw_image_mask_valid * (jupiter_mask_egde_dir == raw_image_egde_dir) * sharp_valid
        offset_sum += np.sum((jupiter_mask_edge-raw_image_mask_edge)*valid)
        offset_count += np.sum(valid)

    return offset_sum, offset_count


def correct_image_start_time_and_frame_delay(raw_image_array, raw_image_time_stamp, frame_delay):
    time_offset = STANDARD_IMAGE_START_TIME_OFFSET

    s1, s2 = raw_image_array.shape
    raw_image_mask = get_raw_image_mask(raw_image_array)
    direction_array = np.concatenate(CAMERA_STRIPE_VECTORS, axis=0)

    for i in range(2):
        pixel_offset = 0
        pixel_offset_count = 0
        frame_wise_offsets = []

        for k in range(s1//384):
            cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(raw_image_time_stamp,
                                                                     add_seconds=time_offset + k * frame_delay)
            direction_array_new = direction_array.dot(cam_orient)
            _, jupiter_mask = project_onto_jupiter_surf(cam_pos, direction_array_new)
            offset_sum, offset_count = detect_mask_offset(raw_image_mask[k * 384:(k+1) * 384],
                                                          jupiter_mask,
                                                          raw_image_array[k * 384:(k+1) * 384])
            pixel_offset += offset_sum / max(offset_count, 1)
            pixel_offset_count += (offset_count > 0)
            if offset_count != 0:
                frame_wise_offsets.append((k, offset_sum/offset_count))
        # 2rpm = 12deg/sec, approx 5deg per stripe and 128px per stripe
        time_offset += 5/(12*128)*pixel_offset/pixel_offset_count

        # frame delay is changed by mean change in offset per stripe
        tmp = [5/(12*128)*(y[1]-x[1])/(y[0]-x[0]) for x, y in zip(frame_wise_offsets[:-1], frame_wise_offsets[1:])]
        frame_delay += sum(tmp) / len(tmp)
    return time_offset, frame_delay


def correct_image_start_time(raw_image_array, raw_image_time_stamp, frame_delay):
    time_offset = STANDARD_IMAGE_START_TIME_OFFSET

    s1, s2 = raw_image_array.shape
    raw_image_mask = get_raw_image_mask(raw_image_array)
    direction_array = np.concatenate(CAMERA_STRIPE_VECTORS, axis=0)

    for i in range(2):
        pixel_offset = 0
        pixel_offset_count = 0

        for k in range(s1//384):
            cam_pos, cam_orient = get_junocam_jupiter_rel_pos_orient(raw_image_time_stamp,
                                                                     add_seconds=time_offset + k * frame_delay)
            direction_array_new = direction_array.dot(cam_orient)
            _, jupiter_mask = project_onto_jupiter_surf(cam_pos, direction_array_new)
            offset_sum, offset_count = detect_mask_offset(raw_image_mask[k * 384:(k+1) * 384],
                                                          jupiter_mask,
                                                          raw_image_array[k * 384:(k+1) * 384])
            pixel_offset += offset_sum / max(offset_count, 1)
            pixel_offset_count += (offset_count > 0)
        # 2rpm = 12deg/sec, approx 5deg per stripe and 128px per stripe
        time_offset += 5/(12*128)*pixel_offset/pixel_offset_count
    return time_offset