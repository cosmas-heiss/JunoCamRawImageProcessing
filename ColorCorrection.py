import numpy as np

def revert_square_root_encoding(raw_image):
    return raw_image.astype(np.int) ** 2