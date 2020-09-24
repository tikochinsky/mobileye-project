import numpy as np


def crop_image(img: np.ndarray, x: int, y: int):
    crop_size = 40
    h, w, d = img.shape

    padded_img = np.zeros((crop_size*2+h, crop_size*2+w, 3), dtype=np.uint8)
    padded_img[crop_size: -crop_size, crop_size: -crop_size] = img

    cropped = padded_img[y: y+crop_size*2+1, x: x+crop_size*2+1]
    return cropped
