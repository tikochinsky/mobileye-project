from skimage.feature import peak_local_max
import numpy as np
from scipy import signal as sg
import cv2
from PIL import Image


def create_kernel(length, radius=None):
    kernel = np.ones((length, length)) * -2 / (length ** 2)
    center = (int(length / 2), int(length / 2))
    if radius is None:
        radius = min(center[0], center[1], length - center[0], length - center[1]) * 0.5

    Y, X = np.ogrid[:length, :length]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    kernel[dist_from_center <= radius] = 20 / (length ** 2)
    return kernel


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    kernel = create_kernel(17, 3)
    red = c_image[:, :, 0]
    green = c_image[:, :, 1]
    grad_r = sg.convolve2d(red, kernel, mode='same')
    grad_g = sg.convolve2d(green, kernel, mode='same')
    coordinates_red = peak_local_max(grad_r, min_distance=20, num_peaks=5)
    coordinates_green = peak_local_max(grad_g, min_distance=20, num_peaks=5)
    x_red = coordinates_red[:, -1]
    y_red = coordinates_red[:, 0]
    x_green = coordinates_green[:, -1]
    y_green = coordinates_green[:, 0]
    return x_red, y_red, x_green, y_green


def run_and_print(c_image: np.ndarray):
    x_red, y_red, x_green, y_green = find_tfl_lights(c_image)

    for x, y in zip(x_red, y_red):
        cv2.circle(c_image, (x, y), 5, (255, 0, 0), -1)

    for x, y in zip(x_green, y_green):
        cv2.circle(c_image, (x, y), 5, (0, 255, 0), -1)

    Image.fromarray(c_image).show()


run_and_print(np.array(Image.open("../mock_data/aachen_000004_000019_leftImg8bit.png")))

