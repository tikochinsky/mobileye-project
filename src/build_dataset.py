import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from typing import List
import os
from find_tfl_lights import find_tfl_lights
from image_data import ImagesData


data_root_path = "data_dir/train"


def insert_to_data_set(image, label):
    with open(f"{data_root_path}/data.bin", "ab") as data_file:
        np.array(image, dtype=np.uint8).tofile(data_file)
    with open(f"{data_root_path}/labels.bin", "ab") as labels_file:
        labels_file.write(label.to_bytes(1, byteorder='big', signed=False))


def read_img_and_label(index, crop_size):
    image = np.memmap(f"{data_root_path}/data.bin",  dtype=np.uint8, mode='r', shape=crop_size,
                      offset=crop_size[0]*crop_size[1]*crop_size[2]*index)
    label = np.memmap(f"{data_root_path}/labels.bin", dtype=np.uint8, mode='r', shape=(1,), offset=index)

    return image, label


def is_tfl_divide(x_axis: np.ndarray, y_axis: np.ndarray, label: np.ndarray):
    x_true, y_true, x_false, y_false = [], [], [], []
    for x, y in zip(x_axis, y_axis):
        if label[y, x] == 19:
            x_true.append(x)
            y_true.append(y)
        else:
            x_false.append(x)
            y_false.append(y)

    return x_true, y_true, x_false, y_false


def crop_image(img: np.ndarray, x: int, y: int):
    crop_size = 40
    h, w, d = img.shape

    padded_img = np.zeros((crop_size*2+h, crop_size*2+w, 3), dtype=np.uint8)
    padded_img[crop_size: -crop_size, crop_size: -crop_size] = img

    cropped = padded_img[y: y+crop_size*2+1, x: x+crop_size*2+1]
    return cropped


def crop_and_set(x_axis, y_axis, label, img):
    for x, y in zip(x_axis, y_axis):
        cropped_img = crop_image(img, x, y)
        insert_to_data_set(cropped_img, label)


def build_dataset():
    for city in os.listdir("../data/leftImg8bit/train"):
        print(city)
        count = 1
        for img, label in ImagesData.images_and_labeled(city):
            print(count)
            x_red, y_red, x_green, y_green = find_tfl_lights(np.array(img))
            x = np.append(x_red, x_green)
            y = np.append(y_red, y_green)
            x_true, y_true, x_false, y_false = is_tfl_divide(x, y, np.array(label))

            min_size = min(len(x_true), len(x_false))

            crop_and_set(x_true[:min_size], y_true[:min_size], 1, np.array(img))
            crop_and_set(x_false[:min_size], y_false[:min_size], 0, np.array(img))
            count += 1


build_dataset()


# im = np.array(Image.open("dusseldorf_000010_000019_leftImg8bit.png"), dtype=np.uint8)
#
# x = 1028
# y = 76
# cropped = crop_image(im, x, y)
# Image.fromarray(cropped).show()
# insert_to_data_set(cropped, 1)
# read_img_and_label(0, (81, 81, 3))


# # test and meta
# def show_lights(img, x_red, y_red, x_green, y_green):
#     for x, y in zip(x_red, y_red):
#         cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
#
#     for x, y in zip(x_green, y_green):
#         cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
#
#     Image.fromarray(img).show()
