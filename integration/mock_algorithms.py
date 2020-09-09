import random
import numpy as np


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    x_red = [random.randint(0, 1024) for i in range(10)]
    y_red = [random.randint(0, 2048) for i in range(10)]
    x_green = [random.randint(0, 1024) for i in range(5)]
    y_green = [random.randint(0, 2048) for i in range(5)]
    return x_red, y_red, x_green, y_green


def is_tfl(frame, candidate):
    val = random.random()
    # if val < 0.5:
    #     return False
    return True


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    curr_container.traffic_lights_3d_location = [[i[0], i[1], random.randint(0, 100)] for i in curr_container.traffic_light]
    return curr_container
