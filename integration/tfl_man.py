import pickle
from frame_container import FrameContainer
from PIL import Image
import numpy as np
import mock_algorithms
import matplotlib.pyplot as plt
from attention.find_tfl_lights import find_tfl_lights
from distance.SFM import calc_TFL_dist
# from tensorflow.keras.models import load_model


class TFL_Man:
    def __init__(self, pkl_file, start_ind):

        with open(pkl_file, 'rb') as pkl_file:
            pkl_data = pickle.load(pkl_file, encoding='latin1')

        self.__egomotions = {k: v for k, v in pkl_data.items() if 'egomotion' == k[:9]}
        self.__pp = pkl_data['principle_point']
        self.__focal = pkl_data['flx']
        self.__prev_frame = None
        self.__index = start_ind
        # self.__model = load_model("model.h5")

    @staticmethod
    def __get_lights(frame):
        image = np.array(Image.open(frame))
        x_red, y_red, x_green, y_green = find_tfl_lights(image)

        return np.array(list(zip(np.concatenate([x_red, x_green]), np.concatenate([y_red, y_green])))),\
               np.array(['red' for i in range(len(x_red))] + ['green' for i in range(len(x_green))])

    def __get_tfl_lights(self, frame, candidates, auxiliary_c):
        # l_predicted_label = []
        # for candidate in candidates:
        #     crop_img = crop_image(np.array(plt.imread(frame)), candidate[0], candidate[1])
        #     predictions = self.__model.predict(crop_img)
        #     print(predictions)
        #     l_predicted_label.append(1 if predictions[0][1] > 0.98 else 0)
        # traffic_lights = [candidates[index] for index in range(len(candidates)) if l_predicted_label[index] == 1]
        # auxilary =  [auxilary[index] for index in range(len(auxilary)) if l_predicted_label[index] == 1]
        # return traffic_lights, auxilary

        traffic_lights = []
        auxiliary_t = []
        for candidate, color in zip(candidates, auxiliary_c):
            if mock_algorithms.is_tfl(frame, candidate):
                traffic_lights.append(candidate)
                auxiliary_t.append(color)

        return np.array(traffic_lights), np.array(auxiliary_t)

    def __get_distance(self, curr_frame):
        curr_frame = calc_TFL_dist(self.__prev_frame, curr_frame, self.__focal, self.__pp)
        return np.array(curr_frame.traffic_lights_3d_location)[:, 2]

    def __display(self, curr_frame, candidates, auxiliary_c, traffic_lights, auxiliary_t, distances):

        fig, (attention, traffic_light, distance) = plt.subplots(1, 3, figsize=(12, 6))
        attention.set_title('attention(' + str(self.__index) + ')')
        attention.imshow(curr_frame.img)
        red_candidates = candidates[auxiliary_c == 'red']
        green_candidates = candidates[auxiliary_c == 'green']
        attention.scatter(red_candidates[:, 0], red_candidates[:, 1], color='r', marker='o', s=5)
        attention.scatter(green_candidates[:, 0], green_candidates[:, 1], color='g', marker='o', s=5)

        traffic_light.set_title('traffic_lights(' + str(self.__index) + ')')
        traffic_light.imshow(curr_frame.img)
        red_traffic_lights = traffic_lights[auxiliary_t == 'red']
        green_traffic_lights = traffic_lights[auxiliary_t == 'green']
        traffic_light.scatter(red_traffic_lights[:, 0], red_traffic_lights[:, 1], color='r', marker='o', s=5)
        traffic_light.scatter(green_traffic_lights[:, 0], green_traffic_lights[:, 1], color='g', marker='o', s=5)

        if self.__prev_frame:
            distance.set_title('distances(' + str(self.__index) + ')')
            distance.imshow(curr_frame.img)
            distance.scatter(red_traffic_lights[:, 0], red_traffic_lights[:, 1], color='r', marker='o', s=5)
            distance.scatter(green_traffic_lights[:, 0], green_traffic_lights[:, 1], color='g', marker='o', s=5)

            for i in range(len(distances)):
                # distance.plot([traffic_lights[i, 0], self.__pp[0]], [traffic_lights[i, 1], self.__pp[1]], 'b')
                distance.text(traffic_lights[i, 0], traffic_lights[i, 1],
                              r'{0:.1f}'.format(distances[i]), color='r')

        plt.show()

    def on_frame(self, frame):
        # step 1
        candidates, auxiliary_c = self.__get_lights(frame)

        # step 2
        traffic_lights, auxiliary_t = self.__get_tfl_lights(frame, candidates, auxiliary_c)

        # step 3
        curr_frame = FrameContainer(frame)
        curr_frame.traffic_light = traffic_lights
        distances = []

        if self.__prev_frame:
            curr_frame.EM = self.__egomotions['egomotion_' + str(self.__index - 1) + '-' + str(self.__index)]
            distances = self.__get_distance(curr_frame)

        self.__display(curr_frame, candidates, auxiliary_c, traffic_lights, auxiliary_t, distances)
        self.__prev_frame = curr_frame
        self.__index += 1

        return traffic_lights, auxiliary_t, distances
