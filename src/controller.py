from tfl_man import TFL_Man


class Controller:
    def __init__(self, pls_path):
        self.__frame = []

        pkl = None
        offset = None

        with open(pls_path, encoding='UTF-8') as f:

            for i, j in enumerate(f):
                if i == 0:
                    assert j[-4:] == 'pkl\n'
                    pkl = j[:-1]
                elif i == 1:
                    assert j[:-1].isdigit()
                    offset = int(j[:-1])
                else:
                    assert j[-4:] == 'png\n'
                    self.__frame.append(j[:-1])

        assert pkl and offset
        self.__tfl_manager = TFL_Man(pkl, offset)

    def run(self):
        for img_path in self.__frame:
            self.__tfl_manager.on_frame(img_path)


# class Controller:
#     def __init__(self, pls_file):
#         self.__play_list = pls_file
#
#         with open(pls_file, encoding='UTF-8') as f:
#             assert sum(1 for _ in f) > 1
#             assert f.readlines()[1][-4:-1] == 'pkl'
#             assert f.readlines()[2][:-1].isdigit()
#
#             f.readline()
#             self.__tfl_man = TFL_Man(f.readline()[:-1], int(f.readline()[:-1]))
#
#     def run(self):
#         with open(self.__play_list, encoding='UTF-8') as f:
#             for path_line in f:
#                 if path_line[-4:] == "png\n":
#                     self.__tfl_man.on_frame(path_line[:-1])
