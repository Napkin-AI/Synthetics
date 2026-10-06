import os
import numpy as np
import cv2
from collections import deque


def log(name: str, object: str, count: int, white_pixel_count: int, log_path: str) -> None:
    record = f'{name};{object};{count};{white_pixel_count}\n'
    log_file = open(log_path + "\\" + "logs.txt", 'a')
    log_file.write(record)
    log_file.close()

#
# def ok(x, y, n, was):
#     return 0 <= x < n and n > y >= 0 and (was[x][y] == 0)
#
# # It is better than cv2.connectedComp if we want to know count of areas
# def find_verices(x, y, was, label_map):
#     deq = deque()
#     deq.append([x, y])
#     n = len(was)
#     cnt = 0
#     while len(deq):
#         x1, y1 = deq.popleft()
#         for pad_x in range(-5, 6):
#             for pad_y in range(-5, 6):
#                 if ok(x1 + pad_x, y1 + pad_y, n, was):
#                     was[x1 + pad_x][y1 + pad_y] = 1
#                     cnt += 1
#                     deq.append([x1 + pad_x, y1 + pad_y])
#
#     return cnt

def add_log(path: str, objects: list) -> None:

    statistics = {}
    for obj_name in objects:
        total_count, total_area = 0, 0

        obj_path = path + f'\\{obj_name}\\'
        files = [item for item in os.scandir(obj_path)]

        for item in files:

            if item.name.lower().endswith(".png"):
                label_map = cv2.imread(obj_path + item.name, cv2.IMREAD_GRAYSCALE)

                # Подсчет количества белых пикселей
                white_pixel_count = np.sum(label_map == 255)

                connectivity = 4
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_map, connectivity, cv2.CV_32S)

                total_area += white_pixel_count
                total_count += num_labels - 1

                log(item.name[:item.name.rfind('.')], obj_name, num_labels - 1, white_pixel_count, path)

        statistics[obj_name] = (total_count, total_area)

    for obj_name, (total_count, total_area) in statistics.items():
        log('TOTAL_STATISTICS',obj_name, total_count, total_area, path)

if __name__ == "__main__":
    objects = ['axon', 'boundaries', 'mitochondria', 'mitochondrial boundaries', 'PSD',  'vesicles']
    synth_path = r"D:\\course_work\\data\\Kasthuri++\\labeled_data\\cropped_images\\cutted_data_3_slices"
    origin_path = r"D:\course_work\4_course\everything\check_psd_count\synthetic_dataset10"

    synth_path = r'D:\course_work\data\SLICES\cutting data 30 slices'
    for path in [
        # r'D:\course_work\data\Kasthuri++\labeled_data\cropped_images\cutted_data_3_slices\datasets\train',
        # r'D:\course_work\data\Kasthuri++\labeled_data\cropped_images\cutted_data_3_slices\datasets\test',
        # r"D:\\course_work\\data\\Kasthuri++\\labeled_data\\cropped_images\\cutted_data_3_slices",
        synth_path,
    ]:
        add_log(path, objects)
        print(f"Done {path}")
