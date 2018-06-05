import cv2

import numpy as np


def list_of_images_to_numpy(images_list):
    return np.stack(images_list, axis=0)


def normalize_grayscale_0_1(x):
    return x / 255


def resize_to(x, width, height):
    return cv2.resize(x, (width, height))
