import math
from random import shuffle

import cv2
import numpy as np


class Shredder:
    @staticmethod
    def shred(np_image, t, shuffle_shreds=False):
        """
        Shredding an image (cv2) into t^2 shreds
        :param np_image: a cv2 loaded image
        :param t: the amount of tiles per dimension
        :param shuffle_shreds: flag indicates whether to return shuffled shreds or not
        :return a list of t-squared cv2 crops, crop at index i is the i-th shred row-major
        """
        if np_image.ndim == 3:
            im = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            im = np_image
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // t
        frac_w = width // t
        result = []
        for h in range(t):
            for w in range(t):
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                result.append(crop)
        if shuffle_shreds:
            shuffle(result)

        return np.stack(result, axis=0)

    @staticmethod
    def reconstruct(crops_list):
        """
        Reconstructing the original image given a list of crops row-major
        :param crops_list: the list of crops
        :return: np array - the reconstructed image
        """
        t = round(math.sqrt(len(crops_list)))
        return np.concatenate([np.concatenate(crops_list[t * idx:t * idx + t], axis=1) for idx in range(t)], axis=0)

    @staticmethod
    def shred_index_to_original_index_to_row_to_column_to_shred_index(crop_position_in_original_image: list):
        t = int(round(math.sqrt(len(crop_position_in_original_image))))

        row_to_column_to_crop_index = np.empty((t, t), dtype=int)

        for crop_index, crop_position in enumerate(crop_position_in_original_image):
            row = crop_position // t
            column = crop_position % t
            row_to_column_to_crop_index[row][column] = crop_index

        return row_to_column_to_crop_index
