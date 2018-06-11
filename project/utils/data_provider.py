import cv2
import imghdr
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from utils.data_manipulations import list_of_images_to_numpy, resize_to, shred_and_resize_to
from utils.image_type import  ImageType

fish_avg_shape = (387, 486)
docs_avg_shape = (2200, 1700)


# TODO: there is likely more or less ready lasy solution for this (at least tf has it if i am not mistaken).
class DataProvider:

    def __init__(self) -> None:
        super().__init__()
        self._fish_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
        self._docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'documents'))

    def get_fish_images(self, num_samples=9999, grayscaled=True, resize=None):
        return self._get_images_from_path(self._fish_path, num_samples, grayscaled, resize)

    def get_docs_images(self, num_samples=9999, grayscaled=True, resize=None):
        return self._get_images_from_path(self._docs_path, num_samples, grayscaled, resize)

    def _get_images_from_path(self, path, num_samples, grayscaled, resize):
        images = list()
        for f in os.listdir(path)[:num_samples]:
            file_path = os.path.join(path, f)
            if imghdr.what(file_path) is None:
                continue
            im = cv2.imread(os.path.join(path, file_path))
            if grayscaled:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if resize is not None:
                im = resize_to(im, resize)
            images.append(im)
        return list_of_images_to_numpy(images) if resize is not None else images

    def get_train_validation_test_sets_as_array_of_shreds_and_array_of_permutations(
            self,
            t: int,
            width: int, height: int,
            image_type: ImageType,
            number_of_samples: int=sys.maxsize,
            train_size=0.7,
            test_size=0.1,

    ):
        """

        :param t:
        :param width:
        :param height:
        :param image_type:
        :param number_of_samples:
        :param train_size:
        :param test_size:
        :return: ((train_x, train_y), (validation_x, validation_y), (test_x, test_y)),  where
            <set>_x is <set>_number_of_samples x t**2 x height x width np.array,
            <set>_y is <set>_number_of_samples x t**2 np.array,
        """
        if image_type == ImageType.DOCUMENTS:
            get_images = self.get_docs_images
        else:
            get_images = self.get_fish_images

        images = get_images(num_samples=number_of_samples)

        x = shred_and_resize_to(images, t, (width, height))
        assert x.shape == (len(images), t ** 2, height, width), '{}'.format(x.shape)
        y = np.repeat(np.arange(t**2).reshape(1, -1), len(images), axis=0)

        train_validation_x, test_x, train_validation_y, test_y = \
            train_test_split(x, y, test_size=test_size)
        train_x, validation_x, train_y, validation_y = \
            train_test_split(train_validation_x, train_validation_y, train_size=train_size)

        assert train_x.shape == (train_x.shape[0], t**2, height, width)
        assert train_y.shape == (train_x.shape[0], t**2)
        assert validation_x.shape == (validation_x.shape[0], t**2, height, width)
        assert validation_y.shape == (validation_x.shape[0], t ** 2)
        assert test_x.shape == (test_x.shape[0], t ** 2, height, width)
        assert test_y.shape == (test_x.shape[0], t ** 2)

        return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)


if __name__ == '__main__':
    dp = DataProvider()
    images = dp.get_fish_images()
    w = np.array([x.shape[0] for x in images])
    h = np.array([x.shape[1] for x in images])
    print("Avg for fish: {}x{}".format(int(np.average(h)), int(np.average(w))))
    print("Min for fish: {}x{}".format(int(np.min(h)), int(np.min(w))))
    images = dp.get_docs_images()
    w = np.array([x.shape[0] for x in images])
    h = np.array([x.shape[1] for x in images])
    print("Avg for docs: {}x{}".format(int(np.average(h)), int(np.average(w))))
    print("Min for docs: {}x{}".format(int(np.min(h)), int(np.min(w))))
