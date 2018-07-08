import cv2
import imghdr
import os

import numpy as np

from utils.data_manipulations import list_of_images_to_numpy, resize_to

fish_avg_shape = (387, 486)
docs_avg_shape = (2200, 1700)


class DataProvider:

    def __init__(self) -> None:
        super().__init__()
        self._fish_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
        self._docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'documents'))
        self._debug_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images_to_debug'))

    def get_fish_images(self, num_samples=9999, grayscaled=True, resize=None, return_names=False):
        return self._get_images_from_path(self._fish_path, num_samples, grayscaled, resize, return_names=return_names)

    def get_docs_images(self, num_samples=9999, grayscaled=True, resize=None, return_names=False):
        return self._get_images_from_path(self._docs_path, num_samples, grayscaled, resize, return_names=return_names)

    def get_debug(self, num_samples=9999, grayscaled=True, resize=None, return_names=False):
        return self._get_images_from_path(self._debug_path, num_samples, grayscaled, resize, return_names=return_names)

    def _get_images_from_path(self, path, num_samples, grayscaled, resize, return_names=False):
        images, image_names = \
            DataProvider.read_images(path, os.listdir(path)[:num_samples], grayscaled=grayscaled, resize=resize)

        if return_names:
            return images, image_names
        else:
            return images

    @staticmethod
    def read_images(path, image_names, grayscaled=True, resize=None):
        true_image_names = list()
        images = list()

        for f in image_names:
            file_path = os.path.join(path, f)
            if imghdr.what(file_path) is None:
                continue
            im = DataProvider.read_image(os.path.join(path, file_path), grayscaled=grayscaled, resize=resize)
            true_image_names.append(f)
            images.append(im)

        return list_of_images_to_numpy(images) if resize is not None else images, true_image_names

    @staticmethod
    def read_image(path, grayscaled=True, resize=None,):
        im = cv2.imread(path)

        if grayscaled:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        if resize is not None:
            im = resize_to(im, resize)

        return im


if __name__ == '__main__':
    dp = DataProvider()
    images = dp.get_fish_images()
    w = np.array([x.shape[0] for x in images])
    h = np.array([x.shape[1] for x in images])
    print("Avg for fish: {}x{}".format(int(np.average(h)), int(np.average(w))))
    images = dp.get_docs_images()
    w = np.array([x.shape[0] for x in images])
    h = np.array([x.shape[1] for x in images])
    print("Avg for docs: {}x{}".format(int(np.average(h)), int(np.average(w))))
