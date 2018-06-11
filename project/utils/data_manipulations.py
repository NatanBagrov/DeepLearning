import cv2

import numpy as np

from utils.shredder import Shredder


def list_of_images_to_numpy(images_list):
    return np.stack(images_list, axis=0)


def resize_to(x, resize_shape):
    # IMPORTANT! note that for unknown reason, bloody cv2.resize() returns images of resize_shape[1], resize_shape[0]
    f = lambda v: cv2.resize(v, resize_shape)
    if type(x) == np.ndarray:
        if x.ndim == 2:
            return f(x)
        assert x.ndim == 3
        return list_of_images_to_numpy([f(im) for im in x[:]])
    return list_of_images_to_numpy([f(im) for im in x])


def shred_and_resize_to(x: list, t, width_height):
    def shred_and_resize_to_single_image(image):
        return resize_to(Shredder.shred(image, t), width_height)

    return list_of_images_to_numpy(list(map(shred_and_resize_to_single_image, x)))


def shred_shuffle_and_reconstruct(x, t):
    f = lambda v: Shredder.reconstruct(Shredder.shred(v, t, shuffle_shreds=True))
    if type(x) == np.ndarray:
        if x.ndim == 2:
            return f(x)
        assert x.ndim == 3, 'Invalid dimensions'
        return list_of_images_to_numpy([f(im) for im in x[:]])
    else:  # Here, we got a list of images with different size
        return [f(v) for v in x]
