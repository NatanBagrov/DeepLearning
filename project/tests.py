import os
from unittest import TestCase, main as unittest_main

import cv2
import numpy as np

from shredder import Shredder

tests_root = os.path.join(os.getcwd(), 'test_files')


class ShredderTests(TestCase):
    shredder_tests_root = os.path.join(tests_root, 'shredder')

    def test_shreds_both_color_and_grayscaled(self):
        im_path = os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG')
        im_color = cv2.imread(im_path)
        im_grayscaled = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY)
        t = 5
        res_from_color = Shredder.shred_image(im_color, t)
        res_from_grayscaled = Shredder.shred_image(im_grayscaled, t)
        [np.testing.assert_equal(c1, c2) for c1, c2 in zip(res_from_grayscaled, res_from_color)]

    def test_shreds_proper_sizes_and_number_of_crops(self):
        for im_path in [os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG'),
                        os.path.join(self.__class__.shredder_tests_root, 'doc_orig.jpg')]:
            im = cv2.imread(im_path)
            for t in (2, 4, 5):
                shredded = Shredder.shred_image(im, t)
                assert len(shredded) == t ** 2
                for crop in shredded:
                    assert crop.shape == (im.shape[0]//t, im.shape[1]//t)

    def test_reconstruction(self):
        for im_path in [os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG'),
                        os.path.join(self.__class__.shredder_tests_root, 'doc_orig.jpg')]:
            im = cv2.imread(im_path)
            im_grayscaled = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            for t in (2, 4, 5):
                shredded = Shredder.shred_image(im, t)
                reconstructed = Shredder.reconstruct(shredded)
                trimmed = im_grayscaled[0:reconstructed.shape[0], 0:reconstructed.shape[1]]
                np.testing.assert_equal(reconstructed, trimmed)


if __name__ == '__main__':
    unittest_main()
