import os
import random
from unittest import TestCase, main as unittest_main

import cv2
import numpy as np

from models.fish_or_doc_classifier import FishOrDocClassifier
from utils.data_manipulations import list_of_images_to_numpy
from utils.data_provider import LimitedDataProvider
from utils.shredder import Shredder

tests_root = os.path.join(os.getcwd(), 'test_files')


class ShredderTests(TestCase):
    shredder_tests_root = os.path.join(tests_root, 'shredder')

    def test_shreds_both_color_and_grayscaled(self):
        im_path = os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG')
        im_color = cv2.imread(im_path)
        im_grayscaled = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY)
        t = 5
        res_from_color = Shredder.shred(im_color, t)
        res_from_grayscaled = Shredder.shred(im_grayscaled, t)
        [np.testing.assert_equal(c1, c2) for c1, c2 in zip(res_from_grayscaled, res_from_color)]

    def test_shreds_proper_sizes_and_number_of_crops(self):
        for im_path in [os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG'),
                        os.path.join(self.__class__.shredder_tests_root, 'doc_orig.jpg')]:
            im = cv2.imread(im_path)
            for t in (2, 4, 5):
                shredded = Shredder.shred(im, t)
                assert len(shredded) == t ** 2
                for crop in shredded:
                    assert crop.shape == (im.shape[0] // t, im.shape[1] // t)

    def test_reconstruction(self):
        for im_path in [os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG'),
                        os.path.join(self.__class__.shredder_tests_root, 'doc_orig.jpg')]:
            im = cv2.imread(im_path)
            im_grayscaled = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            for t in (2, 4, 5):
                shredded = Shredder.shred(im, t)
                reconstructed = Shredder.reconstruct(shredded)
                trimmed = im_grayscaled[0:reconstructed.shape[0], 0:reconstructed.shape[1]]
                np.testing.assert_equal(reconstructed, trimmed)

    def test_shuffle_shreds(self):
        im_path = os.path.join(self.__class__.shredder_tests_root, 'fish_orig.JPEG')
        im_grayscaled = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_RGB2GRAY)
        t = 5
        shuffled = Shredder.shred(im_grayscaled, t, shuffle_shreds=True)
        reconstructed = Shredder.reconstruct(shuffled)
        trimmed = im_grayscaled[0:reconstructed.shape[0], 0:reconstructed.shape[1]]
        assert np.linalg.norm(trimmed - reconstructed) > 1000  # just some magic number


class FishOrDocClassifierTests(TestCase):
    def test_accuracy(self):
        fod_clf = FishOrDocClassifier(
            data_provider=None,
            weights_file=os.path.join(os.getcwd(), 'models', 'saved_weights', 'fish_or_doc_clf_weights_acc_1.000.h5'))
        samples_num = 30
        dp = LimitedDataProvider(samples_num)
        fish = dp.get_fish_images(grayscaled=True)
        docs = dp.get_docs_images(grayscaled=True)
        fish = list(
            map(lambda im: Shredder.reconstruct(Shredder.shred(im, random.choice([1, 2, 4, 5]), shuffle_shreds=True)),
                fish))
        docs = list(
            map(lambda im: Shredder.reconstruct(Shredder.shred(im, random.choice([1, 2, 4, 5]), shuffle_shreds=True)),
                docs))
        fish_samples, docs_samples = len(fish), len(docs)
        results = np.sum([fod_clf.is_fish(f) for f in fish] * 1)
        print('Fish accuracy {}/{} ({:.3f})'.format(results, fish_samples, results / fish_samples))
        results = np.sum([fod_clf.is_doc(f) for f in docs] * 1)
        print('Docs accuracy {}/{} ({:.3f})'.format(results, docs_samples, results / docs_samples))


if __name__ == '__main__':
    unittest_main()
