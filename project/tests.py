import itertools
import os
import random
import time
from unittest import TestCase, main as unittest_main

import cv2
import numpy as np
from keras import Model, Input
from keras.utils import to_categorical
from sinkhorn_knopp.sinkhorn_knopp import SinkhornKnopp  # TODO: install it (add to dep)

from models.fish_or_doc_classifier import FishOrDocClassifier
from utils.data_manipulations import shred_shuffle_and_reconstruct, list_of_images_to_numpy
from utils.data_provider import DataProvider
from utils.shredder import Shredder
from models.deep_permutation_network import DeepPermutationNetwork
from utils.layers import RepeatLayer, ExpandDimension

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

    def prepare_data(self, num_samples, resize=None):
        num_samples = 100
        dp = DataProvider()
        ts = (1, 2, 4, 5)
        fish = dp.get_fish_images(num_samples=num_samples, resize=resize)
        docs = dp.get_docs_images(num_samples=num_samples, resize=resize)
        fish = list(itertools.chain(*[shred_shuffle_and_reconstruct(fish, t) for t in ts]))
        docs = list(itertools.chain(*[shred_shuffle_and_reconstruct(docs, t) for t in ts]))
        if resize is not None:
            fish = list_of_images_to_numpy(fish)
            docs = list_of_images_to_numpy(docs)
        return fish, docs

    def test_accuracy_list_different_sizes(self):
        fod_clf = FishOrDocClassifier(
            data_provider=None,
            weights_file=os.path.join(os.getcwd(), 'models', 'saved_weights', 'fish_or_doc_clf_weights_acc_1.000.h5'))
        fish, docs = self.prepare_data(100)
        num_fish, num_docs = len(fish), len(docs)
        results = np.sum(fod_clf.is_fish(fish))
        print('Fish accuracy {}/{} ({:.3f})'.format(results, num_fish, results / num_fish))
        results = num_docs - np.sum(fod_clf.is_fish(docs))
        print('Docs accuracy {}/{} ({:.3f})'.format(results, num_docs, results / num_docs))

    def test_accuracy_numpy_same_sizes(self):
        fod_clf = FishOrDocClassifier(
            data_provider=None,
            weights_file=os.path.join(os.getcwd(), 'models', 'saved_weights', 'fish_or_doc_clf_weights_acc_1.000.h5'))
        fish, docs = self.prepare_data(100, resize=(260, 240))
        num_fish, num_docs = len(fish), len(docs)
        results = np.sum(fod_clf.is_fish(fish))
        print('Fish accuracy {}/{} ({:.3f})'.format(results, num_fish, results / num_fish))
        results = num_docs - np.sum(fod_clf.is_fish(docs))
        print('Docs accuracy {}/{} ({:.3f})'.format(results, num_docs, results / num_docs))


class DeepPermutationNetworkTests(TestCase):
    @staticmethod
    def _find_closest_l1_p_by_dsm_bf(dsm):
        n, n = dsm.shape
        best_permutations_matrix = to_categorical(np.arange(n))

        for current_permutation in itertools.permutations(list(range(n))):
            permutations_matrix = to_categorical(current_permutation, num_classes=n)
            assert np.allclose(current_permutation, permutations_matrix @ np.arange(n))

            if np.linalg.norm(permutations_matrix - dsm, ord=1) < np.linalg.norm(best_permutations_matrix - dsm, ord=1):
                best_permutations_matrix = permutations_matrix

        return best_permutations_matrix

    def test_find_p_by_dsm(self):
        np.random.seed(42)

        for i in range(100):
            n = np.random.randint(2, 5)
            matrix = np.random.sample((n, n))
            dsm = SinkhornKnopp().fit(matrix)
            p_predicted = DeepPermutationNetwork._find_p_by_dsm_using_lp(dsm)
            p_true = DeepPermutationNetworkTests._find_closest_l1_p_by_dsm_bf(dsm)
            l1_predicted = np.linalg.norm(p_predicted - dsm, ord=1)
            l1_true = np.linalg.norm(p_true - dsm, ord=1)

            print('Predicted L1 error: {}. Optimal L1 error: {}. Difference: {}'.format(l1_predicted, l1_true, l1_predicted - l1_true))

    def test_find_dsm_by_m_1(self):
        dsm =[[0.9998869,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.004297,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,1.0003198,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.9978018,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.9952281,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.9858321,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.99285775,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0032644,0.0,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0015687,0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0107807,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0041155,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.010773,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9950303,0.0],
                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0094427]]
        DeepPermutationNetwork._find_dsm_by_m(dsm)

    def test_find_p_by_m_t(self):
        np.random.seed(42)

        for t in (2, 4, 5):
            n = 1000
            start_time = time.time()

            for i in range(n):
                n = t**2
                matrix = np.random.sample((n, n))
                p_predicted = DeepPermutationNetwork._find_dsm_by_m(matrix)

            end_time = time.time()

            print('for t={}: {} s per run'.format(t, (end_time-start_time) / 100))



class RepeatLayerTests(TestCase):
    def test_repeat_channels_3(self):
        number_of_samples = 64
        shape = (128, 256, 1)
        input = Input(shape=shape)
        rl = RepeatLayer(3, -1)
        output = rl(input)
        model = Model([input], [output])

        x = np.arange(number_of_samples * int(np.prod(shape))).reshape(tuple([number_of_samples,] + list(shape)))
        y = model.predict(x)
        self.assertEqual(y.shape, tuple([number_of_samples, ] + list(shape[:-1]) + [3, ]))
        desired = np.repeat(x, 3, -1)
        np.testing.assert_allclose(y, desired)


class AppendDimensionTests(TestCase):
    def test_append_dimension(self):
        number_of_samples = 64
        shape = (128, 256)
        input = Input(shape=shape)
        ad = ExpandDimension()
        output = ad(input)
        model = Model([input], [output])

        x = np.arange(number_of_samples * int(np.prod(shape))).reshape(tuple([number_of_samples,] + list(shape)))
        y = model.predict(x)
        self.assertEqual(y.shape, tuple([number_of_samples, ] + list(shape) + [1, ]))
        desired = np.reshape(x, list(x.shape) + [1, ])
        np.testing.assert_allclose(y, desired)


if __name__ == '__main__':
    unittest_main()
