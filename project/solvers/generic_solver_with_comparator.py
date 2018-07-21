import math
import os
import time
from abc import ABC, abstractmethod

import numpy as np

from utils.data_manipulations import resize_to
from utils.data_provider import DataProvider
from utils.shredder import Shredder
from utils.visualizer import Visualizer


class GenericSolverWithComparator(ABC):
    def __init__(self, t_to_comparator, image_type=None):
        self._image_type = image_type
        self._t_to_comparator = t_to_comparator

    def predict(self, shreds: list, return_log_objective=False) -> list:
        t_square = len(shreds)
        t = int(round(math.sqrt(t_square)))
        assert t ** 2 == t_square
        assert t in self._t_to_comparator
        comparator = self._t_to_comparator[t]
        shreds = resize_to(shreds, (comparator.width, comparator.height))
        left_index_to_right_index_to_probability = \
            GenericSolverWithComparator._get_first_index_to_second_index_to_probability(
                shreds,
                comparator.predict_is_left_probability)
        top_index_to_bottom_index_to_probability = \
            GenericSolverWithComparator._get_first_index_to_second_index_to_probability(
                shreds,
                comparator.predict_is_top_probability)

        prediction_and_maybe_objective_log = self._predict(
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability,
            return_log_objective
        )

        return prediction_and_maybe_objective_log

    def evaluate(self, images: list, ts=(2, 4, 5), epochs=1):
        index_to_accuracy = list()

        for t in ts:
            sample_index_to_shred_index_to_image = [Shredder.shred(image, t) for image in images]
            accuracies = list()

            for epoch in range(epochs):
                for sample_index, shred_index_to_image in enumerate(sample_index_to_shred_index_to_image):
                    permutation = np.random.permutation(t ** 2)
                    # permutation = np.arange(t ** 2)
                    current_accuracy = self.evaluate_image_for_permutation(shred_index_to_image,
                                                                           permutation,
                                                                           sample_index)
                    accuracies.append(current_accuracy)

            current_accuracy = np.average(accuracies)
            print('For t={} and image_type={} 0-1 is {}'.format(t, self._image_type, current_accuracy))
            index_to_accuracy.append(current_accuracy)

        return np.average(index_to_accuracy)

    def evaluate_image_for_permutation(self, shred_index_to_image, permutation, sample_index=None):
        t = int(round(math.sqrt(len(permutation))))

        if isinstance(shred_index_to_image, str):
            shred_index_to_image = DataProvider.read_image(shred_index_to_image)

        if np.shape(shred_index_to_image)[0] != len(permutation):
            shred_index_to_image = Shredder.shred(shred_index_to_image, t)

        shreds_permuted = shred_index_to_image[permutation]
        permutation_predicted = self.predict(shreds_permuted)
        current_accuracy = np.average(permutation_predicted == permutation)

        if not np.isclose(current_accuracy, 1.0):
            print('On #{} 0-1 is {}: {}!={}'.format(sample_index, current_accuracy,
                                                    permutation, permutation_predicted))
            visualize = True

            if visualize:
                directory_path = 'problems/{}/{}'.format(t, self._image_type)
                os.makedirs(directory_path, exist_ok=True)
                time_stamp = int(time.time())

                Visualizer.visualize_crops(shreds_permuted[np.argsort(permutation)],
                                           show=False,
                                           save_path=os.path.join(directory_path, '{}-original.png'.format(time_stamp)))
                Visualizer.visualize_crops(shreds_permuted[np.argsort(permutation_predicted)],
                                           show=False,
                                           save_path=os.path.join(directory_path, '{}-restored.png'.format(time_stamp)))
                print('visualized')

        return current_accuracy

    @staticmethod
    def _get_first_index_to_second_index_to_probability(images, predict_probability):
        first_index_to_second_index_to_first_image = np.array([
            [
                images[first_index]
                for second_image in range(images.shape[0])
            ]
            for first_index in range(images.shape[0])
        ])
        first_index_to_second_index_to_second_image = np.array([
            [
                images[second_image]
                for second_image in range(images.shape[0])
            ]
            for first_index in range(images.shape[0])
        ])

        assert (images.shape[0], images.shape[0], images.shape[1], images.shape[2]) \
               == first_index_to_second_index_to_first_image.shape
        assert (images.shape[0], images.shape[0], images.shape[1], images.shape[2]) \
               == first_index_to_second_index_to_second_image.shape

        first_index_to_second_index_to_first_image = np.reshape(first_index_to_second_index_to_first_image,
                                                                (
                                                                    images.shape[0] ** 2, images.shape[1],
                                                                    images.shape[2]))
        first_index_to_second_index_to_second_image = np.reshape(first_index_to_second_index_to_second_image,
                                                                 (images.shape[0] ** 2, images.shape[1],
                                                                  images.shape[2]))

        first_index_to_second_index_to_probability = predict_probability(first_index_to_second_index_to_first_image,
                                                                         first_index_to_second_index_to_second_image)

        assert (images.shape[0] ** 2, 2) == \
               first_index_to_second_index_to_probability.shape

        first_index_to_second_index_to_probability = first_index_to_second_index_to_probability[:, 1]
        first_index_to_second_index_to_probability = np.reshape(first_index_to_second_index_to_probability,
                                                                (images.shape[0], images.shape[0]))

        return first_index_to_second_index_to_probability

    @staticmethod
    def _compute_objective(crop_position_in_original_image,
                           left_index_to_right_index_to_probability,
                           top_index_to_bottom_index_to_probability):
        t = int(round(math.sqrt(len(crop_position_in_original_image))))

        row_to_column_to_crop_index = \
            GenericSolverWithComparator._shred_index_to_original_index_to_row_to_column_to_shred_index(
                crop_position_in_original_image
            )

        objective = 1.0
        log_objective = 0.0

        for row in range(t):
            for column in range(t):
                if row + 1 < t:
                    top_index = row_to_column_to_crop_index[row][column]
                    bottom_index = row_to_column_to_crop_index[row + 1][column]
                    objective *= top_index_to_bottom_index_to_probability[top_index][bottom_index]

                    try:
                        log_objective += math.log(top_index_to_bottom_index_to_probability[top_index][bottom_index])
                    except ValueError:
                        print('Can not calculate log({})'.format(
                            top_index_to_bottom_index_to_probability[top_index][bottom_index])
                        )
                        log_objective = float("-inf")

                if column + 1 < t:
                    left_index = row_to_column_to_crop_index[row][column]
                    right_index = row_to_column_to_crop_index[row][column + 1]
                    objective *= left_index_to_right_index_to_probability[left_index][right_index]
                    # TODO: bug bug bug, fails when executing log!!!!
                    try:
                        log_objective += math.log(left_index_to_right_index_to_probability[left_index][right_index])
                    except ValueError:
                        print('Can not calculate log({})'.format(
                            left_index_to_right_index_to_probability[left_index][right_index])
                        )
                        log_objective = float("-inf")

        return objective, log_objective

    @abstractmethod
    def _predict(self, left_index_to_right_index_to_probability, top_index_to_bottom_index_to_probability,
                 return_log_objective=False):
        raise NotImplementedError

    @staticmethod
    def _shred_index_to_original_index_to_row_to_column_to_shred_index(crop_position_in_original_image: list):
        t = int(round(math.sqrt(len(crop_position_in_original_image))))

        row_to_column_to_crop_index = np.empty((t, t), dtype=int)

        for crop_index, crop_position in enumerate(crop_position_in_original_image):
            row = crop_position // t
            column = crop_position % t
            row_to_column_to_crop_index[row][column] = crop_index

        return row_to_column_to_crop_index

    @staticmethod
    def _row_to_column_to_shred_index_to_shred_index_to_original_index(row_to_column_to_crop_index) -> list:
        t = row_to_column_to_crop_index.shape[0]
        crop_position_in_original_image = [None, ] * (t**2)

        for row in range(t):
            for column in range(t):
                crop_position_in_original_image[row_to_column_to_crop_index[row][column]] = row * t + column

        return crop_position_in_original_image
