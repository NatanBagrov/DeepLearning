import math

import numpy as np

from models.comparator_cnn import ComparatorCNN
from utils.data_manipulations import resize_to
from utils.shredder import Shredder


class AdjacencyMatrixBuilder:
    @staticmethod
    def build_adjacency_matrices(comparator: ComparatorCNN, shreds):
        # Actually this check is redundant, but left to cover external usage (not from generic_solver)
        t_square = len(shreds)
        t = int(round(math.sqrt(t_square)))
        assert t ** 2 == t_square and comparator.t == t
        shreds = resize_to(shreds, (comparator.width, comparator.height))
        left_index_to_right_index_to_probability = \
            AdjacencyMatrixBuilder. \
                _get_first_index_to_second_index_to_probability(shreds, comparator.predict_is_left_probability)
        top_index_to_bottom_index_to_probability = \
            AdjacencyMatrixBuilder. \
                _get_first_index_to_second_index_to_probability(shreds, comparator.predict_is_top_probability)

        return left_index_to_right_index_to_probability, top_index_to_bottom_index_to_probability

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


class ObjectiveFunction:
    @staticmethod
    def compute(crop_position_in_original_image,
                left_index_to_right_index_to_probability,
                top_index_to_bottom_index_to_probability):
        t = int(round(math.sqrt(len(crop_position_in_original_image))))

        row_to_column_to_crop_index = \
            Shredder.shred_index_to_original_index_to_row_to_column_to_shred_index(crop_position_in_original_image)

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
                        log_objective = float("-inf")

                if column + 1 < t:
                    left_index = row_to_column_to_crop_index[row][column]
                    right_index = row_to_column_to_crop_index[row][column + 1]
                    objective *= left_index_to_right_index_to_probability[left_index][right_index]
                    # TODO: bug bug bug, fails when executing log!!!!
                    try:
                        log_objective += math.log(left_index_to_right_index_to_probability[left_index][right_index])
                    except ValueError:
                        log_objective = float("-inf")

        return objective, log_objective
