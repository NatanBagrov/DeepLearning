import math
import sys
import os
import time
from timeit import timeit
import itertools

import numpy as np
from sklearn.model_selection import train_test_split

from utils.shredder import Shredder
from utils.data_manipulations import resize_to, shred_and_resize_to
from utils.data_provider import DataProvider
from utils.image_type import ImageType
from utils.visualizer import Visualizer
from models.comparator_cnn import ComparatorCNN
from solvers.generic_solver_with_comparator import GenericSolverWithComparator


class SolverGreedy(GenericSolverWithComparator):
    def __init__(self, t_to_comparator, image_type=None):
        GenericSolverWithComparator.__init__(self, t_to_comparator, image_type)

    @staticmethod
    def _predict_greedy_iterating_on_generic_in_generic_order(
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability,
            iterate_on_bottom: bool,
            iterate_on_right: bool,
            column_then_row: bool,
    ):
        t_square = left_index_to_right_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))
        best_log_objective = float("-inf")
        best_shred_index_to_original_index = list(range(t_square))

        assert t**2 == t_square

        if column_then_row and iterate_on_right or not column_then_row and iterate_on_bottom:
            first_indices = list(reversed(range(t)))
        else:
            first_indices = list(range(t))

        if column_then_row and iterate_on_bottom or not column_then_row and iterate_on_right:
            second_indices = list(reversed(range(t)))
        else:
            second_indices = list(range(t))

        for first_shred_index in range(t_square):
            current_shred_index_to_original_index = SolverGreedy._continue_predicting_in_generic_order(
                left_index_to_right_index_to_probability,
                top_index_to_bottom_index_to_probability,
                column_then_row,
                first_shred_index,
                first_indices,
                second_indices
            )

            current_objective, current_log_objective = SolverGreedy._compute_objective(
                     current_shred_index_to_original_index,
                     left_index_to_right_index_to_probability,
                     top_index_to_bottom_index_to_probability
            )

            if current_log_objective > best_log_objective:
                best_log_objective = current_log_objective
                best_shred_index_to_original_index = current_shred_index_to_original_index

        return best_shred_index_to_original_index

    @staticmethod
    def _continue_predicting_in_generic_order(
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability,
            column_then_row: bool,
            first_shred_index,
            first_indices,
            second_indices,
    ):
        t_square = left_index_to_right_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))
        row_to_column_to_shred_index = [[None, ] * t for row in range(t)]
        current_shred_index_to_original_index = [None, ] * t_square

        for first_index in first_indices:
            for second_index in second_indices:
                if column_then_row:
                    column = first_index
                    row = second_index
                else:
                    row = first_index
                    column = second_index

                if first_indices[0] == first_index and second_indices[0] == second_index:
                    row_to_column_to_shred_index[row][column] = first_shred_index
                    current_shred_index_to_original_index[first_shred_index] = row * t + column
                else:
                    best_probability = float("-inf")
                    best_shred_index = None

                    for shred_index in range(t_square):
                        if current_shred_index_to_original_index[shred_index] is None:
                            current_probability = 1.0

                            for row_change, column_change in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                                if 0 != row_change and 0 <= row + row_change < t \
                                        and row_to_column_to_shred_index[row + row_change][column] is not None:
                                    if row_change == -1:
                                        top_index = row_to_column_to_shred_index[row + row_change][column]
                                        bottom_index = shred_index
                                    else:
                                        top_index = shred_index
                                        bottom_index = row_to_column_to_shred_index[row + row_change][column]

                                    vertical_probability = \
                                        top_index_to_bottom_index_to_probability[top_index][bottom_index]
                                else:
                                    vertical_probability = 1.0

                                if 0 != column_change and 0 <= column + column_change < t \
                                        and row_to_column_to_shred_index[row][column + column_change] is not None:
                                    if column_change == -1:
                                        left_index = row_to_column_to_shred_index[row][column + column_change]
                                        right_index = shred_index
                                    else:
                                        left_index = shred_index
                                        right_index = row_to_column_to_shred_index[row][column + column_change]

                                    horizontal_probability = \
                                        left_index_to_right_index_to_probability[left_index][right_index]
                                else:
                                    horizontal_probability = 1.0

                                current_probability *= vertical_probability * horizontal_probability

                            if current_probability > best_probability:
                                best_probability = current_probability
                                best_shred_index = shred_index

                    row_to_column_to_shred_index[row][column] = best_shred_index
                    current_shred_index_to_original_index[best_shred_index] = row * t + column

        current_shred_index_to_original_index = SolverGreedy._try_to_improve_with_row_permutation(
            current_shred_index_to_original_index,
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability,
        )

        return current_shred_index_to_original_index

    def _predict(self,
                 left_index_to_right_index_to_probability,
                 top_index_to_bottom_index_to_probability):
        best_log_objective = float("-inf")
        best_crop_position_in_original_image = None
        best_configuration = None

        for iterate_on_bottom in (False, True):
            for iterate_on_right in (False, True):
                for column_then_row in (False, True):
                    current_crop_position_in_original_image = \
                        SolverGreedy._predict_greedy_iterating_on_generic_in_generic_order(
                        left_index_to_right_index_to_probability,
                        top_index_to_bottom_index_to_probability,
                        iterate_on_bottom,
                        iterate_on_right,
                        column_then_row
                    )

                    current_objective, current_log_objective = \
                        SolverGreedy._compute_objective(
                            current_crop_position_in_original_image,
                            left_index_to_right_index_to_probability,
                            top_index_to_bottom_index_to_probability)

                    if best_crop_position_in_original_image is None or current_log_objective > best_log_objective:
                        best_log_objective = current_log_objective
                        best_configuration = (iterate_on_bottom, iterate_on_right, column_then_row)
                        best_crop_position_in_original_image = current_crop_position_in_original_image

        print('Using {} {} {}-wise'.format(
            'bottom' if best_configuration[0] else 'top',
            'right' if best_configuration[1] else 'left',
            'column' if best_configuration[2] else 'row',
        ))

        return best_crop_position_in_original_image

    @staticmethod
    def _try_to_improve_with_row_permutation(shred_index_to_original_index: list,
                                             left_index_to_right_index_to_probability: np.array,
                                             top_index_to_bottom_index_to_probability: np.array):
        t = int(round(math.sqrt(len(shred_index_to_original_index))))
        row_to_column_to_shred_index = \
            GenericSolverWithComparator._shred_index_to_original_index_to_row_to_column_to_shred_index(
                shred_index_to_original_index
            )

        best_shred_index_to_original_index = shred_index_to_original_index
        best_objective, best_log_objective = GenericSolverWithComparator._compute_objective(
            shred_index_to_original_index,
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability)

        for row_permutation in itertools.permutations(range(t)):
            current_row_to_column_to_shred_index = row_to_column_to_shred_index[list(row_permutation)]
            current_shred_index_to_original_index = GenericSolverWithComparator.\
                _row_to_column_to_shred_index_to_shred_index_to_original_index(
                    current_row_to_column_to_shred_index
                )
            current_objective, current_log_objecitve = GenericSolverWithComparator._compute_objective(
                current_shred_index_to_original_index,
                left_index_to_right_index_to_probability,
                top_index_to_bottom_index_to_probability
            )

            if best_log_objective < current_log_objecitve:
                best_log_objective = current_log_objecitve
                best_objective = current_objective
                best_shred_index_to_original_index = current_shred_index_to_original_index

        return best_shred_index_to_original_index


def main():
    if 'debug' in sys.argv:
        print('Debug')
        number_of_samples = 20
        epochs = 1
    else:
        print('Release')
        number_of_samples = sys.maxsize
        epochs = 5

    ts = list()

    if '2' in sys.argv:
        ts.append(2)

    if '4' in sys.argv:
        ts.append(4)

    if '5' in sys.argv:
        ts = [5,]

    if 0 == len(ts):
        ts = (2, 4, 5)

    image_types = list()

    if 'image' in sys.argv:
        image_types.append(ImageType.IMAGES)

    if 'document' in sys.argv:
        image_types.append(ImageType.DOCUMENTS)

    if 0 == len(image_types):
        image_types = ImageType

    np.random.seed(42)

    width = 2200 // 5
    height = 2200 // 5

    for image_type in image_types:
        print(image_type.value)

        if image_type == ImageType.IMAGES:
            get_images = DataProvider().get_fish_images
            mean = 100.52933494138787
            std = 65.69793156777682
        else:
            get_images = DataProvider().get_docs_images
            mean = 241.46115784237548
            std = 49.512839464023564

        images, names = get_images(num_samples=number_of_samples, return_names=True)

        images_train, images_validation, names_train, names_validation = train_test_split(images, names,
                                                                                          random_state=42)
        t_to_comparator = {
            t: ComparatorCNN(t, width, height, image_type, mean=mean, std=std)
                .load_weights()
            for t in ts
        }

        clf = SolverGreedy(t_to_comparator, image_type=image_type)
        print('Train: ', names_train)
        accuracy = clf.evaluate(images_train, epochs=epochs, ts=ts)
        print('Train 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))
        print('Validation: ', names_validation)
        accuracy = clf.evaluate(images_validation, epochs=epochs, ts=ts)
        print('Validation 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))


def debug():
    names_train = ['n01440764_11593.JPEG', 'n01440764_11602.JPEG', 'n01440764_4562.JPEG', 'n01440764_5148.JPEG', 'n01440764_11897.JPEG', 'n01440764_29057.JPEG', 'n01440764_22135.JPEG', 'n01440764_8003.JPEG', 'n01440764_3566.JPEG', 'n01440764_44.JPEG', 'n01440764_10910.JPEG', 'n01440764_10382.JPEG', 'n01440764_6508.JPEG', 'n01440764_10290.JPEG', 'n01440764_910.JPEG']
    images_train, names_train = DataProvider.read_images('../images', names_train)
    indices = [4, 5, 13]
    images_validation = [images_train[index] for index in indices]
    index = 4
    t = 4
    permutation = [2, 8, 4, 12, 0, 10, 6, 5, 7, 3, 13, 15, 11, 9, 1, 14]
    width = 224
    height = 224
    image_type = ImageType.IMAGES

    cmp = ComparatorCNN(t, width, height, image_type)\
        ._fit_standardisation(images_train)\
        .load_weights()

    slv = SolverGreedy({t: cmp})
    score = slv.evaluate_image_for_permutation(images_validation[0], permutation, sample_index=index)
    print('done with ', score)


if __name__ == '__main__':
    # debug()
    main()