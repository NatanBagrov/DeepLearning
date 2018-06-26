import math
import sys
import os
import time
from timeit import timeit

import numpy as np
from sklearn.model_selection import train_test_split
from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, lpSum, lpDot, LpStatusOptimal, LpStatus

from utils.shredder import Shredder
from utils.data_manipulations import resize_to, shred_and_resize_to
from utils.data_provider import DataProvider
from utils.image_type import ImageType
from utils.visualizer import Visualizer
from models.comparator_cnn import ComparatorCNN


class SolverWithComparator:
    def __init__(self, t_to_comparator, image_type=None):
        self._t_to_comparator = t_to_comparator
        self._image_type = image_type

    def predict(self, shreds: list):
        t_square = len(shreds)
        t = int(round(math.sqrt(t_square)))
        assert t ** 2 == t_square
        assert t in self._t_to_comparator
        comparator = self._t_to_comparator[t]
        shreds = resize_to(shreds, (comparator.width, comparator.height))
        left_index_to_right_index_to_probability = SolverWithComparator._get_first_index_to_second_index_to_probability(
            shreds,
            comparator.predict_is_left_probability)
        top_index_to_bottom_index_to_probability = SolverWithComparator._get_first_index_to_second_index_to_probability(
            shreds,
            comparator.predict_is_top_probability)

        prediction = self._predict_greedy(
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability
        )

        return prediction

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
                os.makedirs('problems/{}/'.format(t), exist_ok=True)
                time_stamp = int(time.time())

                Visualizer.visualize_crops(shreds_permuted[np.argsort(permutation)],
                                           show=False,
                                           save_path='problems/{}/{}-original.png'.format(t, time_stamp))
                Visualizer.visualize_crops(shreds_permuted[np.argsort(permutation_predicted)],
                                           show=False,
                                           save_path='problems/{}/{}-restored.png'.format(t, time_stamp))
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
                                                                (images.shape[0] ** 2, images.shape[1], images.shape[2]))
        first_index_to_second_index_to_second_image = np.reshape(first_index_to_second_index_to_second_image,
                                                                (images.shape[0] ** 2, images.shape[1], images.shape[2]))

        first_index_to_second_index_to_probability = predict_probability(first_index_to_second_index_to_first_image,
                                                                         first_index_to_second_index_to_second_image)

        assert (images.shape[0] ** 2, 2) == \
               first_index_to_second_index_to_probability.shape

        first_index_to_second_index_to_probability = first_index_to_second_index_to_probability[:, 1]
        first_index_to_second_index_to_probability = np.reshape(first_index_to_second_index_to_probability,
                                                                (images.shape[0], images.shape[0]))

        return first_index_to_second_index_to_probability

    @staticmethod
    def _predict_with_lp(left_index_to_right_index_to_probability,
                         top_index_to_bottom_index_to_probability):
        print('Solving lp')

        t_square = left_index_to_right_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))
        assert t ** 2 == t_square

        problem = LpProblem(name="permutation", sense=LpMaximize)
        index_to_row_to_column = LpVariable.matrix(
            'index_to_row_to_column',
            (list(range(t_square)), list(range(t)), list(range(t))),
            lowBound=0,
            upBound=1,
            cat=LpInteger,
        )
        left_index_to_right_index_to_is_left, horizontal_objective = \
            SolverWithComparator._partial_objective(
                index_to_row_to_column,
                left_index_to_right_index_to_probability,
                'left_index_to_right_index_to_is_left'
            )
        top_index_to_bottom_index_to_is_top, vertical_objective = \
            SolverWithComparator._partial_objective(
                index_to_row_to_column,
                top_index_to_bottom_index_to_probability,
                'top_index_to_bottom_index_to_is_top'
            )

        problem += lpSum(horizontal_objective + vertical_objective)

        left_index_to_right_index_to_row_to_column_to_is = SolverWithComparator._partial_constraints_add(
            problem,
            0, 1,
            index_to_row_to_column,
            left_index_to_right_index_to_is_left,
            'left_index_to_right_index_to_row_to_column_to_is'
        )

        top_index_to_bottom_index_to_row_to_column_to_is = SolverWithComparator._partial_constraints_add(
            problem,
            1, 0,
            index_to_row_to_column,
            top_index_to_bottom_index_to_is_top,
            'top_index_to_bottom_index_to_row_to_column_to_is'
        )

        # Each element has single position
        for index in range(t_square):
                problem += 1 == lpSum(index_to_row_to_column[index][row][column]
                                      for row in range(t)
                                      for column in range(t))

        # Each position has single stander
        for row in range(t):
            for column in range(t):
                problem += 1 == lpSum(index_to_row_to_column[index][row][column]
                                      for index in range(t_square))

        # Here comes the voodoo, you do not really need, for sanity or debug probably
        problem += t * (t - 1) == lpSum(SolverWithComparator._flatten(left_index_to_right_index_to_is_left))
        problem += t * (t - 1) == lpSum(SolverWithComparator._flatten(top_index_to_bottom_index_to_is_top))

        problem.writeLP('current-problem.lp')
        print('took {}s'.format(timeit('problem.solve()', number=1)))

        if LpStatusOptimal != problem.status:
            print('Warning: status is ', LpStatus[problem.status])

        prediction = [np.argmax([index_to_row_to_column[index][row][column].value()
                                 for row in range(t)
                                 for column in range(t)
                                 ])
                      for index in range(t_square)]

        return prediction

    @staticmethod
    def _partial_objective(index_to_row_to_column, first_to_second_to_probability, name):
        t_square = len(index_to_row_to_column)

        first_to_second_to_is = LpVariable.matrix(
            name,
            (list(range(t_square)), list(range(t_square))),
            lowBound=0,
            upBound=1,
            cat=LpInteger
        )

        objective = [
            lpDot(first_to_second_to_is[first][second], first_to_second_to_probability[first][second])
            for first in range(t_square)
            for second in range(t_square)
            if first != second
        ]

        return first_to_second_to_is, objective

    @staticmethod
    def _partial_constraints_add(problem,
                                 row_increment, column_increment,
                                 index_to_row_to_column, first_to_second_to_is,
                                 name):
        t_square = len(index_to_row_to_column)
        t = int(round(math.sqrt(t_square)))
        assert t ** 2 == t_square

        first_index_to_second_index_to_row_to_column = LpVariable.matrix(
            name,
            (list(range(t_square)), list(range(t_square)),
             list(range(t - row_increment)),  list(range(t - column_increment))),
            lowBound=0,
            upBound=1,
            cat=LpInteger
        )

        for first_index in range(t_square):
            for second_index in range(t_square):
                for row in range(t - row_increment):
                    for column in range(t - column_increment):
                        problem += \
                            first_index_to_second_index_to_row_to_column[first_index][second_index][row][column] <= \
                            index_to_row_to_column[first_index][row][column]
                        problem += \
                            first_index_to_second_index_to_row_to_column[first_index][second_index][row][column] <= \
                            index_to_row_to_column[second_index][row + row_increment][column + column_increment]

                problem += first_to_second_to_is[first_index][second_index] == \
                    lpSum(SolverWithComparator._flatten(
                        first_index_to_second_index_to_row_to_column[first_index][second_index]))

        return first_index_to_second_index_to_row_to_column

    @staticmethod
    def _flatten(list_of_variables) -> list:
        result = list()

        if isinstance(list_of_variables, list):
            for element in list_of_variables:
                result.extend(SolverWithComparator._flatten(element))
        else:
            result = [list_of_variables]

        return result

    @staticmethod
    def _predict_greedy_from_bottom_right(left_index_to_right_index_to_probability,
                                          top_index_to_bottom_index_to_probability):
        t_square = left_index_to_right_index_to_probability.shape[0]
        right_index_to_left_index_to_probability = np.transpose(left_index_to_right_index_to_probability)
        bottom_index_to_top_index_to_probability = np.transpose(top_index_to_bottom_index_to_probability)
        crop_position_in_transposed_image = SolverWithComparator._predict_greedy_from_top_left(
            right_index_to_left_index_to_probability,
            bottom_index_to_top_index_to_probability,
        )
        crop_position_in_original_image = list(map(lambda position: t_square - 1 - position,
                                                   crop_position_in_transposed_image))

        return crop_position_in_original_image

    @staticmethod
    def _predict_greedy_iterating_on_bottom_right(left_index_to_right_index_to_probability,
                                                  top_index_to_bottom_index_to_probability):
        t_square = left_index_to_right_index_to_probability.shape[0]
        right_index_to_left_index_to_probability = np.transpose(left_index_to_right_index_to_probability)
        bottom_index_to_top_index_to_probability = np.transpose(top_index_to_bottom_index_to_probability)
        crop_position_in_transposed_image = SolverWithComparator._predict_greedy_iterating_on_top_left(
            right_index_to_left_index_to_probability,
            bottom_index_to_top_index_to_probability,
        )
        crop_position_in_original_image = list(map(lambda position: t_square - 1 - position,
                                                   crop_position_in_transposed_image))

        return crop_position_in_original_image

    @staticmethod
    def _predict_greedy_from_top_left(left_index_to_right_index_to_probability,
                                      top_index_to_bottom_index_to_probability):
        t_square = left_index_to_right_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))
        assert t**2 == t_square
        top_left_index = -1
        top_left_probability = float("inf")

        for second_index in range(t_square):
            current_probability = max(max(
                left_index_to_right_index_to_probability[first_index][second_index],
                top_index_to_bottom_index_to_probability[first_index][second_index]
            ) for first_index in range(t_square) if first_index != second_index)

            if current_probability < top_left_probability:
                top_left_probability = current_probability
                top_left_index = second_index

        crop_position_in_original_image = SolverWithComparator._continue_greedy_given_top_left(
            top_left_index,
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability
        )

        return crop_position_in_original_image

    @staticmethod
    def _predict_greedy_iterating_on_top_left(left_index_to_right_index_to_probability,
                                              top_index_to_bottom_index_to_probability):
        t_square = left_index_to_right_index_to_probability.shape[0]
        best_objective = float("-inf")
        best_crop_position_in_original_image = list(range(t_square))

        for top_left_index in range(t_square):
            current_crop_position_in_original_image = SolverWithComparator._continue_greedy_given_top_left(
                top_left_index,
                left_index_to_right_index_to_probability,
                top_index_to_bottom_index_to_probability)

            current_objective, current_log_objective = \
                SolverWithComparator._compute_objective(current_crop_position_in_original_image,
                                                        left_index_to_right_index_to_probability,
                                                        top_index_to_bottom_index_to_probability)

            if current_log_objective > best_objective:
                best_objective = current_log_objective
                best_crop_position_in_original_image = current_crop_position_in_original_image

        return best_crop_position_in_original_image

    @staticmethod
    def _continue_greedy_given_top_left(top_left_index,
                                        left_index_to_right_index_to_probability,
                                        top_index_to_bottom_index_to_probability):
        t_square = top_index_to_bottom_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))
        order = [top_left_index, ]
        crop_position_in_original_image = [-1, ] * t_square
        crop_position_in_original_image[top_left_index] = 0

        for row in range(t):
            for column in range(1 if 0 == row else 0, t):
                best_second_index = -1
                best_second_probability = float("-inf")

                for second_index in range(t_square):
                    if second_index not in order:  # TODO: Should I?
                        if 0 == row:
                            is_bottom_probability = 1.0
                        else:
                            top_row = row - 1
                            top_column = column
                            top_index = order[top_row * t + top_column]
                            is_bottom_probability = top_index_to_bottom_index_to_probability[top_index][second_index]

                        if 0 == column:
                            is_right_probability = 1.0
                        else:
                            left_row = row
                            left_column = column - 1
                            left_index = order[left_row * t + left_column]
                            is_right_probability = left_index_to_right_index_to_probability[left_index][second_index]

                        current_probability = is_bottom_probability * is_right_probability

                        if current_probability > best_second_probability:
                            best_second_probability = current_probability
                            best_second_index = second_index

                crop_position_in_original_image[best_second_index] = len(order)
                order.append(best_second_index)

        return crop_position_in_original_image

    @staticmethod
    def _compute_objective(crop_position_in_original_image,
                           left_index_to_right_index_to_probability,
                           top_index_to_bottom_index_to_probability):
        t = int(round(math.sqrt(len(crop_position_in_original_image))))

        row_to_column_to_crop_index = np.empty((t, t), dtype=int)

        for crop_index, crop_position in enumerate(crop_position_in_original_image):
            row = crop_position // t
            column = crop_position % t
            row_to_column_to_crop_index[row][column] = crop_index

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
                        return float("-inf")

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

    def _predict_greedy(self,
                        left_index_to_right_index_to_probability,
                        top_index_to_bottom_index_to_probability):
        crop_position_in_original_image_1 = \
            self.__class__._predict_greedy_iterating_on_top_left(
                left_index_to_right_index_to_probability,
                top_index_to_bottom_index_to_probability)
        crop_position_in_original_image_2 = \
            self.__class__._predict_greedy_iterating_on_bottom_right(
                left_index_to_right_index_to_probability,
                top_index_to_bottom_index_to_probability)

        objective_1, log_objective_1 = self.__class__._compute_objective(
            crop_position_in_original_image_1,
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability)

        objective_2, log_objective_2 = self.__class__._compute_objective(
            crop_position_in_original_image_2,
            left_index_to_right_index_to_probability,
            top_index_to_bottom_index_to_probability)

        if log_objective_1 < log_objective_2:
            print('Using bottom right')
            crop_position_in_original_image = crop_position_in_original_image_2
        else:
            print('Using top left')
            crop_position_in_original_image = crop_position_in_original_image_2

        return crop_position_in_original_image


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

    width = 224
    height = 224

    for image_type in image_types:
        print(image_type.value)

        if image_type == ImageType.IMAGES:
            get_images = DataProvider().get_fish_images
        else:
            get_images = DataProvider().get_docs_images

        images, names = get_images(num_samples=number_of_samples, return_names=True)

        images_train, images_validation, names_train, names_validation = train_test_split(images, names,
                                                                                          random_state=42)
        t_to_comparator = {
            t: ComparatorCNN(t, width, height, image_type)
                ._fit_standardisation(images_train)
                .load_weights()
            for t in ts
        }

        clf = SolverWithComparator(t_to_comparator)
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

    slv = SolverWithComparator({t: cmp})
    score = slv.evaluate_image_for_permutation(images_validation[0], permutation, sample_index=index)
    print('done with ', score)


if __name__ == '__main__':
    # debug()
    main()