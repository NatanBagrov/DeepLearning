import math
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, lpSum, lpDot, LpStatusOptimal, LpStatus

from utils.shredder import Shredder
from utils.data_manipulations import resize_to, shred_and_resize_to
from utils.data_provider import DataProvider
from utils.image_type import ImageType
from models.comparator_cnn import ComparatorCNN


class SolverWithComparator:
    def __init__(self, t_to_comparator):
        self._t_to_comparator = t_to_comparator

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

        prediction = SolverWithComparator._predict_with_lp(
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
                    shreds_permuted = shred_index_to_image[permutation]
                    permutation_predicted = self.predict(shreds_permuted)
                    current_accuracy = np.average(permutation_predicted == permutation)
                    print('For {} 0-1 is {}'.format(sample_index, current_accuracy))
                    accuracies.append(current_accuracy)

            current_accuracy = np.average(accuracies)
            print('For t={} 0-1 is {}'.format(t, current_accuracy))
            index_to_accuracy.append(current_accuracy)

        return np.average(index_to_accuracy)

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

        problem = LpProblem("permutation", LpMaximize)
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
        problem.solve()

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

    for image_type in ImageType:
        print(image_type.value)

        if image_type == ImageType.IMAGES:
            get_images = DataProvider().get_fish_images
        else:
            get_images = DataProvider().get_docs_images

        images = get_images(num_samples=number_of_samples)
        images_train, images_validation = train_test_split(images, random_state=42)

        t_to_comparator = {
            t: ComparatorCNN(t, width, height, image_type)
                ._fit_standardisation(images_train)
                .load_weights()
            for t in ts
        }

        clf = SolverWithComparator(t_to_comparator)
        print('Train: ')
        accuracy = clf.evaluate(images_train, epochs=epochs)
        print('Train 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))
        print('Validation: ')
        accuracy = clf.evaluate(images_validation, epochs=epochs)
        print('Validation 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))


if __name__ == '__main__':
    main()