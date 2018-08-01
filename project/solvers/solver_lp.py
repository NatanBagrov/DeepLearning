import math
import sys
from timeit import timeit

import numpy as np
from sklearn.model_selection import train_test_split
from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, lpSum, lpDot, LpStatusOptimal, LpStatus

from constants import IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH
from solvers.generic_solver_with_comparator import GenericSolverWithComparator
from models.comparator_cnn import ComparatorCNN
from utils.image_type import ImageType
from utils.data_provider import DataProvider


class SolverLP(GenericSolverWithComparator):
    def _predict(self,
                 left_index_to_right_index_to_probability,
                 top_index_to_bottom_index_to_probability,
                 return_log_objective=False):
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
            SolverLP._partial_objective(
                index_to_row_to_column,
                left_index_to_right_index_to_probability,
                'left_index_to_right_index_to_is_left'
            )
        top_index_to_bottom_index_to_is_top, vertical_objective = \
            SolverLP._partial_objective(
                index_to_row_to_column,
                top_index_to_bottom_index_to_probability,
                'top_index_to_bottom_index_to_is_top'
            )

        problem += lpSum(horizontal_objective + vertical_objective)

        left_index_to_right_index_to_row_to_column_to_is = SolverLP._partial_constraints_add(
            problem,
            0, 1,
            index_to_row_to_column,
            left_index_to_right_index_to_is_left,
            'left_index_to_right_index_to_row_to_column_to_is'
        )

        top_index_to_bottom_index_to_row_to_column_to_is = SolverLP._partial_constraints_add(
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
        problem += t * (t - 1) == lpSum(SolverLP._flatten(left_index_to_right_index_to_is_left))
        problem += t * (t - 1) == lpSum(SolverLP._flatten(top_index_to_bottom_index_to_is_top))

        problem.writeLP('current-problem.lp')
        problem.solve()

        if LpStatusOptimal != problem.status:
            print('Warning: status is ', LpStatus[problem.status])

        prediction = [np.argmax([index_to_row_to_column[index][row][column].value()
                                 for row in range(t)
                                 for column in range(t)
                                 ])
                      for index in range(t_square)]

        return (prediction, None) if return_log_objective else prediction

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
             list(range(t - row_increment)), list(range(t - column_increment))),
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
                           lpSum(SolverLP._flatten(
                               first_index_to_second_index_to_row_to_column[first_index][second_index]))

        return first_index_to_second_index_to_row_to_column

    @staticmethod
    def _flatten(list_of_variables) -> list:
        result = list()

        if isinstance(list_of_variables, list):
            for element in list_of_variables:
                result.extend(SolverLP._flatten(element))
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
        epochs = 1

    ts = list()

    if '2' in sys.argv:
        ts.append(2)

    if '4' in sys.argv:
        ts.append(4)

    if '5' in sys.argv:
        ts = [5, ]

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
            t: ComparatorCNN(t,
                             IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].width,
                             IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].height,
                             image_type, mean=mean, std=std)
                .load_weights(IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].model_path)
            for t in ts
        }

        clf = SolverLP(t_to_comparator, image_type=image_type)
        print('Train: ', names_train)
        accuracy = clf.evaluate(images_train, epochs=epochs, ts=ts)
        print('Train 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))
        print('Validation: ', names_validation)
        accuracy = clf.evaluate(images_validation, epochs=epochs, ts=ts)
        print('Validation 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))


if __name__ == '__main__':
    main()
