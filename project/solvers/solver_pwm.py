import itertools
import math
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from models.comparator_cnn import ComparatorCNN
from solvers.generic_solver_with_comparator import GenericSolverWithComparator
from utils.data_provider import DataProvider
from utils.image_type import ImageType


class SolverPairwiseMerge(GenericSolverWithComparator):
    def __init__(self, t_to_comparator, t_to_backup_solver, image_type=None):
        GenericSolverWithComparator.__init__(self, t_to_comparator, image_type)
        self._t_to_backup_solver = t_to_backup_solver

    def _predict(self,
                 left_index_to_right_index_to_probability,
                 top_index_to_bottom_index_to_probability,
                 return_log_objective=False):

        t_square = left_index_to_right_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))

        try:
            rows_list = \
                self._merge_crops_given_probability_matrix(t, np.copy(left_index_to_right_index_to_probability))
        except NotImplementedError:
            print("SolverPairwiseMerge failed to construct rows, will use backup solver")
            return self._t_to_backup_solver._predict(left_index_to_right_index_to_probability,
                                                     top_index_to_bottom_index_to_probability,
                                                     return_log_objective)
        btt_probas = np.transpose(np.copy(top_index_to_bottom_index_to_probability))
        m = SolverPairwiseMerge._construct_merged_bottom_to_top_probability_matrix(rows_list, t, btt_probas)
        try:
            column_list = self._merge_crops_given_probability_matrix(1, np.copy(m))[0]
            permutation = list(itertools.chain(*[rows_list[c] for c in column_list]))
            return permutation
        except NotImplementedError:
            print("SolverPairwiseMerge failed to construct final permutation, will use backup solver")
            return self._t_to_backup_solver._predict(left_index_to_right_index_to_probability,
                                                     top_index_to_bottom_index_to_probability,
                                                     return_log_objective)

    def _merge_crops_given_probability_matrix(self, merged_groups_number, probability_matrix):
        merged_lists = [[] for _ in range(merged_groups_number)]

        def get_empty_merged_list_index():
            for idx in range(len(merged_lists)):
                if len(merged_lists[idx]) == 0:
                    return idx
            return None

        merged_indices = [None] * probability_matrix.shape[0]  # here, crops should point to their corresponding row

        while None in merged_indices:  # NOTE THAT LEFT AND RIGHT ARE FOR CONVENIENCE. USED ALSO AS BOTTOM AND TOP.
            left_idx, right_idx = np.unravel_index(np.argmax(probability_matrix, axis=None),
                                                   probability_matrix.shape)

            if merged_indices[left_idx] is None and merged_indices[right_idx] is None:
                # both left and right were not assigned to any group. we should 'open a row' for them.
                i = get_empty_merged_list_index()
                if i is None:
                    # Here, we got a problem, we filled all rows, but need to 'open' a new one.
                    raise NotImplementedError("Use backup solver")
                merged_lists[i].extend([left_idx, right_idx])
                merged_indices[left_idx] = merged_indices[right_idx] = i

            elif merged_indices[left_idx] is not None:
                # we will assign right to left's group, at the rightmost position
                row_idx = merged_indices[left_idx]
                row = merged_lists[row_idx]
                if merged_indices[right_idx] is not None:
                    # Here, we try to assign already assigned pieces, we should not get here
                    raise NotImplementedError("Use backup solver")
                if left_idx != row[-1]:
                    # Here, left is not the current rightmost, meaning it has a right neighbor, problem.
                    raise Exception("This should never happen, since we zeroed left's row before")
                row.append(right_idx)
                merged_indices[right_idx] = row_idx

            elif merged_indices[right_idx] is not None:
                # we will assign left to right's group, at the leftmost position
                row_idx = merged_indices[right_idx]
                row = merged_lists[row_idx]
                if merged_indices[left_idx] is not None:
                    # Here, we try to assign already assigned pieces, we should not get here
                    # Actually, we'll never get here since we already covered this above, so it is for symmetry.
                    raise NotImplementedError("Use backup solver")
                if right_idx != row[0]:
                    # Here, right is not the current leftmost, meaning it has a left neighbor, problem.
                    raise Exception("This should never happen, since we zeroed rights's column before")
                row.insert(0, left_idx)
                merged_indices[left_idx] = row_idx

            else:
                raise NotImplementedError("I missed something apparently :(")

            # if we got here, we are cool!
            # left crop can not be adjacent to anything from left, right crop can not be adjacent to anything from right
            probability_matrix[left_idx][:] = 0
            probability_matrix[:][right_idx] = 0

        return merged_lists

    @staticmethod
    def _construct_merged_bottom_to_top_probability_matrix(merged_lists, t, probability_matrix):
        bottom_row_to_top_row_adj_matrix = np.zeros((t, t))  # (i,j) = probability of row i to fit below row j
        for bottom_idx in range(t):
            for top_idx in range(t):
                bottom_list = merged_lists[bottom_idx]
                top_list = merged_lists[top_idx]
                for bottom_crop, top_crop in zip(bottom_list, top_list):
                    bottom_row_to_top_row_adj_matrix[bottom_idx][top_idx] += probability_matrix[bottom_crop][top_crop]
        res = bottom_row_to_top_row_adj_matrix / t
        np.fill_diagonal(res, 0.0)
        return res


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

    width = 224
    height = 224

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

        clf = SolverPairwiseMerge(t_to_comparator, t_to_backup_solver=None, image_type=image_type)
        print('Train: ', names_train)
        accuracy = clf.evaluate(images_train, epochs=epochs, ts=ts)
        print('Train 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))
        print('Validation: ', names_validation)
        accuracy = clf.evaluate(images_validation, epochs=epochs, ts=ts)
        print('Validation 0-1 accuracy on {}: {}'.format(image_type.value, accuracy))


if __name__ == '__main__':
    main()