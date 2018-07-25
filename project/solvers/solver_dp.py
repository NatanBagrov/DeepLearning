import math
import itertools
import time
import numpy as np

from solvers.generic_solver_with_comparator import GenericSolverWithComparator


class SolverGreedy(GenericSolverWithComparator):
    def __init__(self, t_to_comparator, image_type=None):
        GenericSolverWithComparator.__init__(self, t_to_comparator, image_type)

    def _predict(self,
                 left_index_to_right_index_to_probability,
                 top_index_to_bottom_index_to_probability,
                 return_log_objective=False,
                 verbose=True):
        if verbose:
            print('Preparing everything ')
        start = time.time()

        t_square = left_index_to_right_index_to_probability.shape[0]
        t = int(round(math.sqrt(t_square)))

        left_index_to_right_index_to_log_probability = \
            np.ma.log(left_index_to_right_index_to_probability).filled(-np.inf)
        top_index_to_bottom_index_to_log_probability = \
            np.ma.log(top_index_to_bottom_index_to_probability).filled(-np.inf)

        row_to_column_to_last_t_pieces_to_log_probability = np.full([t, t] + [t_square, ] * t, -np.inf)
        row_to_column_to_last_t_pieces_to_previous = np.empty([t, t] + [t_square, ] * t, dtype=np.int)
        combinations_of_t_from_t_square = list(itertools.combinations(range(t_square), t))

        if verbose:
            print('Initializing DP after', time.time() - start, 'seconds')

        for first_t in combinations_of_t_from_t_square:
            log_probability = 0.0

            for column in range(t - 1):
                log_probability += left_index_to_right_index_to_log_probability[first_t[column], first_t[column + 1]]

            row_to_column_to_last_t_pieces_to_log_probability[0, t - 1][first_t] = log_probability

        if verbose:
            print('Running DP after', time.time() - start, 'seconds')

        for row in range(1, t):
            for previous_last_t in itertools.combinations(range(t_square), t):
                old_log_probability = row_to_column_to_last_t_pieces_to_log_probability[row - 1, t - 1][previous_last_t]

                if old_log_probability > -np.inf:
                    for current in set(range(t_square)) - set(previous_last_t):
                        current_last_t = tuple(list(previous_last_t[1:]) + [current])
                        current_log_probability = \
                            old_log_probability + \
                            top_index_to_bottom_index_to_log_probability[previous_last_t[0]][current]

                        if current_log_probability > row_to_column_to_last_t_pieces_to_log_probability[row, 0][current_last_t]:
                            row_to_column_to_last_t_pieces_to_log_probability[row, 0][current_last_t] = current_log_probability
                            row_to_column_to_last_t_pieces_to_previous[row, 0][current_last_t] = previous_last_t[0]

            for column in range(1, t):
                for previous_last_t in combinations_of_t_from_t_square:
                    old_log_probability = \
                        row_to_column_to_last_t_pieces_to_log_probability[row, column - 1][previous_last_t]

                    if old_log_probability > -np.inf:
                        for current in set(range(t_square)) - set(previous_last_t):
                            current_last_t = tuple(list(previous_last_t[1:]) + [current])
                            current_log_probability = \
                                old_log_probability + \
                                top_index_to_bottom_index_to_log_probability[previous_last_t[0]][current] + \
                                left_index_to_right_index_to_log_probability[previous_last_t[-1]][current]

                            if current_log_probability > \
                                    row_to_column_to_last_t_pieces_to_log_probability[row, column][current_last_t]:
                                row_to_column_to_last_t_pieces_to_log_probability[row, column][current_last_t] = \
                                    current_log_probability
                                row_to_column_to_last_t_pieces_to_previous[row, column][current_last_t] = \
                                    previous_last_t[0]

        if verbose:
            print('Building solution after', time.time() - start, 'seconds')

        row_to_column_to_shred_index = np.empty((t, t), dtype=np.int)

        last_t = np.unravel_index(
            np.argmax(row_to_column_to_last_t_pieces_to_log_probability[t - 1, t - 1]),
            row_to_column_to_last_t_pieces_to_log_probability[t - 1, t - 1].shape
        )

        log_objective = row_to_column_to_last_t_pieces_to_log_probability[t - 1, t - 1][last_t]

        if verbose:
            print('Log objective is', log_objective)

        row_to_column_to_shred_index[t - 1] = last_t

        for row in reversed(range(t - 1)):
            for column in reversed(range(t)):
                previous = row_to_column_to_last_t_pieces_to_previous[row + 1, column][last_t]
                row_to_column_to_shred_index[row][column] = previous
                last_t = tuple([previous, ] + list(last_t[: - 1]))

        shred_index_to_original_index = \
            self.__class__._row_to_column_to_shred_index_to_shred_index_to_original_index(row_to_column_to_shred_index)

        # true_objective, true_log_objective = self.__class__._compute_objective(
        #     shred_index_to_original_index,
        #     left_index_to_right_index_to_probability,
        #     top_index_to_bottom_index_to_probability)
        #
        # assert np.isclose(true_log_objective, log_objective)

        if verbose:
            print('All done after', time.time() - start, 'seconds')

        return shred_index_to_original_index


def main():
    import sys
    from utils.image_type import ImageType
    from utils.data_provider import DataProvider
    from models.comparator_cnn import ComparatorCNN
    from sklearn.model_selection import train_test_split

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

    width = 2200 // 5
    height = 2200 // 5
    # width = 224
    # height = 224

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


if __name__ == '__main__':
    # debug()
    main()

