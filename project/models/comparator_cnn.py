import sys
import copy

import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.activations import relu, softmax
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, Dense, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import ModelCheckpoint, Callback

from utils.image_type import ImageType
from utils.visualizer import PlotCallback, Visualizer
from utils.data_manipulations import shred_and_resize_to
from utils.data_provider import DataProvider
from models.generic_cnn import GenericCNN


class ComparatorCNN(GenericCNN):
    """
    Network checks if one picture lies to the left of another
    """
    def __init__(self, t, width, height, image_type: ImageType, mean=None, std=None):
        super().__init__(t, width, height, image_type, mean=mean, std=std)
        self._model = self.__class__._build_model(width, height, post_activation_bn=True)

    @staticmethod
    def _build_model(width, height,
                     pre_activation_bn=False, post_activation_bn=False,
                     input_depth=2, classes=2, padding='same',
                     learning_rate=1e-3, learning_rate_decay=1e-6):
        def maybe_bn(is_pre_activation):
            if is_pre_activation and pre_activation_bn or post_activation_bn:
                return [BatchNormalization()]
            return list()

        def activation():
            return [Activation(relu)]

        def convolution(filters, **kwargs):
            return \
                [Conv2D(
                    filters,
                    (3, 3),
                    padding=padding,
                    **kwargs
                )] + \
                maybe_bn(True) + \
                activation() + \
                maybe_bn(False)

        def max_pooling():
            return [MaxPooling2D(
                padding=padding
            )]

        def fully_connected(units):
            return \
                [Dense(units)] + \
                maybe_bn(True) + \
                activation() + \
                maybe_bn(False)

        def output_layer():
            return [Dense(classes), Activation(softmax)]

        def flatten():
            return [Flatten()]

        # model = Sequential(
        #     convolution(64, input_shape=(height, width, input_depth)) +
        #     convolution(64) +
        #     max_pooling() +
        #     convolution(128) +
        #     convolution(128) +
        #     max_pooling() +
        #     convolution(256) +
        #     convolution(256) +
        #     convolution(256) +
        #     max_pooling() +
        #     convolution(512) +
        #     convolution(512) +
        #     convolution(512) +
        #     max_pooling() +
        #     convolution(512) +
        #     convolution(512) +
        #     convolution(512) +
        #     max_pooling() +
        #     flatten() +
        #     fully_connected(4096) +
        #     fully_connected(4096) +
        #     output_layer()
        # )

        model = Sequential(
            convolution(8, input_shape=(height, width, input_depth)) +
            convolution(8) +
            max_pooling() +
            convolution(16) +
            convolution(16) +
            max_pooling() +
            convolution(32) +
            convolution(32) +
            convolution(32) +
            max_pooling() +
            convolution(64) +
            convolution(64) +
            convolution(64) +
            max_pooling() +
            convolution(128) +
            convolution(128) +
            convolution(128) +
            max_pooling() +
            flatten() +
            fully_connected(512) +
            fully_connected(512) +
            output_layer()
        )

        model.summary()

        optimizer = Adam(
            lr=learning_rate,
            decay=learning_rate_decay
        )

        loss = binary_crossentropy

        metrics = [
            binary_accuracy
        ]

        model.compile(
            optimizer,
            loss,
            metrics=metrics,
        )

        return model

    def predict_is_left_probability(self, left, right, standardise=True):
        tensor = ComparatorCNN._prepare_left_right_check(left, right)

        return self.predict_probability(tensor, standardise=standardise)

    def predict_is_top_probability(self, top, bottom, standardise=True):
        tensor = ComparatorCNN._prepare_top_bottom_check(top, bottom)

        return self.predict_probability(tensor, standardise=standardise)

    @staticmethod
    def _prepare_left_right_check(left, right):
        right = np.flip(right, axis=-1)

        return np.stack((left, right), axis=-1)

    @staticmethod
    def _prepare_top_bottom_check(top, bottom):
        top = np.rot90(top, axes=(-2, -1))
        bottom = np.rot90(bottom, axes=(-2, -1))

        tensor = ComparatorCNN._prepare_left_right_check(top, bottom)

        return tensor

    @staticmethod
    def _is_same_patch(patch_a, patch_b):
        return np.linalg.norm(patch_a - patch_b) < 1e-12  # TODO: EPSILON IS HARDCODED

    @staticmethod
    def _generate_regular_shreds_stratified(images: list, width, height, t, batch_size=None):
        number_of_samples = len(images)
        sample_index_to_shred_index_to_image = shred_and_resize_to(images, t, (width, height))
        print('Mean: {} std: {} shape: {} type: {}'.format(
            np.mean(sample_index_to_shred_index_to_image),
            np.std(sample_index_to_shred_index_to_image),
            np.shape(sample_index_to_shred_index_to_image),
            np.array(sample_index_to_shred_index_to_image).dtype
        ))
        assert (number_of_samples, t ** 2, height, width) == sample_index_to_shred_index_to_image.shape
        probabilities_regular_pattern = np.full((t * t), 1.0 / (2.0 * (t**2 - 1.0)))
        probabilities_edge = np.full((t * t), 1.0 / (t ** 2))
        options = np.arange(t * t)

        if batch_size is None:
            batch_size = 2 * number_of_samples * (t ** 2)

        while True:
            for batch_offset in range(0, 2 * number_of_samples * t * t, batch_size):
                x = list()
                y = list()

                for batch_index in range(batch_size):
                    while len(x) <= batch_index:
                        is_top = np.random.choice((False, True))
                        sample = np.random.choice(len(images))
                        row, col = np.random.choice(t, size=2)

                        if is_top:
                            neighbour_row = row + 1
                            neighbour_col = col
                            is_edge = neighbour_row == t
                        else:
                            neighbour_row = row
                            neighbour_col = col + 1
                            is_edge = neighbour_col == t

                        if is_edge:
                            p = probabilities_edge
                        else:
                            assert options[neighbour_row * t + neighbour_col] == neighbour_row * t + neighbour_col
                            probabilities_regular_pattern[neighbour_row * t + neighbour_col] = 1.0 / 2.0
                            p = probabilities_regular_pattern

                        second_row_col = np.random.choice(options, p=p)
                        second_row = second_row_col // t
                        second_col = second_row_col % t

                        if not is_edge:
                            probabilities_regular_pattern[neighbour_row * t + neighbour_col] = 1.0 / (2.0 * (t**2 - 1.0))

                        image_above = sample_index_to_shred_index_to_image[sample, row * t + col]
                        image_beyond = sample_index_to_shred_index_to_image[sample, second_row * t + second_col]
                        same = ComparatorCNN._is_same_patch(image_above, image_beyond)

                        if same:
                            print('Same patch is sampled at ({},{}) and ({},{}). Skipping'.format(
                                row, col,
                                second_row, second_col))
                            continue

                        if is_top:
                            tensor = ComparatorCNN._prepare_top_bottom_check(image_above, image_beyond)
                        else:
                            tensor = ComparatorCNN._prepare_left_right_check(image_above, image_beyond)

                        assert (height, width, 2) == tensor.shape, "{}!={}".format((height, width, 2), tensor.shape)

                        x.append(tensor)

                        if neighbour_row == second_row and neighbour_col == second_col:
                            one_hot = [0, 1]
                        else:
                            one_hot = [1, 0]

                        y.append(one_hot)

                x = np.stack(x)
                y = np.stack(y)

                assert (batch_size, height, width, 2) == x.shape
                assert (batch_size, 2) == y.shape

                yield x, y


def main():
    if 'debug' in sys.argv:
        print('Debug')
        number_of_samples = 20
        epochs = 5
    else:
        print('Release')
        number_of_samples = sys.maxsize
        epochs = 200

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

    if 'train' in sys.argv:
        force = True
    elif 'evaluate' in sys.argv:
        force = False
    else:
        force = False

    np.random.seed(42)

    width = 2200 // 5
    height = 2200 // 5
    batch_size = 32

    for t in ts:
        for image_type in image_types:
            print('t={}. image type is {}'.format(t, image_type.value))

            if image_type == ImageType.IMAGES:
                get_images = DataProvider().get_fish_images
            else:
                get_images = DataProvider().get_docs_images

            images = get_images(num_samples=number_of_samples)
            images_train, images_validation = train_test_split(images, random_state=42)

            clf = ComparatorCNN(t, width, height, image_type)

            if force:
                clf.fit_generator(
                    images_train,
                    batch_size,
                    epochs,
                    images_validation,
                )
            else:
                clf.load_weights()
                clf._fit_standardisation(images_train)

            print('Train 0-1:', clf.evaluate(images_train))
            print('Validation 0-1:', clf.evaluate(images_validation))


if __name__ == '__main__':
    main()
