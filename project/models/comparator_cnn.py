import sys

import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.activations import relu, softmax
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, Dense, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import ModelCheckpoint

from utils.image_type import ImageType
from utils.visualizer import PlotCallback
from utils.data_manipulations import shred_and_resize_to
from utils.data_provider import DataProvider


class ComparatorCNN:
    """
    Network checks if one picture lies to the left of another
    """
    def __init__(self, t, width, height, image_type: ImageType):
        self._t = t
        self._width = width
        self._height = height
        self._image_type = image_type
        self._model = ComparatorCNN._build_model(width, height)

    @staticmethod
    def _build_model(width, height,
                     pre_activation_bn=False, post_activation_bn=False,
                     input_depth=2, classes=2, padding='same',
                     learning_rate=1e-3, learning_rate_decay=1e-6):
        def maybe_bn(is_pre_activation):
            if is_pre_activation and pre_activation_bn or post_activation_bn:
                return [BatchNormalization]
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

        model = Sequential(
            convolution(64, input_shape=(height, width, input_depth)) +
            convolution(64) +
            max_pooling() +
            convolution(128) +
            convolution(128) +
            max_pooling() +
            convolution(256) +
            convolution(256) +
            convolution(256) +
            max_pooling() +
            convolution(512) +
            convolution(512) +
            convolution(512) +
            max_pooling() +
            convolution(512) +
            convolution(512) +
            convolution(512) +
            max_pooling() +
            flatten() +
            fully_connected(4096) +
            fully_connected(4096) +
            output_layer()
        )

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

    def fit_generator(self, images_train: list, batch_size, epochs, images_validation:list):
        train_generator = ComparatorCNN._generate_regular_shreds_stratified(images_train, self._width, self._height, self._t, batch_size)
        # generator = ComparatorCNN.generate_all_shreds(images)
        validation_generator = ComparatorCNN._generate_regular_shreds_stratified(images_validation, self._width, self._height, self._t, batch_size)
        validation_data = next(validation_generator)
        print('Part of true in validation dataset is ', np.mean(validation_data[1][1]))
        steps_per_epoch = 2 * len(images_train) * (self._t ** 2) / batch_size

        self._model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=2,
            callbacks=[
                PlotCallback(['loss', 'val_loss'],
                             file_path='graphs/one_picture_classify_{}_{}_loss.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                PlotCallback(['binary_accuracy', 'val_binary_accuracy'],
                             file_path='graphs/one_picture_classify_{}_{}_acc.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                ModelCheckpoint(self._get_model_checkpoint_file_path(),
                                monitor='val_binary_accuracy',
                                save_best_only=True)

            ],
            validation_data=validation_data
        )

    def predict_is_left(self, left, right):
        tensor = ComparatorCNN._prepare_left_right_check(left, right)

        return self.predict(tensor)

    def predict_is_top(self, top, bottom):
        tensor = ComparatorCNN._prepare_top_bottom_check(top, bottom)

        return self.predict(tensor)

    def predict(self, tensor):
        soft_prediction = self._model.predict(tensor)
        hard_prediction = np.argmax(soft_prediction, axis=-1)

        return hard_prediction

    def evaluate(self, images: list):
        x, y = next(ComparatorCNN._generate_regular_shreds_stratified(images, self._width, self._height, self._t))
        print('True in evaluation dataset is {} * {}', np.mean(y[1]), y.shape[0])
        y_true = np.argmax(y, axis=-1)
        y_predicted = self.predict(x)

        return np.mean(y_true == y_predicted)

    def _get_model_checkpoint_file_path(self):
        return 'saved_weights/comparator-best-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )

    @staticmethod
    def _prepare_left_right_check(left, right):
        right = np.flip(right, axis=-1)

        return np.stack((left, right), axis=-1)

    @staticmethod
    def _prepare_top_bottom_check(top, bottom):
        top = np.rot90(top, axes=(-2, -1))
        bottom = np.rot90(bottom, axes=(-2, -1))

        return ComparatorCNN._prepare_left_right_check(top, bottom)

    @staticmethod
    def _generate_regular_shreds_stratified(images:list, width, height, t, batch_size=None):
        number_of_samples = len(images)
        sample_index_to_shred_index_to_image = shred_and_resize_to(images, t, (width, height))
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

                    if is_top:
                        tensor = ComparatorCNN._prepare_top_bottom_check(image_above, image_beyond)
                    else:
                        tensor = ComparatorCNN._prepare_left_right_check(image_above, image_beyond)

                    assert (height, width, 2) == tensor.shape

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


if __name__ == '__main__':
    if 'debug' in sys.argv:
        print('Debug')
        number_of_samples = 20
        epochs = 5
    else:
        print('Release')
        number_of_samples = sys.maxsize
        epochs = 200

    np.random.seed(42)

    width = 224
    height = 224
    batch_size = 32
    force = True

    for t in (
            2,
            4,
            5):
        for image_type in ImageType:
            print('t={}. image type is {}'.format(t, image_type.value))

            if image_type == ImageType.IMAGES:
                get_images = DataProvider().get_fish_images
            else:
                get_images = DataProvider().get_docs_images

            images = get_images(num_samples=number_of_samples)
            images_train, images_validation = train_test_split(images, random_state=42)

            clf = ComparatorCNN(t, width, height, image_type)
            clf.fit_generator(
                images_train,
                batch_size,
                epochs,
                images_validation,
            )
            print('Train 0-1:', clf.evaluate(images_train))
            print('Validation 0-1:', clf.evaluate(images_validation))
