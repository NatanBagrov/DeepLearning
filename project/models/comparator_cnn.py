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
        self._model = ComparatorCNN._build_model(width, height,
                                                 post_activation_bn=True)

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

    def fit_generator(self, images_train: list, batch_size, epochs, images_validation:list):
        self._fit_standardisation(images_train)
        images_train = self.standardise(images_train)
        images_validation = self.standardise(images_validation)
        train_generator = ComparatorCNN._generate_regular_shreds_stratified(images_train, self._width, self._height, self._t, batch_size)
        # generator = ComparatorCNN.generate_all_shreds(images)
        validation_generator = ComparatorCNN._generate_regular_shreds_stratified(images_validation, self._width, self._height, self._t, batch_size)
        validation_data = next(validation_generator)
        print('Part of true in validation dataset is ', np.mean(validation_data[1][1]), ' of ', validation_data[1].shape[0])
        steps_per_epoch = 2 * len(images_train) * (self._t ** 2) / batch_size
        print('Train mean is {} std is {} number is {}. Validation mean is {} std is {} number is {}'.format(
            ComparatorCNN._mean_of_a_list(images_train), ComparatorCNN._std_of_a_list(images_train), len(images_train),
            ComparatorCNN._mean_of_a_list(images_validation), ComparatorCNN._std_of_a_list(images_validation), len(images_validation),))

        class UpdateMonitorCallback(Callback):
            def __init__(self, should_monitor_updates):
                super().__init__()
                self._should_monitor_updates = should_monitor_updates
                self._weights_before_batch = None

            def on_batch_begin(self, batch, logs=None):
                if self._should_monitor_updates:
                    self._weights_before_batch = self._get_weights()

            def on_batch_end(self, batch, logs=None):
                if self._should_monitor_updates:
                    weights_after_batch = self._get_weights()
                    update = list(map(lambda weights: np.linalg.norm(weights[1] - weights[0]) / np.linalg.norm(weights[0]),
                                      zip(self._weights_before_batch, weights_after_batch)))
                    print('Update magnitudes mean is {}, they (should be ~1e-3) are: {}'.format(np.mean(update), update))

            def _get_weights(self):
                weights = list()

                for layer in self.model.model.layers:
                    current_weights = layer.get_weights()

                    if len(current_weights) >= 1:
                        weights.append(current_weights[0])

                return weights

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
                                save_best_only=True),
                UpdateMonitorCallback(False)

            ],
            validation_data=validation_data
        )

    def _fit_standardisation(self, images):
        self._mean = ComparatorCNN._mean_of_a_list(images)
        self._std = ComparatorCNN._std_of_a_list(images)

        print('Mean is {}. Std is {}'.format(self._mean, self._std))

        assert not np.isclose(self._std, 0.0), 'Variance is too low'

    @staticmethod
    def _mean_of_a_list(images_list):
        mean = np.mean(np.concatenate(list(map(np.ndarray.flatten, images_list))))

        return mean

    @staticmethod
    def _std_of_a_list(images_list):
        std = np.std(np.concatenate(list(map(np.ndarray.flatten, images_list))))

        return std

    def standardise(self, images):
        images = copy.deepcopy(images)

        for index, current_image in enumerate(images):
            images[index] = (current_image - self._mean) / self._std

        return images

    def predict_is_left(self, left, right, standardise=True):
        tensor = ComparatorCNN._prepare_left_right_check(left, right)

        return self.predict(tensor, standardise=standardise)

    def predict_is_top(self, top, bottom, standardise=True):
        tensor = ComparatorCNN._prepare_top_bottom_check(top, bottom)

        return self.predict(tensor, standardise=standardise)

    def predict(self, tensor, standardise=True):
        if standardise:
            tensor = self.standardise(tensor)

        print('Mean after standardization is {}, std is {}, shape is {}'.format(np.mean(tensor), np.std(tensor), np.shape(tensor)))

        soft_prediction = self._model.predict(tensor)
        hard_prediction = np.argmax(soft_prediction, axis=-1)

        return hard_prediction

    def evaluate(self, images: list, standardise=True):
        print('Mean of evalutaion images is {}, std is {}'.format(
            ComparatorCNN._mean_of_a_list(images),
            ComparatorCNN._std_of_a_list(images)))
        x, y = next(ComparatorCNN._generate_regular_shreds_stratified(images, self._width, self._height, self._t))
        print('True in evaluation dataset is {} * {}', np.mean(y[1]), y.shape[0])
        print('Mean before standardiztion is {}. Std is {}. Shape is {}'.format(np.mean(x), np.std(x), np.shape(x)))
        y_true = np.argmax(y, axis=-1)
        y_predicted = self.predict(x, standardise=standardise)

        return np.mean(y_true == y_predicted)

    def load_weights(self, file_path=None):
        if file_path is None:
            file_path = self._get_model_checkpoint_file_path()

        self._model.load_weights(file_path)

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

    ts = list()

    if '2' in sys.argv:
        ts.append(2)

    if '4' in sys.argv:
        ts.append(4)

    if '5' in sys.argv:
        ts = [5,]

    if 0 == len(ts):
        ts = (2, 4, 5)

    np.random.seed(42)

    width = 224
    height = 224
    batch_size = 32
    force = True

    for t in ts:
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
