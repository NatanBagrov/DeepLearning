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


class TopLeftCNN:
    """
    Network classifies which picture lies on top left
    """
    def __init__(self, t, width, height, image_type: ImageType, mean=None, std=None):
        self._t = t
        self._width = width
        self._height = height
        self._image_type = image_type
        self._mean = mean
        self._std = std
        self._model = TopLeftCNN._build_model(width, height, post_activation_bn=True)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

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

        # model.summary()

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

    # TODO: move to superclass
    def standardise(self, images):
        if isinstance(images, list):
            images = copy.deepcopy(images)
        elif isinstance(images, np.ndarray):
            images = np.copy(images).astype(np.float)
        else:
            assert False

        for index, current_image in enumerate(images):
            images[index] = (current_image - self._mean) / self._std

        return images

    # TODO: move to superclass
    def fit_generator(self, images_train: list, batch_size, epochs, images_validation:list):
        self._fit_standardisation(images_train)
        images_train = self.standardise(images_train)
        images_validation = self.standardise(images_validation)
        train_generator = self.__class__._generate_regular_shreds(images_train, self.width, self.height, self._t, batch_size)
        # generator = ComparatorCNN.generate_all_shreds(images)
        validation_generator = self.__class__._generate_regular_shreds(images_validation, self.width, self.height, self._t, batch_size)
        validation_data = next(validation_generator)
        print('Part of true in validation dataset is ', np.mean(validation_data[1][1]), ' of ', validation_data[1].shape[0])
        steps_per_epoch = 2 * len(images_train) * (self._t ** 2) / batch_size
        print('Train mean is {} std is {} number is {}. Validation mean is {} std is {} number is {}'.format(
            self.__class__._mean_of_a_list(images_train), self.__class__._std_of_a_list(images_train), len(images_train),
            self.__class__._mean_of_a_list(images_validation), self.__class__._std_of_a_list(images_validation), len(images_validation),))

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
                             file_path='graphs/comparator_cnn_{}_{}_loss.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                PlotCallback(['binary_accuracy', 'val_binary_accuracy'],
                             file_path='graphs/comparator_cnn_{}_{}_acc.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                ModelCheckpoint(self._get_model_checkpoint_file_path(),
                                monitor='val_loss',
                                save_best_only=True),
                UpdateMonitorCallback(False)

            ],
            validation_data=validation_data
        )

    # TODO: move to superclass
    def _fit_standardisation(self, images):
        # TODO: save it somewhere!!
        self._mean = ComparatorCNN._mean_of_a_list(images)
        self._std = ComparatorCNN._std_of_a_list(images)

        print('Mean is {}. Std is {}'.format(self._mean, self._std))

        assert not np.isclose(self._std, 0.0), 'Variance is too low'

        return self

    #TODO: move to utils
    @staticmethod
    def _mean_of_a_list(images_list):
        mean = np.mean(np.concatenate(list(map(np.ndarray.flatten, images_list))))

        return mean

    @staticmethod
    def _std_of_a_list(images_list):
        std = np.std(np.concatenate(list(map(np.ndarray.flatten, images_list))))

        return std

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

        if batch_size is None:
            batch_size = number_of_samples

        while True:
            permutation = np.random.permutation(number_of_samples)
            sample_index_to_shred_index_to_image = sample_index_to_shred_index_to_image[permutation]

            for batch_offset in range(0, number_of_samples, batch_size):
                batch_x = list()
                batch_y = list()

                for index in range(batch_offset, min(batch_offset + batch_size, number_of_samples)):
                    permutation = np.random.permutation(t ** 2)
                    y = 0 == permutation
                    x = sample_index_to_shred_index_to_image[index, permutation]
                    batch_x.append(x)
                    batch_y.append(y)

                batch_x = np.stack(batch_x)
                assert batch_x.shape == (min(batch_size, number_of_samples - batch_offset), t ** 2, height, width)
                batch_y = np.stack(batch_y)
                assert batch_y.shape == (min(batch_size, number_of_samples - batch_offset), t ** 2)




