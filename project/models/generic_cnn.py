import sys
import os
import time
import copy
from abc import ABC, abstractmethod, abstractstaticmethod

import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.activations import relu, softmax
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, Dense, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

from utils.image_type import ImageType
from utils.visualizer import PlotCallback, Visualizer
from utils.data_manipulations import shred_and_resize_to
from utils.data_provider import DataProvider


class GenericCNN(ABC):
    def __init__(self, t, width, height, image_type: ImageType, mean=None, std=None):
        self._t = t
        self._width = width
        self._height = height
        self._image_type = image_type
        self._mean = mean
        self._std = std
        self._model = None

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def fit_generator(self, images_train: list, batch_size, epochs, images_validation: list):
        self._fit_standardisation(images_train)
        images_train = self.standardise(images_train)
        images_validation = self.standardise(images_validation)
        train_generator = self.__class__._generate_regular_shreds_stratified(images_train,
                                                                             self.width, self.height, self._t,
                                                                             batch_size)
        # generator = ComparatorCNN.generate_all_shreds(images)
        validation_generator = self.__class__._generate_regular_shreds_stratified(images_validation,
                                                                                  self.width, self.height, self._t,
                                                                                  batch_size)
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

        timestamp = str(int(time.time()))
        print('time_stamp=', timestamp)
        log_directory_path = os.path.join('logs', '{}-{}-{}-{}'.format(
            self.__class__.__name__,
            self._t,
            self._image_type.value,
            timestamp
        ))

        self._model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=[
                PlotCallback(['loss', 'val_loss'],
                             file_path='graphs/{}_{}_{}_loss.png'.format(
                                 self.__class__.__name__,
                                 self._t,
                                 self._image_type.value),
                             show=True),
                PlotCallback(['binary_accuracy', 'val_binary_accuracy'],
                             file_path='graphs/{}_{}_{}_acc.png'.format(
                                 self.__class__.__name__,
                                 self._t,
                                 self._image_type.value),
                             show=True),
                ModelCheckpoint(self._get_model_checkpoint_file_path(),
                                monitor='val_loss',
                                save_best_only=True),
                TensorBoard(log_dir=log_directory_path),
                UpdateMonitorCallback(False)

            ],
            validation_data=validation_data
        )

    def predict(self, tensor, standardise=True):
        soft_prediction = self.predict_probability(tensor, standardise=standardise)
        hard_prediction = np.argmax(soft_prediction, axis=-1)

        return hard_prediction

    def predict_probability(self, tensor, standardise=True):
        if standardise:
            tensor = self.standardise(tensor)

        # print('Mean after standardization is {}, std is {}, shape is {}'.format(np.mean(tensor), np.std(tensor), np.shape(tensor)))

        soft_prediction = self._model.predict(tensor,
                                              verbose=0)

        visualize = False

        if visualize:
            indices = [0 * 4 + 1, 2 * 4 + 3] + [0 * 4 + 2, 1 * 4 + 3]
            Visualizer.visualize_tensor(tensor[indices], title=list(map(str, soft_prediction[indices, 1])), show=True)
            print('visualized')

        return soft_prediction

    def evaluate(self, images: list, standardise=True):
        print('Mean of evalutaion images is {}, std is {}'.format(
            self.__class__._mean_of_a_list(images),
            self.__class__._std_of_a_list(images)))
        x, y = next(self._generate_regular_shreds_stratified(images, self.width, self.height, self._t))
        print('True in evaluation dataset is {} * {}'.format(np.mean(y[1]), y.shape[0]))
        print('Mean before standardiztion is {}. Std is {}. Shape is {}'.format(np.mean(x), np.std(x), np.shape(x)))
        y_true = np.argmax(y, axis=-1)
        y_predicted = self.predict(x, standardise=standardise)
        assert y_true.shape == y_predicted.shape

        return np.mean(y_true == y_predicted)

    def load_weights(self, file_path=None):
        if file_path is None:
            file_path = self._get_model_checkpoint_file_path()

        self._model.load_weights(file_path)

        return self

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

    @staticmethod
    @abstractmethod
    def _build_model(width, height,
                     pre_activation_bn=False, post_activation_bn=False,
                     input_depth=2, classes=2, padding='same',
                     learning_rate=1e-3, learning_rate_decay=1e-6):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _generate_regular_shreds_stratified(images: list, width, height, t, batch_size=None):
        raise NotImplementedError

    @staticmethod
    def _mean_of_a_list(images_list):
        mean = \
            np.sum(list(map(np.sum, images_list))) / \
            np.sum(list(map(lambda image: np.prod(image.shape), images_list)))

        # _mean = np.mean(np.concatenate(list(map(np.ndarray.flatten, images_list))))

        return mean

    @staticmethod
    def _std_of_a_list(images_list):
        mean = GenericCNN._mean_of_a_list(images_list)
        std = np.sqrt(
            np.sum(list(map(lambda image: np.sum(np.power(image.astype(float) - mean, 2)), images_list))) /
            np.sum(list(map(lambda image: np.prod(image.shape).astype(float), images_list))))

        # _std = np.std(np.concatenate(list(map(np.ndarray.flatten, images_list))))

        return std

    def _fit_standardisation(self, images):
        # TODO: save it somewhere!!
        self._mean = self.__class__._mean_of_a_list(images)
        self._std = self.__class__._std_of_a_list(images)

        print('Mean is {}. Std is {}'.format(self._mean, self._std))

        assert not np.isclose(self._std, 0.0), 'Variance is too low'

        return self

    def _get_model_checkpoint_file_path(self):
        return 'saved_weights/{}-{}-{}-model.h5'.format(
            self.__class__.__name__,
            self._t,
            self._image_type.value
        )

