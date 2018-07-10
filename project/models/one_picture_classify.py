from enum import Enum

from keras import Sequential
from keras.activations import relu, softmax
from keras.layers import Conv1D, Conv2D, MaxPool2D, Dense, Activation, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
import numpy as np

from utils.data_provider import DataProvider
from utils.visualizer import PlotCallback
from utils.data_manipulations import shred_and_resize_to
from utils.image_type import ImageType

class OnePictureClassify:
    def __init__(self, t, width, height,
                 image_type: ImageType,
                 data_provider: DataProvider=None, weights_file=None) -> None:
        self._t = t
        self._width = width
        self._height = height
        self._image_type = image_type
        self._data_provider = data_provider
        self._model = self._build_model(t, width, height)

    @staticmethod
    def _build_model(t,
                     width: int, height: int,
                     learning_rate: float = 0.001,
                     use_pre_activation_bn: bool = False,
                     use_post_activation_bn: bool = False):
        def get_pre_activation_bn_layer() -> list:
            return [BatchNormalization()] if use_pre_activation_bn else []

        def get_post_activation_bn_layer() -> list:
            return [BatchNormalization()] if use_post_activation_bn else []

        model = Sequential(
            [Conv2D(2, (3, 3), strides=(2, 2), input_shape=(height, width, 1))] +
            get_pre_activation_bn_layer() +
            [Activation(relu)] +
            get_post_activation_bn_layer() +
            [Conv2D(4, (3, 3), strides=(2, 2))] +
            get_pre_activation_bn_layer() +
            [Activation(relu)] +
            get_post_activation_bn_layer() +
            [MaxPool2D(pool_size=(2, 2), strides=(2, 2))] +
            [Conv2D(8, (3, 3), strides=(2, 2))] +
            get_pre_activation_bn_layer() +
            [Activation(relu)] +
            get_post_activation_bn_layer() +
            [Conv2D(16, (3, 3), strides=(2, 2))] +
            get_pre_activation_bn_layer() +
            [Activation(relu)] +
            get_post_activation_bn_layer() +
            [MaxPool2D(pool_size=(2, 2), strides=(2, 2))] +
            [Flatten()] +
            [Dense(32)] +
            get_pre_activation_bn_layer() +
            [Activation(relu)] +
            [Dense(t**2, activation=softmax)]
        )

        optimizer = Adam(lr=learning_rate)

        model.compile(
            optimizer,
            categorical_crossentropy,
            metrics=[categorical_accuracy, ]
        )

        model.summary()

        return model

    def fit(self, batch_size=32, epochs=10):
        (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = self._load_data()

        print('Train      x mean: {} std: {} shape: {}. y shape: {}.'
              'Validation x mean: {} std: {} shape: {}. y shape: {}.'
              'Test       x mean: {} std: {} shape: {}. y shape: {}.'.format(
            np.mean(train_x), np.std(train_y), train_x.shape, train_y.shape,
            np.mean(validation_x), np.std(validation_y), validation_x.shape, validation_y.shape,
            np.mean(test_x), np.std(test_y), test_x.shape, test_y.shape
        ))  # TODO: standardise

        self._model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                PlotCallba_get_model_checkpoint_file_pathck(['loss', 'val_loss'],
                             file_path='graphs/one_picture_classify_{}_{}_loss.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                PlotCallback(['categorical_accuracy', 'val_categorical_accuracy'],
                             file_path='graphs/one_picture_classify_{}_{}_acc.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                ModelCheckpoint(self._get_model_checkpoint_file_path(), save_best_only=True)
            ],
            validation_data=(validation_x, validation_y),
        )

        test_loss, test_accuracy = self._model.evaluate(test_x, test_y)
        print('Test accuracy {}.\nTrivial accuracy: {}.\nTest Loss {}'.format(test_accuracy, 1.0 / (self._t**2), test_loss))

        self._model.save(self._get_model_final_file_path())

    def predict(self, x):
        assert x.shape[1:] == (self._height, self._width, 1), 'Please resize/reshape pictures'

        y_softmax = self._model.predict(x)
        y = np.argmax(y_softmax, axis=1)

        return y

    def _load_data(self):
        np.random.seed(42)
        number_of_samples = 2000

        if ImageType.DOCUMENTS == self._image_type:
            get_images = self._data_provider.get_docs_images
        else:
            get_images = self._data_provider.get_fish_images

        images = get_images(
            num_samples=number_of_samples,
        )

        shreds = shred_and_resize_to(images, self._t, (self._height, self._width))
        assert shreds.shape == (len(images), self._t ** 2, self._height, self._width)

        train_validation_shreds, test_shreds = train_test_split(shreds, test_size=0.1)
        train_shreds, validation_shreds = train_test_split(train_validation_shreds, train_size=0.7)

        def single_shred_to_x_y(shred):
            assert shred.shape == (self._t ** 2, self._height, self._width)

            x = np.reshape(shred, (self._t ** 2, self._height, self._width, 1))
            y = np.arange(self._t ** 2)

            return x, y

        def shreds_to_x_y(shreds):
            xs = list()
            ys = list()

            for row in range(shreds.shape[0]):
                x, y = single_shred_to_x_y(shreds[row])
                xs.append(x)
                ys.append(y)

            return np.concatenate(xs), np.concatenate(ys)

        train_x, train_y = shreds_to_x_y(train_shreds)
        validation_x, validation_y = shreds_to_x_y(validation_shreds)
        test_x, test_y = shreds_to_x_y(test_shreds)

        assert train_x.shape == (self._t ** 2 * train_shreds.shape[0], self._height, self._width, 1)
        assert train_y.shape == (self._t ** 2 * train_shreds.shape[0],)
        assert validation_x.shape == (self._t ** 2 * validation_shreds.shape[0], self._height, self._width, 1)
        assert validation_y.shape == (self._t ** 2 * validation_shreds.shape[0],)
        assert test_x.shape == (self._t ** 2 * test_shreds.shape[0], self._height, self._width, 1)
        assert test_y.shape == (self._t ** 2 * test_shreds.shape[0],)

        return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)

    def _get_model_checkpoint_file_path(self):
        return 'saved_weights/one-picture-classify-best-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )

    def _get_model_final_file_path(self):
        return 'saved_weights/one-picture-classify-final-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )


if "__main__" == __name__:
    for t in (2, 4, 5):
        for image_type in ImageType:
            clf = OnePictureClassify(t, 220, 220, image_type, DataProvider())
            clf.fit(epochs=50)
