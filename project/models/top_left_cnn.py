import sys
import copy

import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.activations import relu, softmax
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, Dense, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback

from utils.image_type import ImageType
from utils.visualizer import PlotCallback, Visualizer
from utils.data_manipulations import shred_and_resize_to
from utils.data_provider import DataProvider
from models.generic_cnn import GenericCNN


class TopLeftCNN(GenericCNN):
    def __init__(self, t, width, height, image_type: ImageType, mean=None, std=None):
        super().__init__(t, width, height, image_type, mean=mean, std=std)
        self._model = self.__class__._build_model(width, height,
                                                  post_activation_bn=True,
                                                  input_depth=t**2,
                                                  classes=t**2)

    @staticmethod
    def _build_model(width, height,
                     pre_activation_bn=False, post_activation_bn=False,
                     input_depth=None, classes=2, padding='same',
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
            convolution(32, input_shape=(height, width, input_depth)) +
            convolution(32) +
            max_pooling() +
            convolution(64) +
            convolution(64) +
            max_pooling() +
            convolution(128) +
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
            flatten() +
            fully_connected(1024) +
            fully_connected(1024) +
            output_layer()
        )

        model.summary()

        optimizer = Adam(
            lr=learning_rate,
            decay=learning_rate_decay
        )

        loss = categorical_crossentropy

        metrics = [
            categorical_accuracy
        ]

        model.compile(
            optimizer,
            loss,
            metrics=metrics,
        )

        return model

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
            # permutation = np.arange(number_of_samples)
            sample_permutation = np.random.permutation(number_of_samples)

            for batch_offset in range(0, number_of_samples, batch_size):
                x = list()
                y = list()

                for batch_index in range(min(batch_size, number_of_samples - batch_offset)):
                    sample_index = sample_permutation[batch_offset + batch_index]
                    # shred_permutation = np.arange(t ** 2)
                    shred_permutation = np.random.permutation(t ** 2)
                    x.append(np.moveaxis(sample_index_to_shred_index_to_image[sample_index][shred_permutation], 0, -1))
                    y.append(0 == shred_permutation)

                x = np.stack(x)
                y = np.stack(y)

                assert (min(batch_size, number_of_samples - batch_offset), width, height, t ** 2) == x.shape
                assert (min(batch_size, number_of_samples - batch_offset), t ** 2) == y.shape

                yield x, y


def main():
    if 'debug' in sys.argv:
        print('Debug')
        number_of_samples = 20
        epochs = 5
    else:
        print('Release')
        number_of_samples = sys.maxsize
        epochs = 50

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

    width = 224
    height = 224
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

            clf = TopLeftCNN(t, width, height, image_type)

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
