from enum import Enum

import numpy as np
from keras import Sequential
from keras.activations import relu
from keras.metrics import categorical_crossentropy
from keras.layers import BatchNormalization, Conv3D, MaxPool2D, Activation, Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sinkhorn_knopp.sinkhorn_knopp import SinkhornKnopp  # TODO: install it (add to dep)
from pulp import LpProblem, LpVariable, LpMinimize, GLPK  # TODO: install it(add to dep)


from utils.visualizer import PlotCallback
from utils.image_type import ImageType
from utils.data_provider import DataProvider
from utils.data_manipulations import shred_and_resize_to


class DeepPermutationNetwork:
    def __init__(self, t, width: int, height: int, image_type: ImageType):
        self._t = t
        self._width = width
        self._height = height
        self._model = DeepPermutationNetwork._build_model(t, width, height)
        self._image_type = image_type

    @staticmethod
    def _build_model(t: int,
                     width: int, height: int,
                     learning_rate: float = 0.001,
                     use_pre_activation_bn: bool = False,
                     use_post_activation_bn: bool = False):
        model = Sequential()
        shape = (height, width, t**2)
        kernel_depth = 1
        stride_depth = 1

        # I am trying to apply VGG-like architecture on each layer separately!
        # According to doc channel is first parameter in kernel, I hope and pray it is the last in strides
        # TODO: CHECK!!!
        for index, (layer, output) in enumerate((
                (Conv3D, 2),
                (Conv3D, 4),
                (MaxPool2D, None),
                (Conv3D, 8),
                (Conv3D, 16),
                (MaxPool2D, None),
        )):
            if 0 == index:
                input_shape = shape
            else:
                input_shape = None

            if Conv3D == layer:
                filters = output

                model.add(layer(
                    filters,
                    (kernel_depth, 3, 3),
                    strides=(2, 2, stride_depth),
                    padding='same',
                    input_shape=input_shape,
                ))
                model.add(Activation(relu))
                shape = ((shape[0] + 1) // 1, (shape[1] + 1) // 1, kernel_depth * (t ** 2))

                stride_depth = output
                kernel_depth = output
            elif MaxPool2D == layer:
                model.add(layer(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding='same',
                    input_shape=input_shape,
                ))

                shape = ((shape[0] + 1) // 1, (shape[1] + 1) // 1, shape[2])
            elif Dense == layer:
                assert False, 'Please implement channel wise dense (with conv?)'

            assert model.layers[-1].output_shape == shape

        # Calculate "DSM"
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation(relu))
        model.add(Dense(t**4))
        # TODO: add Sinkhorn units

        optimizer = Adam(lr=learning_rate)

        model.compile(
            optimizer,
            DeepPermutationNetwork._mse,
            metrics=[]
        )

        model.summary()

        return model

    def fit(self, train_x, train_y, validation_x, validation_y, batch_size=32, epochs=10):
        train_x = self._convert_x_to_network_format(train_x)
        train_y = self._convert_y_to_network_format(train_y)
        validation_x = self._convert_x_to_network_format(validation_x)
        validation_y = self._convert_y_to_network_format(validation_y)

        history = self._model.fit(
            train_x,
            train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[
                PlotCallback(['loss', 'val_loss'],
                             file_path='graphs/deep_permutation_network_{}_{}_loss.png'.format(
                                 self._t,
                                 self._image_type.value),
                             show=True),
                ModelCheckpoint(self._get_model_checkpoint_file_path(), save_best_only=True)
            ],
            validation_data=(validation_x, validation_y),
        )

        return history

    def predict(self, x):
        x = self._convert_x_to_network_format(x)
        matrices = self._model.predict(x).reshape((-1, self._t, self._t))
        permutations = list()

        for m in matrices:
            dsm = DeepPermutationNetwork._find_dsm_by_m(m)
            p = DeepPermutationNetwork._find_p_by_dsm(dsm)
            current_permutation = np.argmax(p, axis=1)
            assert np.all((current_permutation @ np.arange(self._t ** 2)) == p)
            permutations.append(current_permutation)

        return permutations

    @staticmethod
    def _find_dsm_by_m(m):
        sk = SinkhornKnopp()
        dsm = sk.fit(m)
        rows_sum = np.sum(dsm, axis=1)
        columns_sum = np.sum(dsm, axis=2)

        if not np.allclose(rows_sum, 0.0):
            print('Warning: non zero row sum: {}'.format(rows_sum[np.argmax(np.abs(rows_sum))]))

        if not np.allclose(columns_sum, 0.0):
            print('Warning: non zero column sum: {}'.format(columns_sum[np.argmax(np.abs(columns_sum))]))

        return dsm

    @staticmethod
    def _find_p_by_dsm(dsm):
        t_square, t_square = dsm.shape
        problem = LpProblem('find_permutation_matrix', sense=LpMinimize)
        p_matrix = [[LpVariable('p_{}_{}'.format(row, column), 0, 1)
                     for column in range(t_square)]
                    for row in range(t_square)]
        d_matrix = [[LpVariable('d_{}_{}'.format(row, column), 0, 1)
                     for column in range(t_square)]
                    for row in range(t_square)]

        # TODO: this is not L2 norm but L1
        problem += sum(d_matrix[row][column] for column in range(t_square) for row in range(t_square))

        for i in range(t_square):
            problem += sum(p_matrix[i][column] for column in range(t_square)) == 1
            problem += sum(p_matrix[row][i] for row in range(t_square)) == 1

            for j in range(t_square):
                problem += p_matrix[i][j] - dsm[i][j] <= d_matrix[i][j]
                problem += dsm[i][j] - p_matrix[i][j] <= d_matrix[i][j]

        GLPK().solve(problem)

        p = np.array([[p_matrix[row][column].value() for column in range(t_square)] for row in range(t_square)])
        print('Objective:{}.\nL2:{}\np:\n{}\ndsm:\n{}\n.'.format(problem.objective, np.linalg.norm(p, dsm), p, dsm))

        return p

    def _convert_x_to_network_format(self, x):
        number_of_samples, t_square, height, width = x.shape
        assert self._t ** 2 == t_square

        x = np.moveaxis(x, 1, -1)

        assert (number_of_samples, height, width, t_square) == x.shape

        return x

    def _convert_y_to_network_format(self, y):
        number_of_samples, t_square = y.shape
        assert self._t ** 2 == t_square

        y = np.vstack([to_categorical(current_permutation).reshape(t_square ** 2) for current_permutation in y])

        assert (number_of_samples, t_square ** t_square) == y.shape

        return y

    @staticmethod
    def _mse(y_true, y_predicted):
        assert y_true.shape == y_predicted.shape

        return K.mean(K.square(y_true - y_predicted))

    def _get_model_checkpoint_file_path(self):
        return 'saved_weights/deep-permutation-network-best-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )

    def _get_model_final_file_path(self):
        return 'saved_weights/deep-permutation-network-final-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )


if "__main__" == __name__:
    number_of_samples = 20
    width=77
    height=97

    for t in (2, 4, 5):
        for image_type in ImageType:
            (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = \
                DataProvider().get_train_validation_test_sets_as_array_of_shreds_and_array_of_permutations(
                t,
                width, height,
                image_type,
                number_of_samples=number_of_samples
            )
