from enum import Enum
import hashlib
import sys
import os

import numpy as np
import tensorflow as tf
from keras import Sequential, Model
from keras.activations import relu, softmax
from keras.applications.vgg16 import VGG16
from keras.metrics import categorical_crossentropy
from keras.layers import BatchNormalization, Conv3D, MaxPool2D, Activation, Dense, Reshape, Flatten, Conv2D, Input, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LambdaCallback
import keras.backend as K
from sinkhorn_knopp.sinkhorn_knopp import SinkhornKnopp  # TODO: install it (add to dep)
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, lpSum, LpStatus, LpStatusOptimal  # TODO: install it(add to dep)


from utils.visualizer import PlotCallback
from utils.image_type import ImageType
from utils.data_provider import DataProvider
from utils.layers import ExpandDimension, RepeatLayer
from utils.metrics import keras_zero_one_loss, numpy_zero_one_loss
from utils.os_functions import maybe_make_directories


class DeepPermutationNetwork:
    def __init__(self, t, width: int, height: int, image_type: ImageType):
        self._t = t
        self._width = width
        self._height = height
        self._model, self._pre_trained, self._after_pre_trained = DeepPermutationNetwork._build_model(t, width, height)
        self._image_type = image_type

    @staticmethod
    def _build_pre_trained(inputs):
        inputs_prepared_for_vgg = list(map(lambda input: RepeatLayer(3, -1)(ExpandDimension()(input)), inputs))

        shared_vgg = VGG16()
        shared_vgg.layers.pop()
        print('shared vgg head')

        # for current_layer in shared_vgg.layers:
        #     current_layer.trainable = False

        shared_vgg = Sequential(shared_vgg.layers)

        shared_vgg.summary()
        vgg_outputs = list(map(shared_vgg, inputs_prepared_for_vgg))
        model = Model(inputs, vgg_outputs)
        model.summary()

        return model

    @staticmethod
    def _build_after_pre_trained(inputs):
        reduce_dimension_of_vgg = Dense(128, activation=relu)
        conc = concatenate(list(map(reduce_dimension_of_vgg, inputs)))
        fc1 = Dense(1024, activation=relu)(conc)
        fc2 = Dense(t**4, activation=relu)(fc1)
        #TODO: add sk instead
        sm = Activation(relu)(fc2)

        output = sm
        model = Model(inputs, [output, ])
        model.summary()
        optimizer = Adam(lr=1e-5)
        model.compile(optimizer,
                      loss=DeepPermutationNetwork._mse,
                      metrics=[])

        return model

    @staticmethod
    def _concatenate_models(t, width, height, *args):
        inputs = [Input((height, width)) for _ in range(t**2)]
        output = inputs

        for model in args:
            output = model(output)

        model = Model(inputs, [output,])

        model.summary()

        return model

    @staticmethod
    def _build_model(t, width, height):
        assert 224 == height and 224 == width

        inputs = [Input((height, width)) for _ in range(t**2)]
        pre_trained_model = DeepPermutationNetwork._build_pre_trained(inputs)
        inputs_ = [Input((4096,)) for _ in range(t ** 2)]
        after_pre_trained_model = DeepPermutationNetwork._build_after_pre_trained(inputs_)
        full_model = DeepPermutationNetwork._concatenate_models(t, width, height, pre_trained_model, after_pre_trained_model)

        return full_model, pre_trained_model, after_pre_trained_model

    def _apply_pretrained_model(self, x):
        file_name = 'pre_trained_prediction_{}.npz'.format(hashlib.sha1(np.ascontiguousarray(np.array(x))).hexdigest())
        maybe_make_directories('cache')
        file_path = os.path.join('cache', file_name)

        if os.path.isfile(file_path):
            print('Using ', file_path)
            y = np.load(file_path)
            y = [value for key, value in y.items()]
        else:
            y = self._pre_trained.predict(x)
            print('Saving ', file_path)
            np.savez(file_path, *y)

        return y

    def fit(self, train_x, train_y, validation_x, validation_y, batch_size=32, epochs=10, train_trail_only=True):
        number_of_samples = train_x.shape[0]

        train_x = self._convert_x_to_network_format(train_x)
        train_y = self._convert_y_to_network_format(train_y)
        validation_x = self._convert_x_to_network_format(validation_x)
        validation_y = self._convert_y_to_network_format(validation_y)
        maybe_make_directories('graphs')

        if train_trail_only:
            model = self._after_pre_trained
            train_x = self._apply_pretrained_model(train_x)
            validation_x = self._apply_pretrained_model(validation_x)
        else:
            model = self._model


        # history = self._after_pre_trained.fit(
        #     train_x,
        #     train_y,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     verbose=2,
        #     callbacks=[
        #         PlotCallback(['loss', 'val_loss'],
        #                      file_path='graphs/deep_permutation_network_{}_{}_loss.png'.format(
        #                          self._t,
        #                          self._image_type.value),
        #                      show=True),
        #         ModelCheckpoint(self._get_model_checkpoint_file_path(), save_best_only=True)
        #     ],
        #     validation_data=(validation_x, validation_y),
        # )

        history = model.fit_generator(
            DeepPermutationNetwork._generate_batch(train_x, train_y, batch_size),
            steps_per_epoch=(number_of_samples + batch_size - 1) // batch_size,
            epochs=epochs,
            verbose=2,
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
        print('Predicing')
        matrices = self._model.predict(x, verbose=1).reshape((-1, self._t ** 2, self._t ** 2))
        permutations = list()
        print('Optimizing matrices:', end='', flush=True)

        for index, m in enumerate(matrices):
            p = DeepPermutationNetwork._find_p_by_m(m)
            current_permutation = np.argmax(p, axis=1)
            assert np.all((p @ np.arange(self._t ** 2)) == current_permutation)
            permutations.append(current_permutation)
            print('.', end='', flush=True)
        print()

        return permutations

    def load_weight(self, after_pre_trained_weights_file_path: str):
        self._after_pre_trained.load_weights(after_pre_trained_weights_file_path)

    @staticmethod
    def _find_dsm_by_m(m):
        sk = SinkhornKnopp()
        dsm = sk.fit(m)
        rows_sum = np.sum(dsm, axis=0)
        columns_sum = np.sum(dsm, axis=1)

        if not np.allclose(rows_sum, 0.0):
            print('Warning: non one row sum: {}'.format(rows_sum[np.argmax(np.abs(rows_sum))]))

        if not np.allclose(columns_sum, 0.0):
            print('Warning: non one column sum: {}'.format(columns_sum[np.argmax(np.abs(columns_sum))]))

        return dsm

    @staticmethod
    def _find_p_by_dsm_using_lp(dsm):
        t_square, t_square = dsm.shape
        problem = LpProblem('find_permutation_matrix', sense=LpMinimize)
        # TODO: use matrix
        p_matrix = LpVariable.dicts('P', (list(range(t_square)), list(range(t_square))), 0, 1, LpInteger)
        d_matrix = LpVariable.dicts('D', (list(range(t_square)), list(range(t_square))))

        # TODO: this is not L2 norm but L1
        problem += lpSum(d_matrix[row][column] for column in range(t_square) for row in range(t_square))

        for i in range(t_square):
            problem += lpSum(p_matrix[i][column] for column in range(t_square)) == 1
            problem += lpSum(p_matrix[row][i] for row in range(t_square)) == 1

            for j in range(t_square):
                problem += p_matrix[i][j] - dsm[i][j] <= d_matrix[i][j]
                problem += dsm[i][j] - p_matrix[i][j] <= d_matrix[i][j]

        problem.writeLP('current-problem.lp')
        problem.solve()
        if LpStatusOptimal != problem.status:
            print('Warning: status is ', LpStatus[problem.status])

        p = np.array([[p_matrix[row][column].value() for column in range(t_square)] for row in range(t_square)])
        # print('Objective:{}.\nL2:{}\np:\n{}\ndsm:\n{}\n.'.format(problem.objective, np.linalg.norm(p - dsm), p, dsm))

        return p

    def _convert_x_to_network_format(self, x):
        number_of_samples, t_square, height, width = x.shape
        assert self._t ** 2 == t_square

        x = [x[:, i, :, :] for i in range(t_square)]

        return x

    def _convert_y_to_network_format(self, y):
        number_of_samples, t_square = y.shape
        assert self._t ** 2 == t_square

        y = np.vstack([to_categorical(current_permutation).reshape(t_square ** 2) for current_permutation in y])

        assert (number_of_samples, t_square ** 2) == y.shape

        return y

    @staticmethod
    def _mse(y_true, y_predicted):
        # assert y_true.shape == y_predicted.shape

        return K.mean(K.square(y_true - y_predicted))

    def _get_model_checkpoint_file_path(self):
        maybe_make_directories('saved_weights')

        return 'saved_weights/deep-permutation-network-best-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )

    def _get_model_final_file_path(self):
        return 'saved_weights/deep-permutation-network-final-{}-{}-model.h5'.format(
            self._t,
            self._image_type.value
        )

    @staticmethod
    def _find_p_by_m(m):
        # dsm = DeepPermutationNetwork._find_dsm_by_m(m)
        p = DeepPermutationNetwork._find_p_by_dsm_using_lp(m)

        return p

    @staticmethod
    def _generate_batch(train_x, train_y, batch_size):
        t_square = len(train_x)
        number_of_samples, features_number = train_x[0].shape
        assert (number_of_samples, t_square ** 2) == train_y.shape

        while True:
            permutation = np.random.permutation(number_of_samples)
            train_x = [train_x[shred_index][permutation] for shred_index in range(t_square)]
            train_y[permutation] = train_y[permutation]

            for batch_offset in range(0, number_of_samples, batch_size):
                permutation = np.random.permutation(t_square ** 2)
                batch_x = [train_x[shred_index][batch_offset:batch_offset + batch_size] for shred_index in range(t_square)]
                batch_y = train_y[batch_offset:batch_offset + batch_size, permutation]

                yield (batch_x, batch_y)


if "__main__" == __name__:
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
    batch_size = 128
    force = True

    for t in (
            2,
            4,
            5):
        for image_type in ImageType:
            print('t={}. image type is {}'.format(t, image_type.value))

            (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = \
                DataProvider().get_train_validation_test_sets_as_array_of_shreds_and_array_of_permutations(
                t,
                width, height,
                image_type,
                number_of_samples=number_of_samples
            )
            clf = DeepPermutationNetwork(t, width, height, image_type)

            if not force and os.path.isfile(clf._get_model_checkpoint_file_path()):
                clf.load_weight(clf._get_model_checkpoint_file_path())
            else:
                clf.fit(train_x, train_y, validation_x, validation_y,
                        batch_size=batch_size,
                        epochs=epochs)

            train_y_predicted = clf.predict(train_x)
            print('Train 0-1: {}'.format(numpy_zero_one_loss(train_y, train_y_predicted)))
            validation_y_predicted = clf.predict(validation_x)
            print('Validation 0-1: {}'.format(numpy_zero_one_loss(validation_y, validation_y_predicted)))
            test_y_predicted = clf.predict(test_x)
            print('Test 0-1: {}'.format(numpy_zero_one_loss(validation_y, validation_y_predicted)))
