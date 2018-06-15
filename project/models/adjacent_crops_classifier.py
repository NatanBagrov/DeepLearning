import os

import keras
import numpy as np
from keras import Sequential
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import regularizers, BatchNormalization, Flatten, Dense, Conv2D, MaxPool2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from utils.data_manipulations import list_of_images_to_numpy, resize_to
from utils.data_provider import DataProvider
from utils.learning_rate_schedulers import step_decay_scheduler_generator
from utils.shredder import Shredder
from utils.visualizer import PlotCallback, Visualizer, visualize_model_history


class AdjacentCropsClassifier:
    def __init__(self, data_provider: DataProvider = None, weights_file=None) -> None:
        super().__init__()
        assert data_provider is not None or weights_file is not None, "Must provide DataProvider -or- weights file"
        # assert data_provider is None or weights_file is None, \
        #     "Must provide DataProvider -or- weights file, not both"
        self._param_dict = {
            'epochs': 300,
            'input_shape': (64, 16),
            'batch_norm': True,
            'weight_decay': 5e-4,
            'dropout_rate': 0,
            'initial_learning_rate': 0.05,
            'scheduler': step_decay_scheduler_generator(initial_lr=0.05, coef=0.9, epoch_threshold=5),
            'batch_size': 256,
            'augmentation': True,
            'datagen': ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
        }
        self._input_shape = self._param_dict['input_shape']
        self._data_provider = data_provider
        self._model = None
        if weights_file is not None:
            print("Loading weights from %s" % weights_file)
            loaded_model = self._build_model(do_batch_norm=self._param_dict['batch_norm'],
                                             weight_decay=self._param_dict['weight_decay'],
                                             dropout_rate=self._param_dict['dropout_rate'],
                                             initial_learning_rate=self._param_dict['initial_learning_rate'])
            loaded_model.load_weights(weights_file)
            self._model = loaded_model

    def get_input_shape(self):
        return self._input_shape

    def get_model(self):
        assert self._model is not None, "Classifier should be fitted first"
        return self._model

    def _build_model(self, do_batch_norm, weight_decay, dropout_rate, initial_learning_rate):
        model = Sequential()
        model.add(
            Conv2D(
                16,
                (5, 5),
                padding='same',
                activation='relu',
                input_shape=(*self._input_shape, 1),
                kernel_regularizer=regularizers.l2(weight_decay)
            )
        )

        if do_batch_norm:
            model.add(BatchNormalization())
        model.add(
            Conv2D(
                16,
                (5, 5),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(weight_decay)

            )
        )
        if do_batch_norm:
            model.add(BatchNormalization())

        model.add(
            Conv2D(
                32,
                (3, 3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay)
            )
        )

        if do_batch_norm:
            model.add(BatchNormalization())
        model.add(
            Conv2D(
                32,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(weight_decay)

            )
        )
        if do_batch_norm:
            model.add(BatchNormalization())

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay)
            )
        )

        model.add(MaxPool2D())

        if do_batch_norm:
            model.add(BatchNormalization())

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay)
            )
        )
        model.add(Flatten())

        if do_batch_norm:
            model.add(BatchNormalization())

        model.add(Dropout(dropout_rate))

        model.add(Dense(self._input_shape[0] * 2, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

        if do_batch_norm:
            model.add(BatchNormalization())

        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay)))

        optimizer = keras.optimizers.SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
        # optimizer = keras.optimizers.adam(lr=initial_learning_rate)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        model.summary()
        self._model = model
        return model

    def fit(self, num_epochs=None, save_weight_dir=None):
        (x_train, y_train), (x_validation, y_validation) = self.load_data(options='train')
        num_epochs = num_epochs if num_epochs is not None else self._param_dict['epochs']

        model = self._build_model(do_batch_norm=self._param_dict['batch_norm'],
                                  weight_decay=self._param_dict['weight_decay'],
                                  dropout_rate=self._param_dict['dropout_rate'],
                                  initial_learning_rate=self._param_dict['initial_learning_rate'])

        learning_rate_scheduler = LearningRateScheduler(self._param_dict['scheduler'], verbose=0)

        batch_size = self._param_dict['batch_size']

        callbacks = [learning_rate_scheduler]

        if save_weight_dir is not None:
            model_name = 'adjacent_crops_clf_weights.{epoch:02d}-{val_acc:.3f}.h5'
            model_path = os.path.join(save_weight_dir, model_name)
            checkpoint_callback = ModelCheckpoint(filepath=model_path,
                                                  monitor='val_acc',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  mode='auto',
                                                  period=1)
            callbacks.append(checkpoint_callback)

        if self._param_dict['augmentation']:
            datagen = self._param_dict['datagen']
            training_history = model.fit_generator(datagen.flow(x_train, y_train,
                                                                batch_size=batch_size),
                                                   callbacks=[learning_rate_scheduler],
                                                   steps_per_epoch=x_train.shape[0] // batch_size,
                                                   epochs=num_epochs,
                                                   validation_data=(x_validation, y_validation))
        else:
            training_history = self._model.fit(x_train, y_train, epochs=num_epochs,
                                               callbacks=callbacks,
                                               validation_data=(x_validation, y_validation),
                                               batch_size=batch_size)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(x_validation, y_validation, verbose=0)
        print('Accuracy loss (after training finished):', test_loss)
        print('Accuracy accuracy (after training finished):', test_accuracy)
        return model, training_history

    def load_data(self, options='train'):
        num_samples = 1100
        orig_fish = self._data_provider.get_fish_images(grayscaled=True, num_samples=num_samples)
        t = 4
        k = round(self._input_shape[1] / 2)
        adjacent_crops, non_adjucent_crops = list(), list()

        for image in orig_fish:
            resize_shape = (64, 64)
            crops = resize_to(Shredder.shred(image, t), resize_shape)

            for i in range(t ** 2):  # We construct (t-1)*t adjacent crops
                if i % t == t - 1:
                    continue
                left_crop = crops[i]
                right_crop = crops[i + 1]
                adj = np.concatenate((left_crop[:, -k:], right_crop[:, :k]), axis=1)
                adjacent_crops.append(adj)

            arr = np.arange(t ** 2)
            for _ in range(t ** 2 - t):  # We construct (t-1)*t non adjacent crops
                left_idx = np.random.choice(arr)
                right_idx = np.random.choice(arr)
                left_crop = crops[left_idx]
                right_crop = crops[right_idx]
                if left_idx == right_idx - 1:
                    non_adj = np.concatenate((left_crop[:, -k:], right_crop[:, :k]), axis=1)
                else:
                    non_adj = np.concatenate((right_crop[:, -k:], left_crop[:, :k]), axis=1)
                non_adjucent_crops.append(non_adj)

        assert len(adjacent_crops) == len(non_adjucent_crops)
        images = np.expand_dims(list_of_images_to_numpy(adjacent_crops + non_adjucent_crops), -1) / 255
        labels = np.concatenate((np.ones(len(adjacent_crops)), np.zeros(len(non_adjucent_crops))), axis=0)

        x_train, x_valid, y_train, y_valid = \
            train_test_split(images, labels, train_size=0.8, random_state=42, stratify=labels)
        # x_valid, y_valid, x_test, y_test = \
        #     train_test_split(x_valid, y_valid, train_size=0.5, random_state=42, stratify=y_valid)
        if options == 'train':
            return (x_train, np.stack(y_train, axis=0)), (x_valid, np.stack(y_valid, axis=0))
        # if options == 'test':
        #     return x_test, np.stack(y_test, axis=0)
        # if options == 'both':
        #     return (x_train, np.stack(y_train, axis=0)), \
        #            (x_valid, np.stack(y_valid, axis=0)), \
        #            (x_test, np.stack(y_test, axis=0))

    def get_adjacent_probabilities(self, crop, others):
        k = round(self._input_shape[1] / 2)
        inputs = np.expand_dims(
            list_of_images_to_numpy([np.concatenate((crop[:, -k:], other[:, :k]), axis=1) for other in others]), -1)
        results = np.array(self._model.predict(inputs))
        return results
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        return softmax(results)


if __name__ == '__main__':
    # This will fit the classifier
    clf = AdjacentCropsClassifier(DataProvider())
    weights = os.path.join(os.path.dirname(__file__), 'saved_weights')
    os.makedirs(weights, exist_ok=True)
    model, history = clf.fit(num_epochs=100, save_weight_dir=weights)
    visualize_model_history(history, file_name_prefix=None, show=True)
