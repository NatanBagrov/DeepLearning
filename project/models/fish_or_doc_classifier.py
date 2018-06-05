import os
import random

import numpy as np

import keras
from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import regularizers, BatchNormalization, Flatten, Dense, Lambda
from sklearn.model_selection import train_test_split

from utils.data_manipulations import list_of_images_to_numpy, normalize_grayscale_0_1, resize_to
from utils.data_provider import DataProvider, LimitedDataProvider
from utils.learning_rate_schedulers import step_decay_scheduler_generator
from utils.shredder import Shredder


class FishOrDocClassifier:
    def __init__(self, data_provider: DataProvider = None, weights_file=None) -> None:
        super().__init__()
        assert data_provider is not None or weights_file is not None, "Must provide DataProvider -or- weights file"
        assert data_provider is None or weights_file is None, \
            "Must provide DataProvider -or- weights file, not both"
        self._param_dict = {
            'epochs': 300,
            'im_width': 100,
            'im_height': 100,
            'batch_norm': True,
            'weight_decay': 5e-4,
            'initial_learning_rate': 0.07,
            'scheduler': step_decay_scheduler_generator(initial_lr=0.07, coef=0.9, epoch_threshold=60),
            'batch_size': 256,
        }
        self._data_provider = data_provider
        self._model = None
        if weights_file is not None:
            print("Loading weights from %s" % weights_file)
            loaded_model = self._build_model(do_batch_norm=self._param_dict['batch_norm'],
                                             weight_decay=self._param_dict['weight_decay'],
                                             initial_learning_rate=self._param_dict['initial_learning_rate'])
            loaded_model.load_weights(weights_file)
            self._model = loaded_model

    def get_model(self):
        assert self._model is not None, "Classifier should be fitted first"
        return self._model

    def _build_model(self, do_batch_norm, weight_decay, initial_learning_rate):
        input_shape = (self._param_dict['im_width'], self._param_dict['im_height'])
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))

        if do_batch_norm:
            model.add(BatchNormalization())

        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        if do_batch_norm:
            model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay)))

        sgd = keras.optimizers.SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
        model.compile(
            loss='binary_crossentropy',
            optimizer=sgd,
            metrics=['accuracy']
        )

        model.summary()
        self._model = model
        return model

    def fit(self, save_weight_dir):
        (x_train, y_train), (x_test, y_test) = self._load_data()

        model = self._build_model(do_batch_norm=self._param_dict['batch_norm'],
                                  weight_decay=self._param_dict['weight_decay'],
                                  initial_learning_rate=self._param_dict['initial_learning_rate'])

        learning_rate_scheduler = LearningRateScheduler(self._param_dict['scheduler'], verbose=0)

        batch_size = self._param_dict['batch_size']
        callbacks = [learning_rate_scheduler]
        training_history = self._model.fit(x_train, y_train, epochs=self._param_dict['epochs'],
                                           callbacks=callbacks,
                                           validation_data=(x_test, y_test),
                                           batch_size=batch_size)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss (after training finished):', test_loss)
        print('Test accuracy (after training finished):', test_accuracy)
        if save_weight_dir is not None:
            weight_file_name = 'fish_or_doc_clf_weights_acc_{:.3f}.h5'.format(test_accuracy)
            weights_full_path = os.path.join(save_weight_dir, weight_file_name)
            print("Weights were saved to: " + weights_full_path)
            model.save_weights(weights_full_path)
        return model, training_history

    def _load_data(self):
        resize = (self._param_dict['im_width'], self._param_dict['im_height'])
        orig_fish = self._data_provider.get_fish_images(grayscaled=True, resize=resize)
        orig_docs = self._data_provider.get_docs_images(grayscaled=True, resize=resize)
        fish, docs = list(), list()
        for _ in range(3):  # get more data
            fish = fish + list(
                map(lambda im: Shredder.reconstruct(Shredder.shred(im, random.choice([1, 2, 4, 5]), shuffle_shreds=True)),
                    orig_fish))  # 1 to also use original images
            docs = list(
                map(lambda im: Shredder.reconstruct(Shredder.shred(im, random.choice([1, 2, 4, 5]), shuffle_shreds=True)),
                    orig_docs))
        images = fish + docs
        labels = np.concatenate((np.ones(len(fish)), np.zeros(len(docs)))).astype(int).tolist()
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images[:], labels[:] = zip(*combined)
        x_train, x_test, y_train, y_test = \
            train_test_split(images, labels, train_size=0.8, random_state=42, stratify=labels)
        x_train, x_test = normalize_grayscale_0_1(list_of_images_to_numpy(x_train)), \
                          normalize_grayscale_0_1(list_of_images_to_numpy(x_test))
        return (x_train, np.stack(y_train, axis=0)), (x_test, np.stack(y_test, axis=0))

    def is_fish(self, x):
        x = resize_to(x, self._param_dict['im_width'], self._param_dict['im_height'])
        x = list_of_images_to_numpy(x)
        x = normalize_grayscale_0_1(x)
        res = self._model.predict(np.array([x])) > 0.5
        return res

    def is_doc(self, image):
        return not self.is_fish(image)


if __name__ == '__main__':
    # This will fit the classifier
    clf = FishOrDocClassifier(DataProvider())
    weights = os.path.join(os.path.dirname(__file__), 'saved_weights')
    os.makedirs(weights, exist_ok=True)
    clf.fit(weights)
