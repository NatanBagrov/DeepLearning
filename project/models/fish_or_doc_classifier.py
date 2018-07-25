import os
import sys

import keras
import numpy as np
from keras import Sequential
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import regularizers, BatchNormalization, Flatten, Dense
from sklearn.model_selection import train_test_split

from utils.data_manipulations import resize_to, shred_shuffle_and_reconstruct
from utils.data_provider import DataProvider
from utils.learning_rate_schedulers import step_decay_scheduler_generator
from utils.pickle_helper import PickleHelper
from utils.visualizer import HistoryVisualizer

visualization_root = os.path.join(os.path.dirname(__file__), 'training_visualization', 'fish_or_doc')


class FishOrDocClassifier:
    def __init__(self, data_provider: DataProvider = None, weights_file=None) -> None:
        super().__init__()
        assert data_provider is not None or weights_file is not None, "Must provide DataProvider -or- weights file"
        assert data_provider is None or weights_file is None, \
            "Must provide DataProvider -or- weights file, not both"
        self._param_dict = {
            'epochs': 100,
            'input_shape': (220, 220),
            'batch_norm': True,
            'weight_decay': 5e-3,
            'initial_learning_rate': 0.07,
            'scheduler': step_decay_scheduler_generator(initial_lr=0.07, coef=0.9, epoch_threshold=60),
            'batch_size': 512,
        }
        self._input_shape = self._param_dict['input_shape']
        self._data_provider = data_provider
        self._model = None
        if weights_file is not None:
            print("Loading weights from %s" % weights_file)
            loaded_model = self._build_model(do_batch_norm=self._param_dict['batch_norm'],
                                             weight_decay=self._param_dict['weight_decay'],
                                             initial_learning_rate=self._param_dict['initial_learning_rate'])
            loaded_model.load_weights(weights_file)
            self._model = loaded_model

    def get_input_shape(self):
        return self._input_shape

    def get_model(self):
        assert self._model is not None, "Classifier should be fitted first"
        return self._model

    def _build_model(self, do_batch_norm, weight_decay, initial_learning_rate):
        model = Sequential()
        model.add(Flatten(input_shape=self._input_shape))

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

    def fit(self, save_weight_dir, epochs=None):
        (x_train, y_train), (x_test, y_test) = self._load_data()

        model = self._build_model(do_batch_norm=self._param_dict['batch_norm'],
                                  weight_decay=self._param_dict['weight_decay'],
                                  initial_learning_rate=self._param_dict['initial_learning_rate'])

        learning_rate_scheduler = LearningRateScheduler(self._param_dict['scheduler'], verbose=0)
        model_name = 'fish_or_doc.{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.h5'
        model_path = os.path.join(save_weight_dir, model_name)
        checkpoint_callback_loss = ModelCheckpoint(filepath=model_path,
                                                   monitor='val_loss',
                                                   verbose=2,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)
        checkpoint_callback_acc = ModelCheckpoint(filepath=model_path,
                                                  monitor='val_acc',
                                                  verbose=2,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='auto',
                                                  period=1)

        batch_size = self._param_dict['batch_size']
        callbacks = [learning_rate_scheduler, checkpoint_callback_loss, checkpoint_callback_acc]
        training_history = self._model.fit(x_train, y_train,
                                           epochs=self._param_dict['epochs'] if epochs is None else epochs,
                                           callbacks=callbacks,
                                           validation_data=(x_test, y_test),
                                           batch_size=batch_size)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss (after training finished):', test_loss)
        print('Test accuracy (after training finished):', test_accuracy)
        return model, training_history

    def _load_data(self):
        num_samples = 2000
        orig_fish = self._data_provider.get_fish_images(grayscaled=True, num_samples=num_samples,
                                                        resize=self._input_shape)
        orig_docs = self._data_provider.get_docs_images(grayscaled=True, num_samples=num_samples,
                                                        resize=self._input_shape)
        ts = (1, 2, 4, 5)
        fish = np.concatenate([shred_shuffle_and_reconstruct(orig_fish, t) for t in ts], axis=0)
        docs = np.concatenate([shred_shuffle_and_reconstruct(orig_docs, t) for t in ts], axis=0)

        images = np.concatenate((fish, docs), axis=0) / 255
        labels = np.concatenate((np.ones(len(fish)), np.zeros(len(docs)))).astype(int).tolist()
        x_train, x_test, y_train, y_test = \
            train_test_split(images, labels, train_size=0.8, random_state=42, stratify=labels)
        return (x_train, np.stack(y_train, axis=0)), (x_test, np.stack(y_test, axis=0))

    def is_fish(self, x):
        x = resize_to(x, self._input_shape)
        x = x / 255
        res = self._model.predict(x) > 0.5
        return res

    def is_doc(self, image):
        return not self.is_fish(image)


if __name__ == '__main__':
    weights = os.path.join(os.path.dirname(__file__), 'saved_weights')
    epochs = 3 if 'debug' in sys.argv else 300
    if 'fit' in sys.argv:
        # This will fit the classifier
        clf = FishOrDocClassifier(DataProvider())
        print('Fitting {}'.format(clf.__class__.__name__))
        os.makedirs(weights, exist_ok=True)
        model, history = clf.fit(weights, epochs=300)
        PickleHelper.dump(history.history, os.path.join(visualization_root, 'history.pkl'))
    if 'visualize' in sys.argv:
        print('Visualizing history')
        history = PickleHelper.load(os.path.join(visualization_root, 'history.pkl'))
        HistoryVisualizer.visualize(history['acc'], history['val_acc'], 'accuracy', visualization_root)
        HistoryVisualizer.visualize(history['loss'], history['val_loss'], 'loss', visualization_root)
