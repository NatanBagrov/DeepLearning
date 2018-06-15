import math
import os

import matplotlib.pyplot as plt
from keras.callbacks import Callback

from utils.shredder import Shredder


class Visualizer:
    @staticmethod
    def visualize_crops(crops_list, show=False, save_path=None):
        t = round(math.sqrt(len(crops_list)))
        grid_color = 255
        img = Shredder.reconstruct(crops_list)
        if t > 1:
            dx = img.shape[0] // t
            dy = img.shape[1] // t
            img[:, ::dy] = grid_color
            img[::dx, :] = grid_color
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        plt.clf()


class PlotCallback(Callback):
    def __init__(self, keys, file_path=None, show=False):
        super().__init__()
        self._file_path = file_path
        self._show = show
        self._i = None
        self._x = None
        self._keys = keys
        self._keys_to_values = None
        self._figure = None

    def on_train_begin(self, logs=None):
        self._i = 0
        self._x = []
        self._keys_to_values = {
            current_key: list()
            for current_key in self._keys
        }
        self._figure = plt.figure()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = dict()

        self._x.append(self._i)
        self._i += 1

        plt.figure(self._figure.number)
        plt.clf()

        for key, values in self._keys_to_values.items():
            values.append(logs.get(key))
            plt.plot(self._x, values, label=key)

        plt.legend()

        if self._file_path is not None:
            plt.savefig(self._file_path)

        if self._show:
            plt.show()


def visualize_model_history(history, file_name_prefix=None, show=False):
    save_dir = os.path.join(os.getcwd(), 'training_visualization')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if show:
        plt.show()
    if file_name_prefix is not None:
        file_name = file_name_prefix + 'accuracy.png'
        plt.savefig(os.path.join(save_dir, file_name))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if show:
        plt.show()
    if file_name_prefix is not None:
        file_name = file_name_prefix + 'loss.png'
        plt.savefig(os.path.join(save_dir, file_name))
