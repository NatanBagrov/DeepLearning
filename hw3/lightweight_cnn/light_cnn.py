import os

import keras
import numpy as np
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def step_decay_scheduler_generator(initial_lr, coef, epoch_threshold):
    return lambda epoch: initial_lr * (coef ** (epoch // epoch_threshold))


def custom_step_decay(initial_lr):
    def scheduler(epoch):
        if epoch <= 50:
            return initial_lr
        if epoch <= 90:
            return initial_lr * 0.4
        if epoch <= 125:
            return initial_lr * 0.4 * 0.2
        if epoch <= 150:
            return initial_lr * 0.4 * 0.2 * 0.25
        if epoch <= 175:
            return initial_lr * 0.4 * 0.2 * 0.25 * 0.4
        if epoch <= 195:
            return initial_lr * 0.4 * 0.2 * 0.25 * 0.4 * 0.4
        return initial_lr * 0.4 * 0.2 * 0.25 * 0.4 * 0.4 * (1 - 195 / (epoch * 2))

    return scheduler


def build_model(do_batch_norm, dropout, weight_decay, initial_learning_rate):
    model = Sequential()
    model.add(
        Conv2D(
            16,
            (3, 3),
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3),
            kernel_regularizer=regularizers.l2(weight_decay)
        )
    )

    if do_batch_norm:
        model.add(BatchNormalization())

    model.add(
        Conv2D(
            16,
            (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)

        )
    )
    if do_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPool2D())

    model.add(
        Conv2D(
            16,
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
    model.add(MaxPool2D())

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
            64,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(weight_decay)
        )
    )
    model.add(MaxPool2D())
    model.add(Flatten())

    if do_batch_norm:
        model.add(BatchNormalization())

    model.add(Dropout(dropout))

    model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay)))

    sgd = keras.optimizers.Adam(lr=initial_learning_rate, decay=1e-4)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )

    model.summary()

    return model


def normalize_data(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def build_and_fit_model(param_dict, number_of_epochs, save_weights_only=False):
    num_classes = 10
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Data normalization
    x_train, x_test = normalize_data(x_train, x_test)

    # Model creation
    model = build_model(do_batch_norm=param_dict['batch_norm'],
                        dropout=param_dict['dropout'],
                        weight_decay=param_dict['weight_decay'],
                        initial_learning_rate=param_dict['initial_learning_rate'])

    learning_rate_scheduler = LearningRateScheduler(param_dict['scheduler'], verbose=1)

    datagen = param_dict['image_data_generator']
    datagen.fit(x_train)

    batch_size = param_dict['batch_size']

    model_name = 'light_cnn_weights.{epoch:02d}-{val_acc:.3f}.h5'
    model_path = os.path.join(save_dir, model_name)
    checkpoint_callback = ModelCheckpoint(filepath=model_path,
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)

    callbacks = [learning_rate_scheduler, checkpoint_callback]

    if param_dict['augmentation']:
        training_history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                               callbacks=callbacks,
                                               steps_per_epoch=x_train.shape[0] // batch_size,
                                               epochs=number_of_epochs,
                                               validation_data=(x_test, y_test),
                                               workers=4)
    else:
        training_history = model.fit(x_train, y_train, epochs=number_of_epochs,
                                     callbacks=callbacks,
                                     validation_data=(x_test, y_test),
                                     batch_size=batch_size)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss (after training finished):', test_loss)
    print('Test accuracy (after training finished):', test_accuracy)

    return training_history


params_0855 = {
    'batch_norm': True,
    'dropout': 0,
    'weight_decay': 5e-4,
    'initial_learning_rate': 0.001,
    'scheduler': step_decay_scheduler_generator(0.05, 0.2, 60),
    'batch_size': 32,
    'augmentation': True,
    'image_data_generator': ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
}


def load_model_and_predict(model_path, model_params=None):
    print("LOADING SAVED MODEL FROM: %s" % model_path)
    if model_params is None:
        loaded_model = keras.models.load_model(model_path)
        loaded_model.summary()
    else:
        loaded_model = build_model(model_params['batch_norm'],
                                   model_params['dropout'],
                                   model_params['weight_decay'],
                                   model_params['initial_learning_rate'])
        loaded_model.load_weights(model_path)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Data normalization
    x_train, x_test = normalize_data(x_train, x_test)

    test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=1)
    print('Loaded model - Test loss:', test_loss)
    print('Loaded model - Test accuracy:', test_accuracy)

    return loaded_model


def visualize_model_history(history, file_name_prefix=None, show=False):
    save_dir = os.path.join(os.getcwd(), 'training_visualization')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
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
    plt.legend(['train', 'test'], loc='upper left')
    if show:
        plt.show()
    if file_name_prefix is not None:
        file_name = file_name_prefix + 'loss.png'
        plt.savefig(os.path.join(save_dir, file_name))


if __name__ == '__main__':
    #  Uncomment this if you with to train the model
    # history = build_and_fit_model(params_0855, number_of_epochs=300, save_weights_only=True)
    # visualize_model_history(history=history, file_name_prefix='light_cnn_', show=True)
    model_path = os.path.join('saved_models', 'light_cnn_weights.189-0.857.h5')
    load_model_and_predict(model_path, params_0855)
