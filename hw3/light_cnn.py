import os

import keras
import numpy as np
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


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

    sgd = keras.optimizers.SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
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


def build_and_fit_model(param_dict, number_of_epochs):
    num_classes = 10

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

    # TODO: add checkpoint to save params when validation > 0.85 + threshold

    if param_dict['augmentation']:
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      callbacks=[learning_rate_scheduler],
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=number_of_epochs,
                                      validation_data=(x_test, y_test),
                                      workers=4)
    else:
        history = model.fit(x_train, y_train, epochs=number_of_epochs, callbacks=[learning_rate_scheduler],
                            validation_data=(x_test, y_test),
                            batch_size=batch_size)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    # Save model and weights
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model_{:4f}.h5'.format(test_accuracy)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


validation_0855_params = {
    'batch_norm': True,
    'dropout': 0,
    'weight_decay': 5e-4,
    'initial_learning_rate': 0.05,
    'scheduler': step_decay_scheduler_generator(0.05, 0.2, 60),
    'batch_size': 256,
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

if __name__ == '__main__':
    build_and_fit_model(validation_0855_params, 300)
