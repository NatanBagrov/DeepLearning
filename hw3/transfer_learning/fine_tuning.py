import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from transfer_learning.cifar100vgg import cifar100vgg


def fine_tuning(pretrained_vgg_model, debug=False):
    pretrained_layers = pretrained_vgg_model.layers
    fine_tuned = list()
    for idx, num_samples in enumerate([100, 1000, 10000]):
        model = keras.Sequential()
        for pretrained_layer in pretrained_layers[:-2]:  # Without last Dense and its Activation
            pretrained_layer.trainable = False
            model.add(pretrained_layer)

        # Adding the trainable layer and its activation
        model.add(Dense(10))
        model.add(Activation('softmax'))

        batch_sizes = 8, 32, 128
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, _, y_train, _ = train_test_split(x_train, y_train,
                                                  train_size=num_samples, random_state=42, stratify=y_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        batch_size = batch_sizes[idx]
        max_epochs = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        if debug:
            max_epochs = 3
            x_test = x_test[:20]
            y_test = y_test[:20]

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.summary()

        history = model.fit_generator(datagen.flow(x_train, y_train,
                                                   batch_size=batch_size),
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=max_epochs,
                                      validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)
        model.save_weights('cifar10vgg_finetuning_{}_trainset.h5'.format(num_samples))
        fine_tuned.append((model, history))

    return fine_tuned


if __name__ == '__main__':
    cifar_100_vgg = cifar100vgg(train=False)
    models_and_histories = fine_tuning(cifar_100_vgg.model, debug=True)