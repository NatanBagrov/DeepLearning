import glob
import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import Model, Input
from keras.activations import relu, softmax
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Activation, TimeDistributed, Add, LSTM, RNN, BatchNormalization, \
    Embedding
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.utils import plot_model

from review_generation import generate_negative_reviews, generate_negative_then_positive_reviews, \
    generate_positive_reviews
from word_data_preparation import inverse_dictionary, SpecialConstants, prepare_data_words


def plot_history(history, title, show=False):
    his = history.history
    x = list(range(len(history.history['val_loss'])))

    plt.figure()
    y_1 = his['val_loss']
    y_2 = his['loss']
    plt.plot(x, y_1)
    plt.plot(x, y_2)
    plt.title(title)
    plt.legend(['validation loss', 'training loss'], loc='upper right')
    os.makedirs('graphs', exist_ok=True)
    plt.savefig('graphs/{}-loss.png'.format(title))

    if show:
        plt.show()


class WordLevelReviewGenerator:
    def __init__(self, index_to_word, review_length: int):
        self._index_to_word = index_to_word
        self._word_to_index = inverse_dictionary(index_to_word)
        self._review_length = review_length
        self._model = self.__class__._build_model(
            self._review_shape,
            self._sentiment_shape,
            self._vocabulary_size,
            use_post_activation_batch_normalization=True,
        )

    def fit(self,
            train_data,
            validation_data,
            batch_size=128,
            epochs=10):
        (train_reviews, train_sentiments), train_y = train_data
        (validation_reviews, validation_sentiments), validation_y = validation_data

        assert self._review_shape == train_reviews.shape[1:]
        assert self._review_shape == validation_reviews.shape[1:]
        assert self._sentiment_shape == train_sentiments.shape[1:]
        assert self._sentiment_shape == validation_sentiments.shape[1:]

        time_stamp = time.strftime("%c")
        print('time_stamp=', time_stamp)
        model_directory_path = os.path.join('weights', time_stamp)
        os.makedirs(model_directory_path, exist_ok=True)
        model_file_path = os.path.join(model_directory_path, '{}.h5'.format(self.__class__.__name__))
        history = self._model.fit(
            x=[train_reviews, train_sentiments],
            y=train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=([validation_reviews, validation_sentiments], validation_y),
            callbacks=[
                TensorBoard(log_dir=os.path.join('logs', time_stamp)),
                ModelCheckpoint(model_file_path, save_best_only=True, verbose=1)
            ]
        )

        return history

    def generate_string(self, seed: str, index_to_sentiment, next_word_chooser):

        for index in self._generate_numbers(seed, index_to_sentiment, next_word_chooser):
            yield self._index_to_word[index] if self._index_to_word[index] != 'br' else '<line_break>'

    def _generate_numbers(self, seed: str, index_to_sentiment, next_word_chooser):
        result = np.zeros([1, ] + list(self._review_shape))
        seed = np.array([SpecialConstants.START.value] +
                        [self._word_to_index[w.lower()] for w in seed.split(' ')]) if len(seed) > 0 else []
        result[0, :len(seed)] = seed

        for index in range(1, len(seed)):
            yield result[0, index]

        for index in range(len(seed), result.shape[1]):
            self._model.reset_states()
            current_sentiment = np.array([index_to_sentiment[index]])
            prediction = self._model.predict([result, current_sentiment])
            next_word_softmax_probabilities = prediction[0][index - 1]
            number = next_word_chooser(next_word_softmax_probabilities)
            result[0, index] = number

            if SpecialConstants.PADDING.value == number:
                return

            yield number

    def load_weights(self, file_path=None):
        if file_path is None:
            file_paths = glob.glob('weights/*/*.h5')  # * means all if need specific format then *.csv
            file_path = max(file_paths, key=os.path.getctime)

        print('Restoring from {}'.format(file_path))
        self._model.load_weights(file_path)

    @property
    def _review_shape(self):
        return self._review_length,  # self._vocabulary_size

    @property
    def _sentiment_shape(self):
        return 1,

    @property
    def _vocabulary_size(self):
        return len(self._index_to_word)

    @classmethod
    def _build_model(
            cls,
            review_shape,
            sentiment_shape,
            vocabulary_size,
            use_pre_activation_batch_normalization=False,
            use_post_activation_batch_normalization=True,
            plot=False
    ):
        embedding_vector_length = 128
        max_review_length = 200
        reviews_input = Input(shape=review_shape)
        reviews_output = reviews_input
        lstm = cls._get_lstm_class()

        def pre_activation_batch_normalization():
            return [BatchNormalization()] if use_pre_activation_batch_normalization else list()

        def post_activation_batch_normalization():
            return [BatchNormalization()] if use_post_activation_batch_normalization else list()

        for layer in [
            Embedding(vocabulary_size, embedding_vector_length, input_length=max_review_length),
            lstm(512, return_sequences=True),  # TODO: does it have some non linearity automatically?
            lstm(512, return_sequences=True),
        ]:
            reviews_output = layer(reviews_output)

        sentiments_input = Input(shape=sentiment_shape)
        sentiments_output = sentiments_input

        for layer in [Dense(512),
                      Activation(relu)]:
            sentiments_output = layer(sentiments_output)

        # output = Concatenate()([reviews_output, sentiments_output])  # TODO: replicate then concatenate
        output = Add()([reviews_output, sentiments_output])

        for layer in (
                [lstm(512, return_sequences=True)] +  # TODO: does it have some non linearity automatically?
                [lstm(512, return_sequences=True)] +
                [TimeDistributed(Dense(512))] +
                pre_activation_batch_normalization() +
                [Activation(relu)] +
                post_activation_batch_normalization() +
                [TimeDistributed(Dense(vocabulary_size))] +
                [Activation(softmax)]
        ):
            output = layer(output)

        model = Model(
            inputs=[reviews_input, sentiments_input],
            outputs=[output]
        )

        model.compile(
            Adam(),
            loss=sparse_categorical_crossentropy,
            metrics=[categorical_accuracy]
        )

        model.summary()
        if plot:
            plot_directory_path = 'models/'
            os.makedirs(plot_directory_path, exist_ok=True)
            plot_file_path = os.path.join(plot_directory_path, '{}.png'.format(cls.__name__))

            if 'Darwin' == platform.system():
                plot_model(
                    model,
                    to_file=plot_file_path,
                    show_shapes=True,
                    show_layer_names=True
                )

        return model

    @staticmethod
    def _get_lstm_class() -> RNN.__class__:
        return LSTM
        # return CuDNNLSTM


def main():
    if 'debug' in sys.argv:
        train_length = 20
        test_length = 20
        epochs = 10
    elif 'relaxed' in sys.argv:
        train_length = 15000
        test_length = 15000
        epochs = 40
    else:
        train_length = sys.maxsize
        test_length = sys.maxsize
        epochs = 60

    train_data, validation_data, index_to_word = prepare_data_words(preview=10,
                                                                    train_length=train_length,
                                                                    test_length=test_length,
                                                                    top_words=8000)
    model = WordLevelReviewGenerator(index_to_word, train_data[0][0].shape[1])

    if 'predict' in sys.argv:
        print("Predicting...")
        model.load_weights()
        num_reviews = 10
        list_of_reviews = \
            generate_negative_then_positive_reviews(model, num_reviews, print_online=True) + \
            generate_negative_reviews(model, num_reviews, print_online=True) + \
            generate_positive_reviews(model, num_reviews, print_online=True)
        pass

    else:
        history = model.fit(train_data, validation_data, epochs=epochs)
        # plot_history(history, "20k top words")


if __name__ == '__main__':
    main()