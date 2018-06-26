import platform
import os
import time
import glob
import sys

import numpy as np
from keras import Model, Input
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.layers import CuDNNLSTM, Concatenate, Dense, Activation, TimeDistributed, Add, LSTM, RNN, BatchNormalization
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu

from data_preparation import inverse_dictionary, encode_characters, decode_characters, SpecialConstants


class CharacterLevelReviewGenerator:
    def __init__(self, index_to_character, review_length:int):
        self._index_to_character = index_to_character
        self._character_to_index = inverse_dictionary(index_to_character)
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
        assert self._review_shape == train_y.shape[1:]
        assert self._review_shape == validation_y.shape[1:]

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
                ModelCheckpoint(model_file_path, monitor='val_categorical_accuracy', save_best_only=True)
            ]
        )

        return history

    def generate_greedy_string(self, seed: str, index_to_sentiment):
        seed = encode_characters(seed, self._character_to_index)

        for index in self._generate_greedy_numbers(seed, index_to_sentiment):
            yield self._index_to_character[index]

    def _generate_greedy_numbers(self,  seed, index_to_sentiment):
        result = np.zeros([1, ] + list(self._review_shape))
        seed = [SpecialConstants.START.value, self._character_to_index[' ']] + seed
        result[0, :len(seed)] = to_categorical(seed, num_classes=len(self._character_to_index))

        for index in range(1, len(seed)):
            yield np.argmax(result[0, index])

        for index in range(len(seed), result.shape[1]):
            self._model.reset_states()
            prediction = self._model.predict([result, np.array([index_to_sentiment[index]])])
            number_to_probability = prediction[0][index - 1]
            number = np.argmax(number_to_probability)
            result[0, index, number] = 1

            yield number

            if self._character_to_index['.'] == number:
                return

    def load_weights(self, file_path=None):
        if file_path is None:
            file_paths = glob.glob('weights/*/*.h5')
            file_path = max(file_paths, key=os.path.getctime)

        print('Restoring from {}'.format(file_path))
        self._model.load_weights(file_path)

    @property
    def _review_shape(self):
        return self._review_length, self._vocabulary_size

    @property
    def _sentiment_shape(self):
        return 1,

    @property
    def _vocabulary_size(self):
        return len(self._index_to_character)

    @classmethod
    def _build_model(
            cls,
            review_shape,
            sentiment_shape,
            vocabulary_size,
            use_pre_activation_batch_normalization=False,
            use_post_activation_batch_normalization=True,
    ):
        reviews_input = Input(shape=review_shape)
        reviews_output = reviews_input
        lstm = cls._get_lstm_class()

        def pre_activation_batch_normalization():
            return [BatchNormalization()] if use_pre_activation_batch_normalization else list()

        def post_activation_batch_normalization():
            return [BatchNormalization()] if use_post_activation_batch_normalization else list()

        for layer in [
            lstm(512, return_sequences=True),
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
                [lstm(512, return_sequences=True)] +
                [lstm(512, return_sequences=True)] +
                [lstm(512, return_sequences=True)] +
                [TimeDistributed(Dense(512))] +
                pre_activation_batch_normalization() +
                [Activation(relu)] +
                post_activation_batch_normalization() +
                [TimeDistributed(Dense(1024))] +
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
            loss=categorical_crossentropy,
            metrics=[categorical_accuracy]
        )

        model.summary()
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

    # TODO: is it what should be done?
    def evaluate(self, test_data):
        (test_reviews, test_sentiments), test_y = test_data

        predicted_y = self._model.predict([test_reviews, test_sentiments])
        test_reviews = self.convert_one_hot_to_string(test_reviews)
        predicted_y = self.convert_one_hot_to_string(predicted_y)
        blue = corpus_bleu(test_reviews, predicted_y)  # TODO: weigths?!

        return blue

    def convert_one_hot_to_string(self, one_hot):
        numbers = np.argmax(one_hot, axis=-1)
        strings = np.apply_along_axis(
            lambda indices: ''.join(map(self._index_to_character.get, indices[1:])),
            -1,
            numbers)

        return strings


def main():
    from data_preparation import prepare_data_characters

    if 'debug' in sys.argv:
        train_length = 20
        test_length = 20
        epochs = 3
    else:
        train_length = sys.maxsize
        test_length = sys.maxsize
        epochs = 40

    train_data, validation_data, index_to_character = prepare_data_characters(preview=10,
                                                                              train_length=train_length,
                                                                              test_length=test_length)
    model = CharacterLevelReviewGenerator(index_to_character, train_data[0][0].shape[1])

    if 'predict' in sys.argv:
        model.load_weights()

        for character in model.generate_greedy_string(
                "",
                ([0, ] * (train_data[0][0].shape[1] // 2)) +
                ([1, ] * (train_data[0][0].shape[1] - train_data[0][0].shape[1] // 2))
        ):
            print(character, end='', flush=True)
        print()

        for character in model.generate_greedy_string("", [0, ] * train_data[0][0].shape[1]):
            print(character, end='', flush=True)
        print()

        for character in model.generate_greedy_string("", [1, ] * train_data[0][0].shape[1]):
            print(character, end='', flush=True)
        print()
    elif 'evaluate' in sys.argv:
        model.load_weights()
        print('Bleu:', model.evaluate(validation_data))
    else:
        history = model.fit(train_data, validation_data, epochs=epochs)
        print(history)


if __name__ == '__main__':
    main()
