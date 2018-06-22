import platform
import os

import numpy as np
from keras import Model, Input
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, Concatenate, Dense, Activation, TimeDistributed, Add, LSTM, RNN
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint

from data_preparation import inverse_dictionary, encode_characters, decode_characters


class CharacterLevelReviewGenerator:
    def __init__(self, index_to_character):
        self._model = None
        self._index_to_character = index_to_character
        self._character_to_index = inverse_dictionary(index_to_character)

    def fit(self,
            train_data,
            validation_data,
            batch_size=128,
            epochs=10):
        (train_reviews, train_sentiments), train_y = train_data
        (validation_reviews, validation_sentiments), validation_y = validation_data

        review_shape = train_reviews.shape[1:]
        sentiment_shape = train_sentiments.shape[1:]
        y_shape = train_y.shape[1:]

        assert review_shape == train_reviews.shape[1:]
        assert review_shape == validation_reviews.shape[1:]
        assert sentiment_shape == train_sentiments.shape[1:]
        assert sentiment_shape == validation_sentiments.shape[1:]
        assert y_shape == train_y.shape[1:]
        assert y_shape == validation_y.shape[1:]

        self._model = self.__class__._build_model(review_shape, sentiment_shape, len(self._index_to_character))
        model_directory_path = 'weights'
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
                TensorBoard(),
                ModelCheckpoint(model_file_path)
            ]
        )

        return history

    def generate_greedy_string(self, seed: str, index_to_sentiment):
        seed = encode_characters(seed, self._character_to_index)
        numbers = self.generate_greedy_numbers(seed)
        string = decode_characters(numbers, self._index_to_character)

        return string

    @classmethod
    def _build_model(
            cls,
            review_shape,
            sentiment_shape,
            vocabulary_size
    ):
        reviews_input = Input(shape=review_shape)
        reviews_output = reviews_input
        lstm = cls._get_lstm_class()

        for layer in [
            lstm(128, return_sequences=True),  # TODO: does it have some non linearity automatically?
            lstm(128, return_sequences=True),
        ]:
            reviews_output = layer(reviews_output)

        sentiments_input = Input(shape=sentiment_shape)
        sentiments_output = sentiments_input

        for layer in [Dense(128),
                      Activation(relu)]:
            sentiments_output = layer(sentiments_output)

        # output = Concatenate()([reviews_output, sentiments_output])  # TODO: replicate then concatenate
        output = Add()([reviews_output, sentiments_output])

        for layer in [
            lstm(256, return_sequences=True),  # TODO: does it have some non linearity automatically?
            lstm(256, return_sequences=True),
            TimeDistributed(Dense(128)),
            Activation(relu),
            TimeDistributed(Dense(128)),
            Activation(relu),
            TimeDistributed(Dense(vocabulary_size)),
            Activation(softmax),
        ]:
            output = layer(output)

        model = Model(
            inputs=[reviews_input, sentiments_input],
            outputs=[output]
        )

        model.compile(
            Adam(),
            loss=categorical_crossentropy,
            # metrics=
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


def main():
    from data_preparation import prepare_data_characters

    train_data, validation_data, index_to_character = prepare_data_characters(preview=10,
                                                                              # train_length=20, test_length=10
                                                                              )
    model = CharacterLevelReviewGenerator(index_to_character)
    history = model.fit(train_data, validation_data, epochs=40)

    print(history)


if __name__ == '__main__':
    main()
