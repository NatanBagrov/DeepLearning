import glob
import os
import sys
import time

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM, Dense, Conv1D, BatchNormalization, MaxPool1D
from keras.preprocessing import sequence
from keras.regularizers import l2

from data_preparation import SpecialConstants
from word_data_preparation import prepare_data_common


class ReviewSentimentClassifier:

    def __init__(self, word_to_index, index_to_word):
        super().__init__()
        self._index_to_word = index_to_word
        self._word_to_index = word_to_index
        self._build_model()

    def _build_model(self):
        self._max_review_length = 200
        embedding_vector_length = 16
        model = Sequential()
        model.add(Embedding(len(self._index_to_word), embedding_vector_length, input_length=self._max_review_length))
        model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.05, unroll=True))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-4)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self._model = model

    def fit(self, train, validation, epochs=10):
        reviews_train = sequence.pad_sequences(train[0], maxlen=self._max_review_length, truncating='post')
        reviews_validation = sequence.pad_sequences(validation[0], maxlen=self._max_review_length, truncating='post')
        time_stamp = time.strftime("%c")
        print('time_stamp=', time_stamp)
        model_directory_path = os.path.join('weights', time_stamp)
        os.makedirs(model_directory_path, exist_ok=True)
        model_file_path = os.path.join(model_directory_path, '{}.h5'.format(self.__class__.__name__))
        history = self._model.fit(reviews_train, train[1], validation_data=(reviews_validation, validation[1]),
                                  epochs=epochs,
                                  verbose=1,
                                  batch_size=128,
                                  callbacks=[ModelCheckpoint(model_file_path, save_best_only=True, verbose=1)])

    def load_weights(self, file_path=None):
        if file_path is None:
            file_path = os.path.join('weights', 'ReviewSentimentClassifier.h5')
        print('Restoring from {}'.format(file_path))
        self._model.load_weights(file_path)

    def _tokenize_review(self, review):
        list_of_words = review.split(' ') if type(review) == str else review
        tokens = [SpecialConstants.START.value] + [
            self._word_to_index[w] if w in self._word_to_index.keys() else SpecialConstants.OUT_OF_VOCABULARY.value
            for w in list_of_words]
        return tokens

    def get_probability(self, review):
        tokens = self._tokenize_review(review)
        seq_in = sequence.pad_sequences([tokens], maxlen=self._max_review_length, truncating='post')
        return self._model.predict_on_batch(seq_in)[0]


if __name__ == '__main__':
    (reviews_train, sentiments_train), (reviews_test, sentiments_test), word_to_index, index_to_word = \
        prepare_data_common(preview=5, train_length=sys.maxsize, test_length=sys.maxsize, top_words=12000)

    model = ReviewSentimentClassifier(word_to_index, index_to_word)
    if 'fit' in sys.argv:
        model.fit((reviews_train, sentiments_train), (reviews_test, sentiments_test), epochs=10)

    if 'predict' in sys.argv:
        model.load_weights(os.path.join('weights', 'ReviewSentimentClassifier.h5'))
        reviews = ['the think the movie is great it was very funny',
                   'i hate this bad movie',
                   'i enjoyed this movie so much']
        for review in reviews:
            probability = model.get_probability(review)
            print("[{}] THE REVIEW: {} IS {}".format(probability, review,
                                                     'POSITIVE' if probability > 0.5 else 'NEGATIVE'))
            print()
