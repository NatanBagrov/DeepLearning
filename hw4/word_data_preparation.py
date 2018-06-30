import sys

import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.preprocessing import sequence

from data_preparation import SpecialConstants


def inverse_dictionary(dictionary: dict):
    return {value: key for key, value in dictionary.items()}


def _prepare_index_to_word(preview=0, top_words=5000):
    word_to_index = imdb.get_word_index()
    index_to_word = dict()
    for word, index in word_to_index.items():
        if top_words is not None and index > top_words:
            continue
        index_to_word[index + len(SpecialConstants)] = word

    index_to_word[SpecialConstants.PADDING.value] = SpecialConstants.PADDING
    index_to_word[SpecialConstants.START.value] = SpecialConstants.START
    index_to_word[SpecialConstants.OUT_OF_VOCABULARY.value] = SpecialConstants.OUT_OF_VOCABULARY

    assert top_words is None or len(index_to_word) == top_words + len(SpecialConstants)

    for index, word in list(index_to_word.items())[:preview]:
        print(index, ':', word)

    return index_to_word


def prepare_word_to_index_to_word(preview=0, top_words=5000):
    index_to_word = _prepare_index_to_word(preview, top_words)
    word_to_index = inverse_dictionary(index_to_word)
    return word_to_index, index_to_word


def prepare_data_common(preview=0, train_length=sys.maxsize, test_length=sys.maxsize, top_words=5000):
    (x, y), _ = imdb.load_data(
        start_char=SpecialConstants.START.value,
        oov_char=SpecialConstants.OUT_OF_VOCABULARY.value,
        num_words=top_words
    )

    reviews_train, reviews_test, sentiments_train, sentiments_test = train_test_split(x, y, test_size=0.1)

    reviews_train = reviews_train[:train_length]
    sentiments_train = sentiments_train[:train_length]
    reviews_test = reviews_test[:test_length]
    sentiments_test = sentiments_test[:test_length]

    word_to_index, index_to_word = prepare_word_to_index_to_word(preview, top_words)

    for index in range(preview):
        print(
            sentiments_train[index],
            reviews_train[index]
        )
        print(
            '+' if sentiments_train[index] else '-',
            ' '.join(str(index_to_word[i]) for i in reviews_train[index])
        )

    return (reviews_train, sentiments_train), (reviews_test, sentiments_test), word_to_index, index_to_word


def convert_to_x_y(reviews, length):
    x = sequence.pad_sequences(
        reviews,
        maxlen=length,
        padding='post',
        truncating='post',
        value=SpecialConstants.PADDING.value
    )

    y = np.roll(x, -1, axis=-1)
    y = np.expand_dims(y, -1)
    return x, y


def convert_to_column(vector):
    assert 1 == len(vector.shape)

    return np.reshape(vector, (-1, 1))


def prepare_data_words(preview=0, train_length=sys.maxsize, test_length=sys.maxsize, top_words=5000):
    (reviews_train, sentiments_train), (reviews_test, sentiments_test), word_to_index, index_to_word = \
        prepare_data_common(preview=preview, train_length=train_length, test_length=test_length, top_words=top_words)

    max_review_length = max(map(len, reviews_train))
    median_review_length = np.median(np.fromiter(map(len, reviews_train), dtype=np.int)).astype(np.int)
    mean_review_length = np.mean(np.fromiter(map(len, reviews_train), dtype=np.int))
    print('max review length: ', max_review_length)
    print('median review length: ', median_review_length)
    print('mean review length: ', mean_review_length)
    review_length = 200
    print('review length: ', review_length)

    train_x_review, train_y_review = convert_to_x_y(reviews_train, review_length)
    test_x_review, test_y_review = convert_to_x_y(reviews_test, review_length)

    # TODO: consider making sentiments per character
    sentiments_train = convert_to_column(sentiments_train)
    sentiments_test = convert_to_column(sentiments_test)

    print('Train:\n'
          '\t x shape: {}\n'
          '\t y shape: {}\n'
          '\t sentiment shape: {}\n'
          'Test:\n'
          '\t x shape: {}\n'
          '\t y shape: {}\n'
          '\t sentiment shape: {}\n'.format(
        train_x_review.shape,
        train_y_review.shape,
        sentiments_train.shape,
        test_x_review.shape,
        test_y_review.shape,
        sentiments_test.shape
    ))

    return ((train_x_review, sentiments_train), train_y_review), \
           ((test_x_review, sentiments_test), test_y_review), \
           index_to_word
