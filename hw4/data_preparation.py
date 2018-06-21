import string
import sys
from enum import Enum

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical


class SpecialConstants(Enum):
    PADDING = 0
    START = 1
    OUT_OF_VOCABULARY = 2


def inverse_dictionary(dictionary: dict):
    return {value: key for key, value in dictionary.items()}


def prepare_index_to_word(preview=0):
    word_to_index = imdb.get_word_index()
    index_to_word = {index + 3: word for word, index in word_to_index.items()}

    index_to_word[SpecialConstants.PADDING.value] = SpecialConstants.PADDING
    index_to_word[SpecialConstants.START.value] = SpecialConstants.START
    index_to_word[SpecialConstants.OUT_OF_VOCABULARY.value] = SpecialConstants.OUT_OF_VOCABULARY

    assert len(index_to_word) == len(word_to_index) + len(SpecialConstants)

    for index, word in list(index_to_word.items())[:preview]:
        print(index, ':', word)

    return index_to_word


def prepare_character_to_index(preview=0):
    characters = string.printable
    character_to_index = {character: index + 3 for index, character in enumerate(characters)}
    character_to_index[SpecialConstants.PADDING.value] = SpecialConstants.PADDING
    character_to_index[SpecialConstants.START.value] = SpecialConstants.START
    character_to_index[SpecialConstants.OUT_OF_VOCABULARY.value] = SpecialConstants.OUT_OF_VOCABULARY
    index_to_character = {index: character for character, index in character_to_index.items()}

    if 0 != preview:
        print(' '.join(map(lambda character_index: '{}:{}'.format(character_index[0], character_index[1]),
                           list(character_to_index.items())[:preview])))

    return character_to_index, index_to_character


def prepare_data_common(preview=0, train_length=sys.maxsize, test_length=sys.maxsize):
    index_to_word = prepare_index_to_word(preview=preview)
    (reviews_train, sentiments_train), (reviews_test, sentiments_test) = imdb.load_data(
        start_char=SpecialConstants.START.value,
        oov_char=SpecialConstants.OUT_OF_VOCABULARY.value
    )

    reviews_train = reviews_train[:train_length]
    sentiments_train = sentiments_train[:train_length]
    reviews_test = reviews_test[:test_length]
    sentiments_test = sentiments_test[:test_length]

    for index in range(preview):
        print(
            sentiments_train[index],
            reviews_train[index]
        )
        print(
            '+' if sentiments_train[index] else '-',
            ' '.join(str(index_to_word[i]) for i in reviews_train[index])
        )

    return (reviews_train, sentiments_train), (reviews_test, sentiments_test), index_to_word


def encode_characters(word: str, character_to_index: dict):
    return [character_to_index[character] for character in word if character in character_to_index]


def decode_characters(characters, index_to_character):
    return ' '.join([index_to_character[index] for index in characters if index in index_to_character])


def decode_join_tokenize(review, index_to_word: dict, character_to_index: dict, separator=' '):
    tokens = list()

    for i, index in enumerate(review):
        if index_to_word[index] in SpecialConstants:
            if SpecialConstants.OUT_OF_VOCABULARY != index_to_word[index]:
                tokens.append(index)
        else:
            # TODO: there are actually some letters which are not "printable" (e.g. maléfique).
            # a) Ignore
            # b) OOV character
            # c) unique on train set and a) or b)
            tokens.extend(encode_characters(index_to_word[index], character_to_index))

        assert None not in tokens

        if i + 1 != len(review):
            tokens.extend(encode_characters(separator, character_to_index))

    # Since it is train set, there is no need
    # tokens.append(character_to_index['.'])

    return tokens


def convert_to_x_y(reviews, length, end_of_line_token, vocabulary_size):
    x = sequence.pad_sequences(
        reviews,
        maxlen=length,
        padding='pre',
        truncating='post',
        value=SpecialConstants.PADDING.value
    )

    y = np.roll(x, -1, axis=-1)
    y[:, -1] = end_of_line_token

    x = to_categorical(x, num_classes=vocabulary_size)
    y = to_categorical(y, num_classes=vocabulary_size)

    return x, y


def convert_to_column(vector):
    assert 1 == len(vector.shape)

    return np.reshape(vector, (-1, 1))


def prepare_data_characters(preview=0, train_length=sys.maxsize, test_length=sys.maxsize):
    character_to_index, index_to_character = prepare_character_to_index(preview=preview)
    (reviews_train, sentiments_train), (reviews_test, sentiments_test), index_to_word = \
        prepare_data_common(preview=preview, train_length=train_length, test_length=test_length)
    reviews_train = list(map(lambda current_review: decode_join_tokenize(current_review,
                                                                         index_to_word, character_to_index),
                            reviews_train))
    reviews_test = list(map(lambda current_review: decode_join_tokenize(current_review,
                                                                        index_to_word, character_to_index),
                            reviews_test))
    max_review_length = max(map(len, reviews_train))
    median_review_length = np.median(np.fromiter(map(len, reviews_train), dtype=np.int)).astype(np.int)
    mean_review_length = np.mean(np.fromiter(map(len, reviews_train), dtype=np.int))
    print('max review length: ', max_review_length)
    print('median review length: ', median_review_length)
    print('mean review length: ', mean_review_length)
    train_x_review, train_y_review = convert_to_x_y(reviews_train, median_review_length,
                                                    character_to_index['.'], len(character_to_index))
    test_x_review, test_y_review = convert_to_x_y(reviews_test, median_review_length,
                                                  character_to_index['.'], len(character_to_index))
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
           index_to_character
