import os
import sys

import numpy as np

from character_level_review_generator import CharacterLevelReviewGenerator
from data_preparation import prepare_data_characters
from review_generation import generate_negative_then_positive_reviews
from review_sentiment_classifier import ReviewSentimentClassifier
from word_data_preparation import prepare_data_words, prepare_data_common
from word_level_review_generator import WordLevelReviewGenerator


def _scale_review_generator(min, max):
    return lambda p: max - p / max - min


def identity_scaler(p):
    return p


class ModelSelector:
    def __init__(self,
                 char_model: CharacterLevelReviewGenerator,
                 word_model: WordLevelReviewGenerator,
                 sentiment_classifier: ReviewSentimentClassifier):
        super().__init__()
        self._char_model = char_model
        self._word_model = word_model
        self._sentiment_clf = sentiment_classifier

    def compare_models(self):
        pos1 = ["i love this movie", "this movie is the best movie ever"]
        neg1 = ["this movie is the worst movie i have ever seen", "the movie was very boring"]
        pos2 = ["i really enjoyed the movie", "the movie was great and very funny"]
        neg2 = ["i hate this movie", "this movie was the worst movie ever"]
        max1 = np.average([self._sentiment_clf.get_probability(r) for r in pos1])
        min1 = np.average([self._sentiment_clf.get_probability(r) for r in neg1])
        max2 = np.average([self._sentiment_clf.get_probability(r) for r in pos2])
        min2 = np.average([self._sentiment_clf.get_probability(r) for r in neg2])
        range = np.arange(0.0, 1.01, 0.2)
        word_scaler = identity_scaler
        # word_scaler = _scale_review_generator(min1, max1)
        char_scaler = _scale_review_generator(min2, max2)
        word_model_classifier_sentiments = []
        for threshold in range:
            word_level_reviews = generate_negative_then_positive_reviews(model=self._word_model,
                                                                         num_reviews=5,
                                                                         pos_ratio=threshold,
                                                                         print_online=True)
            word_model_classifier_sentiments.append(
                np.average([word_scaler(self._sentiment_clf.get_probability(r)) for r in word_level_reviews]))
            print(word_model_classifier_sentiments)
            # char_level_reviews = generate_negative_then_positive_reviews(model=self._char_model,
            #                                                              num_reviews=10,
            #                                                              relative_part_pos=threshold)
        pass


if __name__ == '__main__':
    # Lots of mess, but works.
    train_data, validation_data, index_to_word = prepare_data_words(preview=10,
                                                                    top_words=12000)
    word_model = WordLevelReviewGenerator(index_to_word, train_data[0][0].shape[1])
    word_model.load_weights(os.path.join('weights', word_model.__class__.__name__ + '.h5'))

    # train_data, validation_data, index_to_character = prepare_data_characters(preview=10)
    # char_model = CharacterLevelReviewGenerator(index_to_character, train_data[0][0].shape[1])
    # word_model.load_weights(os.path.join('weights', char_model.__class__.__name__ + '.h5'))  # TODO: note this, place file

    (reviews_train, sentiments_train), (reviews_test, sentiments_test), word_to_index, index_to_word = \
        prepare_data_common(preview=5, train_length=sys.maxsize, test_length=sys.maxsize, top_words=12000)

    sentiment_clf = ReviewSentimentClassifier(word_to_index, index_to_word)
    sentiment_clf.load_weights(os.path.join('weights', sentiment_clf.__class__.__name__ + '.h5'))

    ms = ModelSelector(None, word_model, sentiment_clf)
    ms.compare_models()
