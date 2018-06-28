import os
import pickle
import sys

import numpy as np

from character_level_review_generator import CharacterLevelReviewGenerator
from data_preparation import prepare_data_characters
from review_generation import generate_negative_then_positive_reviews
from review_sentiment_classifier import ReviewSentimentClassifier
from word_data_preparation import prepare_data_words, prepare_data_common
from word_level_review_generator import WordLevelReviewGenerator


def _scale_review_generator(min, max):
    print("Scaler is set to [{},{}]".format(min, max))
    return lambda p: 1 if p > max else max - p / max - min


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

    def compare_models(self, num_reviews_per_threshold=5, review_length=70, pickle_path=None):
        word_sentiments, word_reviews = self._get_statistics(self._word_model,
                                                             num_reviews_per_threshold,
                                                             review_length,
                                                             pickle_path)
        # char_stats = self._get_statistics(self._char_model, num_reviews_per_threshold, review_length) TODO

        # TODO: plot some nice graphs...

    def _get_statistics(self, model, num_reviews_per_threshold, review_length, pickle_path=None):
        thresholds = [0, 0.25, 0.5, 0.75, 1]
        print(
            "Comparing models with {} reviews per sentiment neg-to-pos shift,"
            "in the thresholds of {} and length of {}..."
                .format(num_reviews_per_threshold, str(thresholds), review_length))

        scaler = identity_scaler
        model_classifier_sentiments = []
        model_generated_reviews = dict()  # keys are thresholds*100 (eg. 0, 25, 50, 75, 100) to avoid floats
        for threshold in thresholds:
            reviews = generate_negative_then_positive_reviews(model=model,
                                                              num_reviews=num_reviews_per_threshold,
                                                              review_len=review_length,
                                                              pos_ratio=threshold,
                                                              print_online=True)
            model_generated_reviews[round(threshold * 100)] = reviews
            average = np.average([scaler(self._sentiment_clf.get_probability(r)) for r in reviews])
            print('{} threshold produces {} probability of positive sentiment'.format(threshold, average))
            model_classifier_sentiments.append(average)

        if pickle_path is not None:
            os.makedirs(pickle_path, exist_ok=True)
            filepath = os.path.join(pickle_path, '{}_reviews.pkl'.format(model.__class__.__name__))
            print('Dumping reviews to ' + str(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(model_generated_reviews, f)

        return model_classifier_sentiments, model_generated_reviews


def _get_word_level_model():
    train_data, validation_data, index_to_word = prepare_data_words(preview=10,
                                                                    top_words=12000)
    word_model = WordLevelReviewGenerator(index_to_word, train_data[0][0].shape[1])
    word_model.load_weights(os.path.join('weights', word_model.__class__.__name__ + '.h5'))
    return word_model


def _get_char_level_model():
    return None  # TODO: implement
    train_data, validation_data, index_to_character = prepare_data_characters(preview=10)
    char_model = CharacterLevelReviewGenerator(index_to_character, train_data[0][0].shape[1])
    char_model.load_weights(
        os.path.join('weights', char_model.__class__.__name__ + '.h5'))  # TODO: note this, place file in correct folder

    return char_model


def _get_sentiment_classifier():
    _, _, word_to_index, index_to_word = \
        prepare_data_common(preview=5, train_length=sys.maxsize, test_length=sys.maxsize, top_words=12000)

    sentiment_clf = ReviewSentimentClassifier(word_to_index, index_to_word)
    sentiment_clf.load_weights(os.path.join('weights', sentiment_clf.__class__.__name__ + '.h5'))

    return sentiment_clf


if __name__ == '__main__':
    word_model = _get_word_level_model()
    char_model = _get_char_level_model()
    review_sentiment_classifier = _get_sentiment_classifier()

    ms = ModelSelector(char_model, word_model, review_sentiment_classifier)
    pickle_path = 'reviews'
    ms.compare_models(num_reviews_per_threshold=10, review_length=100, pickle_path=pickle_path)
