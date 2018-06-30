import os
import pickle
import sys
import itertools

import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from character_level_review_generator import CharacterLevelReviewGenerator
from data_preparation import prepare_data_characters, prepare_data_for_as_words_lists, SpecialConstants
from review_generation import generate_negative_then_positive_reviews
from review_sentiment_classifier import ReviewSentimentClassifier
from word_data_preparation import prepare_data_words, prepare_data_common
from word_level_review_generator import WordLevelReviewGenerator
from next_number_choosers import greedy_number_chooser, temperature_number_chooser_generator
from plotting import plt


def _scale_review_generator(min, max):
    print("Scaler is set to [{},{}]".format(min, max))
    return lambda p: 1 if p > max else max - p / max - min


def identity_scaler(p):
    return p


def _do_bleu(references, hypotheses, weight_combination):
    return corpus_bleu([references, ] * len(hypotheses), hypotheses, weights=weight_combination)


def calc_and_plot_correlation(x, y, texts: dict):
    print(np.corrcoef(x, y)[0, 1])
    plt.plot(x, y, marker='o')
    if texts is not None:
        plt.title(texts['title'])
        plt.xlabel(texts['x'])
        plt.ylabel(texts['y'])
    plt.show()
    plt.clf()


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

    @staticmethod
    def _generate_reviews(model, next_word_chooser, seeds, positiveness, review_length,
                          file_path_to_cache=None, force=False, preview=0):
        if file_path_to_cache is None or force or not os.path.isfile(file_path_to_cache):
            print('Calculating'.format(file_path_to_cache))
            positive_suffix_length = max(0, min(review_length, int(round(positiveness * review_length / 100.0))))
            negative_prefix_length = review_length - positive_suffix_length
            index_to_sentiment = [0,] * negative_prefix_length + [1,] * positive_suffix_length
            generated_reviews = list()

            # TODO: fix it better
            if isinstance(model, CharacterLevelReviewGenerator):
                separator = ''
            elif isinstance(model, WordLevelReviewGenerator):
                separator = ' '
            else:
                assert False

            for current_seed in seeds:
                current_review = separator.join(itertools.islice(
                    model.generate_string(current_seed, index_to_sentiment, next_word_chooser),
                    review_length))
                current_review = current_review.split(' ')

                # Patch for character level model.
                while '' in current_review:
                    current_review.remove('')

                generated_reviews.append(current_review)

            print('Dumping to {}'.format(file_path_to_cache))

            if file_path_to_cache is not None:
                with open(file_path_to_cache, 'wb') as file_handler_to_cache:
                    pickle.dump(generated_reviews, file_handler_to_cache)
        else:
            print('Restoring from {}'.format(file_path_to_cache))

            with open(file_path_to_cache, 'rb') as file_handler_to_cache:
                generated_reviews = pickle.load(file_handler_to_cache)

        for review in generated_reviews[:preview]:
            print(review)

        return generated_reviews

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

        model_generated_reviews['sentiments'] = model_classifier_sentiments

        if pickle_path is not None:
            os.makedirs(pickle_path, exist_ok=True)
            filepath = os.path.join(pickle_path, '{}_reviews.pkl'.format(model.__class__.__name__))
            print('Dumping reviews to ' + str(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(model_generated_reviews, f)

        return model_classifier_sentiments, model_generated_reviews

    @staticmethod
    def get_bleu_score(generated_reviews_dict, test_data, weight_combinations):
        (test_reviews, test_sentiments) = test_data
        positive_reviews = test_reviews[1 == test_sentiments]
        negative_reviews = test_reviews[0 == test_sentiments]
        positive_generated_reviews = generated_reviews_dict[100]
        negative_generated_reviews = generated_reviews_dict[0]
        overall_scores = dict()
        overall_scores['x'] = 'BLEU method'
        overall_scores['y'] = 'score'
        overall_scores['title'] = 'BLEU scores for 30 generated reviews'
        overall_scores['xticks'] = weight_combinations
        full_scores = list()
        for weights in weight_combinations:
            pos_ref_to_pos_gen_bleu = _do_bleu(positive_reviews, positive_generated_reviews, weights)
            pos_ref_to_neg_gen_bleu = _do_bleu(positive_reviews, negative_generated_reviews, weights)
            neg_ref_to_neg_gen_bleu = _do_bleu(negative_reviews, negative_generated_reviews, weights)
            neg_ref_to_pos_gen_bleu = _do_bleu(negative_reviews, positive_generated_reviews, weights)
            all_30_reviews = generated_reviews_dict[0] + generated_reviews_dict[100] + generated_reviews_dict[50]
            full_bleu = _do_bleu(test_reviews, all_30_reviews, weights)
            print('BLEU for {} score:'.format(weights))
            print('for 30 reviews on the test set is: {:.3f}'.format(full_bleu))
            print('comparing POSITIVE test reviews with: ', end='', flush=True)
            print('positive generated scores: {:.3f} ,'.format(pos_ref_to_pos_gen_bleu), end='', flush=True)
            print('and with negative generated scores: {:.3f}'.format(pos_ref_to_neg_gen_bleu), end='', flush=True)
            print()
            print('comparing NEGATIVE test reviews with: ', end='', flush=True)
            print('negative generated scores: {:.3f} ,'.format(neg_ref_to_neg_gen_bleu), end='', flush=True)
            print('and with positive generated scores: {:.3f}'.format(neg_ref_to_pos_gen_bleu), end='', flush=True)
            print()
            print()
            full_scores.append(full_bleu)
        calc_and_plot_correlation(np.arange(4) + 1, full_scores, overall_scores)


def _get_word_level_model():
    train_data, validation_data, index_to_word = prepare_data_words(preview=10,
                                                                    top_words=12000)
    word_model = WordLevelReviewGenerator(index_to_word, train_data[0][0].shape[1])
    word_model.load_weights(os.path.join('weights', word_model.__class__.__name__ + '.h5'))
    return word_model


def _get_char_level_model():
    train_data, validation_data, index_to_character = prepare_data_characters(preview=10,
                                                                              train_length=10,
                                                                              test_length=10)
    review_length = 923
    char_model = CharacterLevelReviewGenerator(index_to_character, review_length)
    char_model.load_weights(
        os.path.join('weights', char_model.__class__.__name__ + '.h5'))

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

    _, test_data = prepare_data_for_as_words_lists(train_length=0, test_length=10)

    if 'bleu_dict_word' in sys.argv:
        with open(os.path.join('reviews', 'WordLevelReviewGenerator_reviews.pkl'), 'rb') as file_handler_to_cache:
            generated_reviews = pickle.load(file_handler_to_cache)
        weight_combinations = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.333, 0.333, 0.333, 0), (0.25, 0.25, 0.25, 0.25)]
        ModelSelector.get_bleu_score(generated_reviews, test_data, weight_combinations)
        exit(0)

    if 'dump_dict_word' in sys.argv:
        review_sentiment_classifier = _get_sentiment_classifier()
        ms = ModelSelector(char_model, word_model, review_sentiment_classifier)
        pickle_path = 'reviews'
        ms._get_statistics(word_model, num_reviews_per_threshold=10, review_length=100, pickle_path=pickle_path)
        exit(0)

    if 'plot_corr_word' in sys.argv:
        with open(os.path.join('reviews', 'WordLevelReviewGenerator_reviews.pkl'), 'rb') as file_handler_to_cache:
            generated_reviews = pickle.load(file_handler_to_cache)
        calc_and_plot_correlation([0, 0.25, 0.5, 0.75, 1], generated_reviews['sentiments'], None)

    # Most popular first word
    # seeds = [
    #     'i',
    #     'this',
    #     'the',
    #     'a',
    #     'if',
    #     'in',
    #     'when',
    #     'as',
    #     'it',
    # ]
    seeds = ['the movie was', 'this movie']
    # Average word length
    characters_in_word = 4.353004514277807

    words_number = min(100, 200)
    characters_number = min(923, round(characters_in_word * words_number))
    number_chooser = temperature_number_chooser_generator(0.5)

    reviews = {
        positiveness: ModelSelector._generate_reviews(
            char_model,
            number_chooser,
            seeds,
            positiveness,
            characters_number,
            file_path_to_cache='cache/{}%-positive-reviews-with-{}-characters-by-{}-with-{}'.format(
                positiveness,
                characters_number,
                char_model.__class__.__name__,
                number_chooser.__name__
            )
        )
        for positiveness in (0, 50, 100)
    }

    ModelSelector.get_bleu_score(reviews,
                                 test_data,
                                 [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.333, 0.333, 0.333, 0), (0.25, 0.25, 0.25, 0.25)])


    # review_sentiment_classifier = _get_sentiment_classifier()
    #
    # ms = ModelSelector(char_model, word_model, review_sentiment_classifier)
    # pickle_path = 'reviews'
    # ms.compare_models(num_reviews_per_threshold=10, review_length=100, pickle_path=pickle_path)
