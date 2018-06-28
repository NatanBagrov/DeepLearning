import os
import sys

import numpy as np

from next_number_choosers import temperature_number_chooser_generator, stochastic_number_chooser


def generate_reviews_and_write_to_files(model, num_reviews,
                                        review_length):  # TODO: maybe pass as param a next_number_chooser?
    review_generators = [fixed_threshold_review_generator_generator(0),
                         fixed_threshold_review_generator_generator(1),
                         fixed_threshold_review_generator_generator(0.5)]
    titles = ['Negative', 'Positive', 'Negative_then_positive']
    model_name = model.__class__.__name__
    dir_name = 'reviews'
    os.makedirs(dir_name, exist_ok=True)
    for title, review_generator in zip(titles, review_generators):
        filename = os.path.join(dir_name, '{}.txt'.format(model_name))
        with open(filename, "a") as f:
            f.write(title)
            f.write('\n')
        for list_of_words in review_generator(model, num_reviews, review_length, print_online=True):
            review = ' '.join(list_of_words)
            with open(filename, "a") as f:
                f.write(review)
                f.write("\n")


def generate_negative_then_positive_reviews(model, num_reviews, review_len=None, print_online=False, pos_ratio=0.5):
    '''

    :param model: the model: word or character level.
    :param num_reviews: number of reviews you wish to generate
    :param review_len: the length of each review
    :param print_online: print the next word/char while generating the review
    :param pos_ratio: when to turn negative to positive. CHOOSE 0 for negative review and 1 for positive
    :return:
    '''
    review_length = model._review_shape[0] if review_len is None else review_len
    print("Generating a review that starts negative and turns positive, by a ratio of: {}".format(pos_ratio))
    sentiment = ([0, ] * (int(review_length * (1 - pos_ratio)))) + (
            [1, ] * (review_length - int(review_length * (1 - pos_ratio))))
    number_chooser = temperature_number_chooser_generator(0.5)
    return [_generate_review_aux(model, sentiment, number_chooser, print_online) for _ in range(num_reviews)]


def _generate_review_aux(model, sentiment, number_chooser, print_online=False):
    prefixes = np.array(['the movie was', 'this movie'])
    result = list()
    for index, word in enumerate(model.generate_string(np.random.choice(prefixes), sentiment, number_chooser)):
        result.append(word)
        if print_online:
            print("{}({})".format(word, sentiment[index]), end=' ', flush=True)
    if print_online:
        print()
        print()
    return result


def fixed_threshold_review_generator_generator(threshold):
    def func(model, num_reviews, review_length, print_online=False):
        return generate_negative_then_positive_reviews(model, num_reviews, review_length, print_online,
                                                       pos_ratio=threshold)

    return func
