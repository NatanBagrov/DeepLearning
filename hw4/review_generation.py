import os
import sys

import numpy as np

from next_number_choosers import temperature_number_chooser_generator, stochastic_number_chooser


def generate_reviews_and_write_to_files(model, num_reviews):
    review_generators = [generate_negative_then_positive_reviews,
                         generate_negative_reviews,
                         generate_positive_reviews]
    titles = ['Negative_then_positive', 'Negative', 'Positive']
    model_name = model.__class__.__name__
    dir_name = 'reviews'
    os.makedirs(dir_name, exist_ok=True)
    for title, review_generator in zip(titles, review_generators):
        filename = os.path.join(dir_name, '{}.txt'.format(model_name))
        with open(filename, "a") as f:
            f.write(title)
            f.write('\n')
        for list_of_words in review_generator(model, num_reviews, print_online=True):
            review = ' '.join(list_of_words)
            with open(filename, "a") as f:
                f.write(review)
                f.write("\n")


def generate_negative_then_positive_reviews(model, num_reviews, print_online=False, pos_ratio=0.5):
    review_length = model._review_shape[0]
    print("Generating a review that starts negative and turns positive, by a ratio of: {}".format(pos_ratio))
    sentiment = ([0, ] * (int(review_length * (1 - pos_ratio)))) + (
            [1, ] * (review_length - int(review_length * (1 - pos_ratio))))
    number_chooser = temperature_number_chooser_generator(0.5)
    return [_generate_review_aux(model, sentiment, number_chooser, print_online) for _ in range(num_reviews)]


def generate_negative_reviews(model, num_reviews, print_online=False):
    review_length = model._review_shape[0]
    sentiment = [0, ] * review_length
    number_chooser = temperature_number_chooser_generator(1.0)
    return [_generate_review_aux(model, sentiment, number_chooser, print_online) for _ in range(num_reviews)]


def generate_positive_reviews(model, num_reviews, print_online=False):
    review_length = model._review_shape[0]
    sentiment = [1, ] * review_length
    number_chooser = stochastic_number_chooser
    return [_generate_review_aux(model, sentiment, number_chooser, print_online) for _ in range(num_reviews)]


def _generate_review_aux(model, sentiment, number_chooser, print_online=False):
    prefixes = np.array(['The movie was', 'I must', ''])
    result = list()
    for index, word in enumerate(model.generate_string(np.random.choice(prefixes), sentiment, number_chooser)):
        result.append(word)
        if print_online:
            print("{}({})".format(word, sentiment[index]), end=' ', flush=True)
    if print_online:
        print()
        print()
    return result
