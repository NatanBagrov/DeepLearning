import numpy as np

from next_number_choosers import temperature_number_chooser_generator, stochastic_number_chooser


def generate_negative_then_positive_reviews(model, num_reviews, print_online=False):
    review_length = model._review_shape[0]
    sentiment = ([0, ] * (review_length // 2)) + ([1, ] * (review_length - review_length // 2))
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
    prefixes = np.array(['The movie was', 'I must'])
    result = list()
    for index, word in enumerate(model.generate_string(np.random.choice(prefixes), sentiment, number_chooser)):
        result.append(word)
        if print_online:
            print("{}({})".format(word, sentiment[index]), end=' ', flush=True)
    if print_online:
        print()
        print()
    return result
