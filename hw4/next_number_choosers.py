import numpy as np

from data_preparation import SpecialConstants


def clean_irrelevant_probabilities(softmax_probabilities):
    for e in SpecialConstants:
        softmax_probabilities[e.value] = 1e-10  # not 0.0 since might be used in log(x)
    return softmax_probabilities / np.sum(softmax_probabilities)


def greedy_number_chooser(next_number_softmax, clean_first=True):
    return np.argmax(clean_irrelevant_probabilities(next_number_softmax)) if clean_first else np.argmax(
        next_number_softmax)


def temperature_number_chooser_generator(temperature, clean_first=True):
    def sample(next_number_softmax):
        next_number_softmax = clean_irrelevant_probabilities(
            next_number_softmax) if clean_first else next_number_softmax
        next_number_softmax = np.asarray(next_number_softmax).astype('float64')
        next_number_softmax = np.log(next_number_softmax) / temperature
        exp_preds = np.exp(next_number_softmax)
        next_number_softmax = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, next_number_softmax, 1)
        return np.argmax(probas)

    return sample


def stochastic_number_chooser(next_number_softmax, clean_first=True):
    length = next_number_softmax.shape[0]
    probs = clean_irrelevant_probabilities(next_number_softmax) if clean_first else next_number_softmax
    indices = np.arange(length)
    return np.random.choice(indices, p=probs)
