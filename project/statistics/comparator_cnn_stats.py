import os
import pickle
import sys

import numpy as np

from constants import IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH, TS, IMAGE_TYPE_TO_MEAN, \
    IMAGE_TYPE_TO_STD
from models.comparator_cnn import ComparatorCNN
from utils.data_manipulations import shred_and_resize_to
from utils.data_provider import DataProvider
from utils.image_type import ImageType
import matplotlib.pyplot as plt

from utils.pickle_helper import PickleHelper

root_path = os.path.dirname(__file__)
dict_file_names = {
    'adj': os.path.join(root_path, 'adj_non_adj_probas.pkl'),
    'plots': os.path.join(root_path, 'plots', 'adj_non_adj')
}
image_type_to_t_to_comparator = dict()


def load_models():
    global image_type_to_t_to_comparator
    image_type_to_t_to_comparator = {
        image_type: {
            t: ComparatorCNN(t,
                             IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].width,
                             IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].height,
                             image_type,
                             IMAGE_TYPE_TO_MEAN[image_type], IMAGE_TYPE_TO_STD[image_type])
                .load_weights(IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].model_path)
            for t in TS
        }
        for image_type in ImageType
    }


def dump_adjacent_and_non_adjacent_probabilities():
    d = {
        image_type: {
            t: {
                'adj': None,
                'non_adj': None
            }
            for t in TS
        }
        for image_type in ImageType
    }
    dp = DataProvider()
    for image_type in ImageType:
        for t in TS:
            print("Getting stats for {}-{}...".format(image_type, t))
            d[image_type][t]['non_adj'] = get_non_adjacent_crops_probabilities(dp, image_type, t)
            d[image_type][t]['adj'] = get_adjacent_crops_probabilities(dp, image_type, t)

    os.makedirs(root_path, exist_ok=True)
    PickleHelper.dump(d, dict_file_names['adj'])


def load_adjacent_and_non_adjacent_probabilities():
    d = PickleHelper.load(dict_file_names['adj'])
    for image_type in ImageType:
        for t in TS:
            adj_probas, non_adj_probas = d[image_type][t]['adj'], d[image_type][t]['non_adj']
            adj_mean, non_adj_mean = np.mean(adj_probas), np.mean(non_adj_probas)
            adj_var, non_adj_var = np.var(adj_probas), np.var(non_adj_probas)
            category = 'Fish' if image_type == ImageType.IMAGES else 'Docs'
            title = "Left to Right: {}, t={}".format(category, t)
            bins = np.linspace(0.00, 1.00, 100)
            plt.hist(adj_probas, bins, alpha=0.5, label='adjacent')
            plt.hist(non_adj_probas, bins, alpha=0.5, label='non adjacent')
            plt.yscale('log')
            plt.legend(loc='upper left')
            plt.xlabel('adjacency probability')
            plt.ylabel('number of samples')
            plt.title(title)
            os.makedirs(dict_file_names['plots'], exist_ok=True)
            plt.savefig(os.path.join(dict_file_names['plots'], '{}-{}-adj-nonadj.png'.format(category, t)))
            plt.clf()
            print(
                '{}-{} adj mean: {:.3f} var: {:.3f}, non_adj_mean: {:.3f} var: {:.3f}'
                    .format(category, t, adj_mean, adj_var, non_adj_mean, non_adj_var))


def get_adjacent_crops_probabilities(dp: DataProvider, image_type: ImageType, t):
    """
    This function calculates the output of the comparator on adjacent crops
    :return: list of probabilities (scalars)
    """
    comparator = image_type_to_t_to_comparator[image_type][t]
    inputs = dp.get_fish_images() if image_type == ImageType.IMAGES else dp.get_docs_images()
    inputs = shred_and_resize_to(inputs, t, (comparator.width, comparator.height))
    adj_probabilities = []
    for stacked_shreds in inputs:
        for left_idx in range(t ** 2):
            right_idx = left_idx + 1
            if left_idx % t == t - 1:
                continue
            softmax = comparator.predict_is_left_probability([stacked_shreds[left_idx]], [stacked_shreds[right_idx]])
            adj_probabilities.append(softmax[0][1])

    return adj_probabilities


def get_non_adjacent_crops_probabilities(dp: DataProvider, image_type: ImageType, t):
    """
    This function calculates the output of the comparator on non adjacent crops
    :return: list of probabilities (scalars)
    """
    comparator = image_type_to_t_to_comparator[image_type][t]
    inputs = dp.get_fish_images() if image_type == ImageType.IMAGES else dp.get_docs_images()
    inputs = shred_and_resize_to(inputs, t, (comparator.width, comparator.height))
    non_adj_probabilities = []
    for stacked_shreds in inputs:
        left_idx, right_idx = 0, 1
        while left_idx + 1 == right_idx:
            left_idx, right_idx = tuple(np.random.choice(t ** 2, 2, replace=False))
        softmax = comparator.predict_is_left_probability([stacked_shreds[left_idx]], [stacked_shreds[right_idx]])
        non_adj_probabilities.append(softmax[0][1])

    return non_adj_probabilities


if __name__ == '__main__':
    if 'dump_adj' in sys.argv:
        print('Dumping ComparatorCNN...')
        load_models()
        dump_adjacent_and_non_adjacent_probabilities()
    if 'load_adj' in sys.argv:
        print('Loading ComparatorCNN...')
        load_adjacent_and_non_adjacent_probabilities()
