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

pickle_path = os.path.dirname(__file__)
pickle_file_names = {
    'adj': os.path.join(pickle_path, 'adj_non_adj_probas.pkl')
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


def main(root):
    dump_adjacent_and_non_adjacent_probabilities(root)
    return
    # kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=40)
    # for image_type in ImageType:
    #     for t in TS:
    #         probabilities = get_adjacent_crops_probabilities(dp, image_type, t)
    #         plt.hist(probabilities, **kwargs)
    #         plt.savefig('left-right-adj-{}-{}.png'.format(image_type, t))


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

    os.makedirs(pickle_path, exist_ok=True)
    file_path = os.path.join(pickle_path, pickle_file_names['adj'])
    print('Dumping data to ' + str(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(d, f)


def load_adjacent_and_non_adjacent_probabilities():
    with open(os.path.join(pickle_path, pickle_file_names['adj']), 'rb') as file_handler_to_cache:
        d = pickle.load(file_handler_to_cache)
        print("Loaded successfully")


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
