import os
import pickle
import sys

import numpy as np

from constants import IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH, IMAGE_TYPE_TO_MEAN, \
    IMAGE_TYPE_TO_STD, TS
from models.comparator_cnn import ComparatorCNN
from solvers.solver_greedy import SolverGreedy
from utils.data_provider import DataProvider
from utils.image_type import ImageType
from utils.shredder import Shredder

pickle_path = os.path.dirname(__file__)
pickle_file_names = {
    'log_obj': os.path.join(pickle_path, 'greedy_log_obj_probas.pkl')
}
image_type_to_solver_with_comparator = dict()


def load_models():
    global image_type_to_solver_with_comparator
    image_type_to_solver_with_comparator = {
        image_type: SolverGreedy({
            t: ComparatorCNN(t,
                             IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].width,
                             IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].height,
                             image_type,
                             IMAGE_TYPE_TO_MEAN[image_type], IMAGE_TYPE_TO_STD[image_type])
                .load_weights(IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].model_path)
            for t in TS
        },
            image_type=image_type)
        for image_type in ImageType
    }


def get_reconstruction_objective_values(dp: DataProvider, image_type: ImageType, t):
    """
    This function dumps the output of the objective function given a permutation from the greedy comparator
    :return: a tuple of 2 lists of log probabilities (scalars)
    """
    solver = image_type_to_solver_with_comparator[image_type]
    inputs = dp.get_fish_images() if image_type == ImageType.IMAGES else dp.get_docs_images()
    inputs = [Shredder.shred(im, t, shuffle_shreds=False) for im in inputs]
    correct_reconstruction_probas = []
    incorrect_reconstruction_probas = []
    for stacked_shreds in inputs:
        predicted_permutation, log_objective = solver.predict(stacked_shreds, return_log_objective=True)
        if np.array_equal(predicted_permutation, np.arange(t ** 2)):
            correct_reconstruction_probas.append(log_objective)
        else:
            incorrect_reconstruction_probas.append(log_objective)

    return correct_reconstruction_probas, incorrect_reconstruction_probas


def dump_reconstruction_objective_values():
    d = {
        image_type: {
            t: {
                'correct': None,
                'incorrect': None
            }
            for t in TS
        }
        for image_type in ImageType
    }
    dp = DataProvider()
    for image_type in ImageType:
        for t in TS:
            print("Getting stats for {}-{}...".format(image_type, t))
            d[image_type][t]['correct'], \
            d[image_type][t]['incorrect'] = get_reconstruction_objective_values(dp, image_type, t)

    os.makedirs(pickle_path, exist_ok=True)
    file_path = os.path.join(pickle_file_names['log_obj'])
    print('Dumping data to ' + str(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(d, f)


def load_reconstruction_objective_values():
    with open(os.path.join(pickle_file_names['log_obj']), 'rb') as file_handler_to_cache:
        d = pickle.load(file_handler_to_cache)
        print("Loaded successfully")


if __name__ == '__main__':
    if 'dump_obj' in sys.argv:
        print('Dumping ReconstructionObjective...')
        load_models()
        dump_reconstruction_objective_values()
    if 'load_obj' in sys.argv:
        print('Loading ReconstructionObjective...')
        load_reconstruction_objective_values()
