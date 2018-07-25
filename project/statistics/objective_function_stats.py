import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from constants import IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH, IMAGE_TYPE_TO_MEAN, \
    IMAGE_TYPE_TO_STD, TS
from models.comparator_cnn import ComparatorCNN
from solvers.solver_greedy import SolverGreedy
from utils.data_provider import DataProvider
from utils.image_type import ImageType
from utils.pickle_helper import PickleHelper
from utils.shredder import Shredder

root_path = os.path.dirname(__file__)
dict_file_names = {
    'log_obj': os.path.join(root_path, 'greedy_log_obj_probas.pkl'),
    'plots': os.path.join(root_path, 'plots', 'objective')
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

    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(dict_file_names['log_obj'])
    PickleHelper.dump(d, file_path)


def load_reconstruction_objective_values():
    d = PickleHelper.load(dict_file_names['log_obj'])
    for image_type in ImageType:
        for t in TS:
            correct_log_obj, incorrect_log_obj = np.array(d[image_type][t]['correct']), \
                                                 np.array(d[image_type][t]['incorrect'])
            correct_log_obj = correct_log_obj[~np.isnan(correct_log_obj)]
            incorrect_log_obj = incorrect_log_obj[~np.isnan(incorrect_log_obj)]
            correct_mean, incorrect_mean = np.mean(correct_log_obj), np.mean(incorrect_log_obj)
            correct_var, incorrect_var = np.var(correct_log_obj), np.var(incorrect_log_obj)
            category = 'Fish' if image_type == ImageType.IMAGES else 'Docs'
            title = "Log objective: {}, t={}".format(category, t)
            bins = np.linspace(-100.0, 0.0, 100)
            plt.hist(correct_log_obj, bins, alpha=0.5, label='correct reconstruction')
            plt.hist(incorrect_log_obj, bins, alpha=0.5, label='incorrect reconstruction')
            plt.yscale('log')
            plt.legend(loc='upper left')
            plt.xlabel('reconstruction log objective')
            plt.ylabel('number of samples')
            plt.title(title)
            os.makedirs(dict_file_names['plots'], exist_ok=True)
            plt.savefig(os.path.join(dict_file_names['plots'], '{}-{}-objective.png'.format(category, t)))
            plt.clf()
            print(
                '{}-{} correct reconstruction mean: {:.3f} var: {:.3f}, '
                'incorrect reconstruction: {:.3f} var: {:.3f}'
                    .format(category, t, correct_mean, correct_var, incorrect_mean, incorrect_var))


if __name__ == '__main__':
    if 'dump_obj' in sys.argv:
        print('Dumping ReconstructionObjective...')
        load_models()
        dump_reconstruction_objective_values()
    if 'load_obj' in sys.argv:
        print('Loading ReconstructionObjective...')
        load_reconstruction_objective_values()
