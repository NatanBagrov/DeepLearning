import os
import cv2
import math

from constants import *
from utils.shredder import Shredder
from utils.image_type import ImageType
from models.fish_or_doc_classifier import FishOrDocClassifier
from models.comparator_cnn import ComparatorCNN
from models.solver_with_comparator import SolverWithComparator


fish_or_doc_classifier = FishOrDocClassifier(weights_file=IMAGE_OR_DOCUMENT_WEIGHT_FILE_ID_AND_FILE_PATH[1])
image_type_to_solver_with_comparator = {
        image_type: SolverWithComparator({
            t: ComparatorCNN(t,
                             WIDTH, HEIGHT,
                             image_type,
                             IMAGE_TYPE_TO_MEAN[image_type], IMAGE_TYPE_TO_STD[image_type])
                .load_weights(IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t][1])
            for t in TS
        },
            image_type=image_type)
    for image_type in ImageType
}


def predict(images):
    reconstructed_image = Shredder.reconstruct(images)

    print('It is ', end='')
    if fish_or_doc_classifier.is_fish([reconstructed_image])[0]:
        print('image')
        solver_with_comparator = image_type_to_solver_with_comparator[ImageType.IMAGES]
    else:
        print('document')
        solver_with_comparator = image_type_to_solver_with_comparator[ImageType.DOCUMENTS]

    labels = solver_with_comparator.predict(images)

    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)


    Y = predict(images)
    return Y


