import os
import sys
import time

from keras.layers import np
from sklearn.model_selection import train_test_split
from vis.visualization import visualize_cam

from constants import IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH
from models.comparator_cnn import ComparatorCNN
from utils.data_manipulations import shred_and_resize_to
from utils.data_provider import DataProvider
from utils.image_type import ImageType
import matplotlib.pyplot as plt


class ComparatorActivationMap:
    def __init__(self, comparator: ComparatorCNN) -> None:
        super().__init__()
        self._comparator = comparator

    def visualize_activations(self, stacked_shreds):
        # TODO: hard-coded left and right indexes. change if you like. don't want to spend time.
        if self._comparator.t > 2:
            left, right = self._comparator.t + 1, self._comparator.t + 2
        else:
            left, right = 0, 1

        tensor = ComparatorCNN._prepare_left_right_check(stacked_shreds[0][left], stacked_shreds[0][right])
        root = os.path.join(os.path.dirname(__file__), 'class_activation_maps')
        os.makedirs('class_activation_maps', exist_ok=True)
        for i, layer in enumerate(self._comparator._model.layers[1:]):
            try:
                layer_name = layer.name
                if 'conv' not in layer_name:
                    continue
                cam = visualize_cam(self._comparator._model, i, None, tensor)
                plt.imshow(stacked_shreds[0][left], cmap='gray')
                plt.imshow(cam, cmap='hot', interpolation='nearest', alpha=0.15)
                plt.title('#{}:{}'.format(i, layer_name))
                current_milli_time = lambda: int(round(time.time() * 1000))
                plt.savefig(
                    os.path.join(root, '{}_{}_{}.png'.format(i, layer_name, current_milli_time())))
                plt.clf()
                print(cam)
            except:
                print("Exception!")


def main():
    if 'large' in sys.argv:
        number_of_samples = sys.maxsize
    else:
        number_of_samples = 20

    ts = list()

    if '2' in sys.argv:
        ts.append(2)

    if '4' in sys.argv:
        ts.append(4)

    if '5' in sys.argv:
        ts = [5, ]

    if 0 == len(ts):
        ts = (2, 4, 5)

    image_types = list()

    if 'image' in sys.argv:
        image_types.append(ImageType.IMAGES)

    if 'document' in sys.argv:
        image_types.append(ImageType.DOCUMENTS)

    if 0 == len(image_types):
        image_types = ImageType

    np.random.seed(42)

    for t in ts:
        for image_type in image_types:
            print('t={}. image type is {}'.format(t, image_type.value))

            if image_type == ImageType.IMAGES:
                get_images = DataProvider().get_fish_images
            else:
                get_images = DataProvider().get_docs_images

            images = get_images(num_samples=number_of_samples)
            images_train, images_validation = train_test_split(images, random_state=42)

            clf = ComparatorCNN(t,
                                IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].width,
                                IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].height,
                                image_type) \
                .load_weights(IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].model_path)

            cam = ComparatorActivationMap(clf)
            for image in images_train:
                cam.visualize_activations(shred_and_resize_to([image], t, (clf.width, clf.height)))


if __name__ == '__main__':
    main()
