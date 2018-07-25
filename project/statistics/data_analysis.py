from collections import defaultdict
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.data_provider import DataProvider
from utils.image_type import ImageType
from utils.data_manipulations import shred_and_resize_to
from utils.shredder import Shredder


def get_number_of_images_with_same_patches_and_number_of_same_patches(
        data_provider: DataProvider,
        image_type: ImageType,
        t: int,
        width_height=None):
    inputs = data_provider.get_fish_images() if image_type == ImageType.IMAGES else data_provider.get_docs_images()

    if width_height is None:
        inputs = list(map(lambda image: Shredder.shred(image, t), inputs))
    else:
        inputs = shred_and_resize_to(inputs, t, width_height)

    number_of_pictures_with_same_patches = 0
    number_of_patches_with_similar_in_same_picture = 0

    for stacked_shreds in inputs:
        picture_has_similar_patches = False

        for left_shred in range(t**2):
            picture_has_similar_to_this_shred = False

            for right_shred in range(t**2):

                if left_shred != right_shred and np.all(stacked_shreds[left_shred] == stacked_shreds[right_shred]):
                    picture_has_similar_to_this_shred = True

            if picture_has_similar_to_this_shred:
                picture_has_similar_patches = True
                number_of_patches_with_similar_in_same_picture += 1

        if picture_has_similar_patches:
            number_of_pictures_with_same_patches += 1

    return \
        number_of_pictures_with_same_patches, \
        number_of_patches_with_similar_in_same_picture, \
        len(inputs), \
        len(inputs) * (t ** 2)


def main():
    data_provider = DataProvider()
    directory = 'plots'
    os.makedirs(directory, exist_ok=True)

    for width_height in (None, (224, 224), (2200 // 5, 2200 // 5)):
        print(width_height)
        type_to_bad_picutres_percent = defaultdict(list)
        type_to_bad_patches_pairs_percent = defaultdict(list)
        ts = 2, 4, 5

        for image_type in ImageType:
            print(image_type)

            for t in ts:
                print(t)

                number_of_pictures_with_same_patches, number_of_patches_with_similar_in_same_picture,\
                    total_number_of_pictures, total_number_of_patches = \
                    get_number_of_images_with_same_patches_and_number_of_same_patches(
                        data_provider,
                        image_type,
                        t,
                        width_height
                    )

                print('{} shredded to {} patches and resized to {} has {}/{} bad pictures and {}/{} patches'.format(
                    image_type,
                    t,
                    width_height,
                    number_of_pictures_with_same_patches, total_number_of_pictures,
                    number_of_patches_with_similar_in_same_picture, total_number_of_patches
                ))

                type_to_bad_picutres_percent[image_type].append(
                    number_of_pictures_with_same_patches * 100.0 / total_number_of_pictures)
                type_to_bad_patches_pairs_percent[image_type].append(
                    number_of_patches_with_similar_in_same_picture * 100.0 / total_number_of_patches)

        handles = list()

        for image_type in ImageType:
            current_handle, = plt.plot(ts, type_to_bad_picutres_percent[image_type], 'o', label=image_type.value)
            handles.append(current_handle)

        plt.title('Percent of bad images as function of t')
        plt.legend(handles)
        plt.xlabel('t')
        plt.ylabel('% of images')
        plt.savefig(os.path.join(directory, 'bad_images.png'))
        plt.show()

        handles = list()

        for image_type in ImageType:
            current_handle, = plt.plot(ts, type_to_bad_patches_pairs_percent[image_type], 'o', label=image_type.value)
            handles.append(current_handle)

        plt.title('Percent of bad crops as function of t')
        plt.legend(handles)
        plt.xlabel('t')
        plt.ylabel('% of pairs')
        plt.savefig(os.path.join(directory, 'bad_crops.png'))
        plt.show()


if __name__ == '__main__':
    main()




