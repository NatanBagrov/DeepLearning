import math

import matplotlib.pyplot as plt

from shredder import Shredder


class Visualizer:
    @staticmethod
    def visualize_crops(crops_list, show=False, save_path=None):
        t = round(math.sqrt(len(crops_list)))
        grid_color = 255
        img = Shredder.reconstruct(crops_list)
        if t > 1:
            dx = img.shape[0]//t
            dy = img.shape[1]//t
            img[:, ::dy] = grid_color
            img[::dx, :] = grid_color
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        plt.clf()
