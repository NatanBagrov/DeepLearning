import cv2
import imghdr
import os


class DataProvider:

    def __init__(self) -> None:
        super().__init__()
        self._fish_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
        self._docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'documents'))

    def get_fish_images(self, grayscaled=True, resize=None):
        return self._get_images_from_path(self._fish_path, grayscaled, resize)

    def get_docs_images(self, grayscaled=True, resize=None):
        return self._get_images_from_path(self._docs_path, grayscaled, resize)

    def _get_images_from_path(self, path, grayscaled, resize):
        images = list()
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            if imghdr.what(file_path) is None:
                continue
            im = cv2.imread(os.path.join(path, file_path))
            if grayscaled:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if resize is not None:
                im = cv2.resize(im, resize)
            images.append(im)
        return images


class LimitedDataProvider(DataProvider):

    def __init__(self, num_images_to_provide) -> None:
        super().__init__()
        self.num_images_to_provide = num_images_to_provide

    def _get_images_from_path(self, path, grayscaled, resize):
        images = list()
        for f in os.listdir(path)[:self.num_images_to_provide]:
            file_path = os.path.join(path, f)
            if imghdr.what(file_path) is None:
                continue
            im = cv2.imread(os.path.join(path, file_path))
            if grayscaled:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if resize is not None:
                im = cv2.resize(im, resize)
            images.append(im)
        return images
