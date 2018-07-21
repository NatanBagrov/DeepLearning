import os

from utils.image_type import ImageType


class InitializationProperties():
    def __init__(self, download_id, model_path, width, height) -> None:
        super().__init__()
        self.download_id = download_id
        self.model_path = model_path
        self.width = width
        self.height = height


# NOTE! I changed this to be kinda absolute. this should not brake anything, hopefully.
WEIGHTS_DIRECTORY_PATH = os.path.join(os.path.dirname(__file__), 'models', 'saved_weight')

IMAGE_OR_DOCUMENT_WEIGHT_FILE_ID_AND_FILE_PATH = InitializationProperties(
    '1BIdFV5dXPDCADKmXbGey_pMdGE7_FuvC',
    os.path.join(WEIGHTS_DIRECTORY_PATH, 'FishOrDocClassifier-model.h5'),
    None,
    None
)

IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH = {
    ImageType.IMAGES: {
        2: InitializationProperties(
            '1BMYopQX2_rh1aC1zn51ALeQGxGMrOkPH',
            os.path.join(WEIGHTS_DIRECTORY_PATH, 'ComparatorCNN-2-images-model.h5'),
            224,
            224
        ),
        4: InitializationProperties(
            '1eL2G5etJwcoWnVjFXA7iBwppN-GCjaxv',
            os.path.join(WEIGHTS_DIRECTORY_PATH, 'ComparatorCNN-4-images-model.h5'),
            224,
            224
        ),
        5: InitializationProperties(
            '1rSbbUHQZFfahKfiZsHnMicOoq4uzED9S',
            os.path.join(WEIGHTS_DIRECTORY_PATH, 'ComparatorCNN-5-images-model.h5'),
            224,
            224
        )
    },
    ImageType.DOCUMENTS: {
        2: InitializationProperties(
            '1SKm30kaAPMcCG095fSSfzgLr0XI5r4Ti',
            os.path.join(WEIGHTS_DIRECTORY_PATH, 'ComparatorCNN-2-documents-model.h5'),
            224,
            224
        ),
        4: InitializationProperties(
            '1ia3P_a3oCmfqQTOHwYzlfjQVllqXZaTk',
            os.path.join(WEIGHTS_DIRECTORY_PATH, 'ComparatorCNN-4-documents-model.h5'),
            224,
            224
        ),
        5: InitializationProperties(
            '1NsUr6QCmjtc7XnZfTfFm-J7saNDNPFj-',
            os.path.join(WEIGHTS_DIRECTORY_PATH, 'ComparatorCNN-5-documents-model.h5'),
            440,
            440
        )
    }
}

TS = (2, 4, 5)

IMAGE_TYPE_TO_MEAN = {
    ImageType.IMAGES: 100.52933494138787,
    ImageType.DOCUMENTS: 241.46115784237548,
}
IMAGE_TYPE_TO_STD = {
    ImageType.IMAGES: 65.69793156777682,
    ImageType.DOCUMENTS: 49.512839464023564,
}
