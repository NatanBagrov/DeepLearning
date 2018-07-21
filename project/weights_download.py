import os
import requests

from constants import \
    WEIGHTS_DIRECTORY_PATH, \
    IMAGE_OR_DOCUMENT_WEIGHT_FILE_ID_AND_FILE_PATH, \
    IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH, \
    TS
from utils.image_type import ImageType


def download_file_from_google_drive(id, destination):
    print('{}->{}'.format(id, destination))
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
    os.makedirs(WEIGHTS_DIRECTORY_PATH, exist_ok=True)

    download_file_from_google_drive(IMAGE_OR_DOCUMENT_WEIGHT_FILE_ID_AND_FILE_PATH.download_id,
                                    IMAGE_OR_DOCUMENT_WEIGHT_FILE_ID_AND_FILE_PATH.model_path)

    for image_type in ImageType:
        for t in TS:
            download_file_from_google_drive(
                IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].download_id,
                IMAGE_TYPE_TO_T_TO_COMPARATOR_CNN_WEIGHT_FILE_ID_AND_FILE_PATH[image_type][t].model_path
            )


main()
