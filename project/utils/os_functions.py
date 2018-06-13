import os


def maybe_make_directories(directory_path):
    try:
        os.makedirs(directory_path)
    except FileExistsError:
        pass
