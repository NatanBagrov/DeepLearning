import os
import pickle


class PickleHelper:
    @staticmethod
    def dump(data, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.isdir(directory):
            print("PickleHelper: directory not found, creating directory.")
            os.makedirs(directory)
        print('Dumping data to ' + str(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(file_path):
        if not os.path.exists(file_path):
            print("PickleHelper: {} not found.".format(file_path))
            raise Exception("PickleHelper: {} not found.".format(file_path))
        with open(file_path, 'rb') as fh:
            data = pickle.load(fh)
        print("Loaded successfully")
        return data
