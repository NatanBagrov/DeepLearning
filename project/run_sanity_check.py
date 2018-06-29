import time
overall_start = time.time()

import os
import shutil
import numpy as np

import weights_download
from shrader_public import main as shrader
from evaluate import evaluate

np.random.seed(42)

for t in (2, 4, 5):
    for type_name in ('document', 'image'):
        input_directory_path = os.path.join('sanity', '{}_input/'.format(type_name))
        output_directory_path = os.path.join('sanity', '{}_{}_output/'.format(type_name, t))
        permutation = np.random.permutation(t ** 2)

        print(input_directory_path, '->', output_directory_path)

        if os.path.exists(output_directory_path):
            shutil.rmtree(output_directory_path)

        os.makedirs(output_directory_path)
        shrader(permutation, input_directory_path, output_directory_path, t)
        test_start = time.time()
        prediction = evaluate(output_directory_path)
        test_duration = time.time() - test_start

        assert isinstance(prediction, list)

        is_correct = list(map(lambda true_predicted: true_predicted[0] == true_predicted[1],
                              zip(permutation, prediction)))
        accuracy = sum(is_correct) * 1.0 / len(is_correct)

        if not all(is_correct):
            print('WARNING:', end='')

        print('accuracy is', accuracy, permutation, end='')

        if not all(is_correct):
            print('!=', prediction)
        else:
            print()

        print('Test duration:', test_duration, 'seconds')

overall_seconds = time.time() - overall_start
print('Overall duration:', overall_seconds, 'seconds')
