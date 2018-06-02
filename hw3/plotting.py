from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, title, show=False):
    his = history.history
    x = list(range(len(history.history['val_acc'])))

    plt.figure()
    y_1 = his['val_acc']
    y_2 = his['acc']
    plt.plot(x, y_1)
    plt.plot(x, y_2)
    plt.legend(['validation accuracy', 'training accuracy'])
    plt.title(title)
    plt.savefig('graphs/{}-accuracy.png'.format(title))

    plt.figure()
    y_1 = his['val_loss']
    y_2 = his['loss']
    plt.plot(x, y_1)
    plt.plot(x, y_2)
    plt.title(title)
    plt.legend(['validation loss', 'training loss'], loc='upper right')
    plt.savefig('graphs/{}-loss.png'.format(title))

    if show:
        plt.show()


def plot_data(x, y):
    number_of_samples = 10
    classes = np.unique(y)
    number_of_classes = classes.shape[0]
    f, axarr = plt.subplots(number_of_samples, number_of_classes)

    for class_index in range(number_of_classes):
        indices, = np.where(class_index == np.squeeze(y))
        chosen_indices = np.random.choice(indices, size=number_of_classes, replace=False)
        # chosen_indices = indices[:number_of_classes_of_samples]

        for sample_index in range(number_of_samples):
            axarr[sample_index][class_index].imshow(x[chosen_indices[sample_index]])
            axarr[sample_index][class_index].set_yticklabels([])
            axarr[sample_index][class_index].set_xticklabels([])
            axarr[sample_index][class_index].tick_params(
                axis=('x', 'y'),  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False
            )  # labels along the bottom edge are off

    plt.show()


def plot_generator(datagen, x_train, y_train):
    number_of_samples = 5
    number_of_classes = np.unique(y_train)
    mean = x_train.mean()
    std = x_train.std()
    f, axarr = plt.subplots(number_of_samples, number_of_classes)
    class_to_number_of_samples = [0,] * number_of_classes

    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=number_of_classes):
        print(x_batch.mean(), x_batch.std())

        for i in range(x_batch.shape[0]):
            current_class = np.argmax(y_batch[i])

            if class_to_number_of_samples[current_class] < number_of_samples:
                axarr[class_to_number_of_samples[current_class]][current_class].imshow(
                    (np.clip(mean + std * x_batch[i], 0, 255)).astype(np.uint8))
                axarr[class_to_number_of_samples[current_class]][current_class].set_yticklabels([])
                axarr[class_to_number_of_samples[current_class]][current_class].set_xticklabels([])
                axarr[class_to_number_of_samples[current_class]][current_class].tick_params(
                    axis=('x', 'y'),  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False
                )  # labels along the bottom edge are off
                class_to_number_of_samples[current_class] += 1

        print(class_to_number_of_samples)

        if min(class_to_number_of_samples) >= number_of_samples:
            break

    plt.show()


def plot_parameter_to_score(parameters, scores, parameter_name, title):
    plt.clf()
    parameters = np.array(parameters)
    scores = np.array(scores)
    assert parameters.shape == scores.shape

    if np.issubdtype(parameters.dtype, np.number):
        permutation = np.argsort(parameters)
        parameters = parameters[permutation]
        scores = scores[permutation]
        x = parameters
    else:
        x = np.arange(scores.shape[0])
        plt.xticks(x, parameters)

    plt.plot(x, scores, '-o')
    plt.xlabel(parameter_name)
    plt.ylabel('Cross validation average score')
    plt.title(title)
    plt.savefig('graphs/{}.png'.format(title))


def plot_parameter_influence_score(classifier_name: str, cv_result: dict):
    parameter_to_value_to_best_score = defaultdict(lambda: defaultdict(lambda: 0.0))

    for score, parameter_to_value in zip(cv_result['mean_test_score'], cv_result['params']):
        for parameter, value in parameter_to_value.items():
            parameter_to_value_to_best_score[parameter][value] = max(
                parameter_to_value_to_best_score[parameter][value],
                score
            )

    common_parameters = ' '.join(sorted(['{}={}'.format(parameter, list(value_to_best_score.keys())[0])
                                  for parameter, value_to_best_score in parameter_to_value_to_best_score.items()
                                  if 1 == len(value_to_best_score) and 'random_state' != parameter]))

    for parameter, value_to_best_score in parameter_to_value_to_best_score.items():
        if len(value_to_best_score) > 1:
            plot_parameter_to_score(
                list(value_to_best_score.keys()),
                list(value_to_best_score.values()),
                parameter,
                'Influence of {} on {} accuracy with {}'.format(parameter, classifier_name, common_parameters)
            )

