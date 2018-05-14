from data_preparation import *
from plotting import *
from mydnn import mydnn
from collections import defaultdict
import logging
import pickle

number_of_classes = 10

if "__main__" == __name__:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-5s %(name)-5s %(threadName)-5s %(filename)s:%(lineno)s - %(funcName)s() '
                               '%(''levelname)s : %(message)s',
                        datefmt="%H:%M:%S")
    fh = logging.FileHandler('logs/architecture.log')
    ch = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    # Load data
    train_set, validation_set, test_set = get_data()

    train_x, train_y = split_set_to_features_and_output(train_set)
    validation_x, validation_y = split_set_to_features_and_output(validation_set)
    test_x, test_y = split_set_to_features_and_output(test_set)

    train_x, validation_x, test_x = standardize_input(train_x, validation_x, test_x)
    train_y = array_to_one_hot(train_y, number_of_classes=number_of_classes)
    validation_y = array_to_one_hot(validation_y, number_of_classes=number_of_classes)
    test_y = array_to_one_hot(test_y, number_of_classes=number_of_classes)

    logger.info('Train shape {} mean {} std {}'.format(train_x.shape, np.mean(train_x), np.std(train_x)))
    logger.info('Validation shape {} mean {} std {}. 30 examples is {} part'.format(validation_x.shape, np.mean(validation_x), np.std(validation_x), 30 / validation_x.shape[0]))
    logger.info('Test shape {} mean {} std {}'.format(test_x.shape, np.mean(test_x), np.std(test_x)))

    number_of_samples = train_x.shape[0]
    batch_size = 128

    configuration_to_accuracy = dict()

    def test_cnn(units, activations):
        units.append(10)
        activations.append('sot-max')

        logger.info('->'.join(map(str, units)))

        architecture = list()
        previous_level_output = train_x.shape[1]

        for current_units, current_activation in zip(units, activations):
            architecture.append({
                'input': previous_level_output,
                'output': current_units,
                'nonlinear': current_activation,
                'regularization': 'l2',
            })
            previous_level_output = current_units

        model = mydnn(architecture, 'cross-entropy')
        epochs = 30
        history = model.fit(train_x, train_y, epochs, 128, 0.01, x_val=validation_x, y_val=validation_y)
        accuracy = history[-1]['validation accuracy']
        configuration_to_accuracy[(tuple(units), tuple(activations))] = history[-1]
        logger.info('For {} and {} validation accuracy after {} epochs was {}'.format(units, activations, epochs, accuracy))

        return accuracy

    accuracy = test_cnn([], [])
    best_configuration = None
    best_configuration_accuracy = -1.0

    if best_configuration_accuracy < accuracy:
        best_configuration = [], []
        best_configuration_accuracy = accuracy
        logger.info('Updating best configuration from {} with {} to {} with {}. Improvement on {} samples. {}'.format(
            best_configuration,
            best_configuration_accuracy,
            ([], []),
            accuracy,
            (accuracy - best_configuration_accuracy) * validation_x.shape[0],
            'Seems like noise' if (accuracy - best_configuration_accuracy) * validation_x.shape[0] < 30 else ''))

    for unit1 in np.logspace(1, 9, num=5, base=2).astype(int):
        unit1 = int(unit1)
        for activation1 in ('sigmoid', 'relu'):
            accuracy = test_cnn([unit1, ], [activation1, ])

            if best_configuration_accuracy < accuracy:
                best_configuration = [unit1, ], [activation1, ]
                logger.info(
                    'Updating best configuration from {} with {} to {} with {}. Improvement on {} samples. {}'.format(
                        best_configuration,
                        best_configuration_accuracy,
                        ([], []),
                        accuracy,
                        (accuracy - best_configuration_accuracy) * validation_x.shape[0],
                        'Seems like noise' if (accuracy - best_configuration_accuracy) * validation_x.shape[
                            0] < 30 else ''))
                best_configuration_accuracy = accuracy

            for unit2 in np.logspace(1, 9, num=5, base=2).astype(int):
                for activation2 in ('sigmoid', 'relu'):
                    accuracy = test_cnn([unit1, unit2], [activation1, activation2])

                    if best_configuration_accuracy < accuracy:
                        best_configuration = [unit1, unit2], [activation1, activation2]
                        best_configuration_accuracy = accuracy
                        logging.info(
                            'Updating best configuration from {} with {} to {} with {}. Improvement on {} samples. {}'.format(
                                best_configuration,
                                best_configuration_accuracy,
                                ([], []),
                                accuracy,
                                (accuracy - best_configuration_accuracy) * validation_x.shape[0],
                                'Seems like noise' if (accuracy - best_configuration_accuracy) * validation_x.shape[
                                    0] < 30 else ''))
    logging.info(best_configuration)
    logging.info(best_configuration_accuracy)
    logging.debug(configuration_to_accuracy)
    logging.info(best_configuration)
    logging.info(best_configuration_accuracy)

    with open('configuration_to_accuracy.pkl', 'wb') as configuration_to_accuracy_handler:
        pickle.dump(configuration_to_accuracy, configuration_to_accuracy_handler)

    # Width

    depth_to_width_to_accuracy = {current_depth: defaultdict(int) for current_depth in (1,2,3)}

    for (units, activations), history_entry in configuration_to_accuracy.items():
        depth = len(units)
        width = max(units)
        depth_to_width_to_accuracy[depth][width] = max(depth_to_width_to_accuracy[depth][width],
                                                       history_entry['validation accuracy'])

    fig = plt.figure()
    handles = list()

    for depth, width_to_accuracy in depth_to_width_to_accuracy.items():
        handle, = plt.plot(list(width_to_accuracy.keys()), list(width_to_accuracy.values()), 'o', label='Depth {}'.format(depth))
        handles.append(handle)

    plt.title('Width to validation accuracy')
    plt.xlabel('width')
    plt.ylabel('accuracy')
    plt.legend(handles=handles, loc='lower right')
    plt.savefig('../graphs/width to accuracy by depth.png')

    plt.close()

    # Depth

    width_to_depth_to_best_accuracy = defaultdict(lambda: defaultdict(int))

    for (units, activations), history_entry in configuration_to_accuracy.items():
        depth = len(units)
        width = max(units)
        width_to_depth_to_best_accuracy[width][depth] = max(width_to_depth_to_best_accuracy[width][depth],
                                                       history_entry['validation accuracy'])

    fig = plt.figure()
    handles = list()

    for width, depth_to_accuracy in width_to_depth_to_best_accuracy.items():
        handle, = plt.plot(list(depth_to_accuracy.keys()), list(depth_to_accuracy.values()), 'o', label='Width {}'.format(width))
        handles.append(handle)

    plt.title('Depth to validation accuracy')
    plt.xlabel('depth')
    plt.ylabel('accuracy')
    plt.legend(handles=handles, loc='upper left')
    plt.savefig('../graphs/depth to accuracy by width.png')

    plt.close()

    # Overall number of neurons

    overall_number_of_weigts_to_best_accuracy = defaultdict(float)

    for (units, activations), history_entry in configuration_to_accuracy.items():
        overall_number_of_weigts = 0
        previous_layer_output = train_x.shape[1]

        for current_layer_output in units:
            overall_number_of_weigts += previous_layer_output * current_layer_output
            previous_layer_output = current_layer_output

        overall_number_of_weigts_to_best_accuracy[overall_number_of_weigts] = max(overall_number_of_weigts_to_best_accuracy[overall_number_of_weigts],
                                                       history_entry['validation accuracy'])

    fig = plt.figure()

    plt.plot(sorted(list(overall_number_of_weigts_to_best_accuracy.keys())),
             sorted(list(overall_number_of_weigts_to_best_accuracy.values())),
             '-o')

    plt.title('Overall number of weights to validation accuracy')
    plt.xlabel('depth')
    plt.ylabel('accuracy')
    plt.savefig('../graphs/overall number of weights to accuracy.png')

    plt.close()

