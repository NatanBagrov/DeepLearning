from data_preparation import *
from plotting import *
from mydnn import mydnn
import logging

number_of_classes = 10

if "__main__" == __name__:
    logging.getLogger('').setLevel(logging.DEBUG)

    # Load data
    train_set, validation_set, test_set = get_data()

    train_x, train_y = split_set_to_features_and_output(train_set)
    validation_x, validation_y = split_set_to_features_and_output(validation_set)
    test_x, test_y = split_set_to_features_and_output(test_set)

    train_x, validation_x, test_x = center_input(train_x, validation_x, test_x)
    # TODO: should I or should i predict even odd?
    train_y = array_to_one_hot(train_y, number_of_classes=number_of_classes)
    validation_y = array_to_one_hot(validation_y, number_of_classes=number_of_classes)
    test_y = array_to_one_hot(test_y, number_of_classes=number_of_classes)

    logging.info('Train shape {} mean {} std {}'.format(train_x.shape, np.mean(train_x), np.std(train_x)))
    logging.info('Validation shape {} mean {} std {}. 30 examples is {} part'.format(validation_x.shape, np.mean(validation_x), np.std(validation_x), 30 / validation_x.shape[0]))
    logging.info('Test shape {} mean {} std {}'.format(test_x.shape, np.mean(test_x), np.std(test_x)))

    number_of_samples = train_x.shape[0]
    batch_size = 128

    for regularization, weight_decay in (('l2', 0.0), ('l2', 5e-4), ('l1', 5e-4)):
        title = 'No regularization' if weight_decay < 1e-9 else '{} regularization with {}'.format(regularization, weight_decay)
        print(title)
        architecture = [
            {
                'input': train_x.shape[1],
                'output': 128,
                'nonlinear': 'relu',
                'regularization': regularization,
            },
            {
                'input': 128,
                'output': number_of_classes,
                'nonlinear': 'sot-max',
                'regularization': regularization,
            },
        ]
        model = mydnn(architecture, 'cross-entropy', weight_decay=weight_decay)
        history = model.fit(train_x, train_y, 100, batch_size, 0.01, x_val=validation_x, y_val=validation_y)
        plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=title)
        validation_accuracies = [history_entry['validation accuracy'] for history_entry in history]
        print('Best on validation set achieved on epoch {} and {}'.format(1 + np.argmax(validation_accuracies),
                                                                          np.max(validation_accuracies)))

    plt.show()

    for regularization in ('l1', 'l2'):
        for weight_decay in np.logspace(-7, 1, num=(1-(-7)+1)):
            title = 'No regularization' if weight_decay < 1e-9 else '{} regularization with {}'.format(regularization, weight_decay)
            print(title)
            architecture = [
                {
                    'input': train_x.shape[1],
                    'output': 128,
                    'nonlinear': 'relu',
                    'regularization': regularization,
                },
                {
                    'input': 128,
                    'output': number_of_classes,
                    'nonlinear': 'sot-max',
                    'regularization': regularization,
                },
            ]
            model = mydnn(architecture, 'cross-entropy', weight_decay=weight_decay)
            history = model.fit(train_x, train_y, 100, batch_size, 0.01, x_val=validation_x, y_val=validation_y)
            # plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=title)
            validation_accuracies = [history_entry['validation accuracy'] for history_entry in history]
            print('Best on validation set achieved on epoch {} and {}'.format(1 + np.argmin(validation_accuracies),
                                                                              np.min(validation_accuracies)))

