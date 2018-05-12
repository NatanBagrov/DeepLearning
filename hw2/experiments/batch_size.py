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

    architecture = [
        {
            'input': train_x.shape[1],
            'output': 128,
            'nonlinear': 'relu',
            'regularization': 'l2',
        },
        {
            'input': 128,
            'output': number_of_classes,
            'nonlinear': 'sot-max',
            'regularization': 'l2',
        },
    ]

    batch_sizes = (128, 1024, 60000)
    time_per_epoch = list()
    best_validation_accuracy = list()

    for current_batch_size in batch_sizes:
        print('Batch size is {}'.format(current_batch_size))
        dnn = mydnn(architecture, 'cross-entropy')
        history = dnn.fit(train_x, train_y, 100, current_batch_size, 0.01, x_val=validation_x, y_val=validation_y)
        plot_iteration_to_loss_accuracy_from_history(history, train_x.shape[0], current_batch_size, 'Batch size is {}'.format(current_batch_size))
        average_time = sum((history_entry['seconds'] for history_entry in history)) / len(history)
        print('Average time {} seconds per epoch'.format(average_time))
        time_per_epoch.append(average_time)
        best_validation_accuracy.append(max([history_entry['validation accuracy'] for history_entry in history]))

    plt.figure()
    plt.plot(batch_sizes, time_per_epoch)
    plt.xlabel('Batch size')
    plt.ylabel('Seconds per epoch')
    plt.title('Epoch time from batch size')
    plt.savefig('graphs/Batch size to time per epoch.png')
    plt.figure()
    plt.plot(batch_sizes, best_validation_accuracy)
    plt.xlabel('Batch size')
    plt.ylabel('Best validation accuracy')
    plt.title('Validation accuracy from batch size')
    plt.savefig('graphs/Batch size to validation accuracy.png')
    plt.show()

    # TODO: design and run more experiments to support your hypothesis
