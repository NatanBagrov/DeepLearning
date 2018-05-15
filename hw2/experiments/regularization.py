from data_preparation import *
from plotting import *
from mydnn import mydnn
import logging

number_of_classes = 10

if "__main__" == __name__:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-5s %(name)-5s %(threadName)-5s %(filename)s:%(lineno)s - %(funcName)s() '
                               '%(''levelname)s : %(message)s',
                        datefmt="%H:%M:%S")
    fh = logging.FileHandler('logs/regularization.log')
    ch = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    # Load data
    logger.info('Loading')
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

    # for regularization, weight_decay in (('l2', 0.0), ('l2', 5e-4), ('l1', 5e-4)):
    #     title = 'No regularization' if weight_decay < 1e-9 else '{} regularization with {}'.format(regularization, weight_decay)
    #     logger.info(title)
    #     architecture = [
    #         {
    #             'input': train_x.shape[1],
    #             'output': 128,
    #             'nonlinear': 'relu',
    #             'regularization': regularization,
    #         },
    #         {
    #             'input': 128,
    #             'output': number_of_classes,
    #             'nonlinear': 'sot-max',
    #             'regularization': regularization,
    #         },
    #     ]
    #     model = mydnn(architecture, 'cross-entropy', weight_decay=weight_decay)
    #     history = model.fit(train_x, train_y, 100, batch_size, 0.01, x_val=validation_x, y_val=validation_y)
    #     plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=title)
    #     validation_accuracies = [history_entry['validation accuracy'] for history_entry in history]
    #     logger.info('Best on validation set achieved on epoch {} and {}'.format(1 + np.argmax(validation_accuracies),
    #                                                                       np.max(validation_accuracies)))

    epochs = 200

    for regularization, weight_decays in (
            ('l1', sorted(list(set.union(set(np.linspace(0.0, 3e-4, num=10)), {5e-4})))),
            ('l2', sorted(list(set.union(set(np.linspace(0.0, 1e-3, num=10)), {5e-4})))),
    ):
        best_validation_accuracies = list()
        best_train_accuracies = list()

        for weight_decay in weight_decays:
            title = 'No regularization' if weight_decay < 1e-9 else '{} regularization with {}'.format(regularization, weight_decay)
            logger.info(title)
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
            history = model.fit(train_x, train_y, epochs, batch_size, 0.01, x_val=validation_x, y_val=validation_y)
            # plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=title)
            validation_accuracies = [history_entry['validation accuracy'] for history_entry in history]
            best_validation_accuracy = np.max(validation_accuracies)
            logger.info('Best on validation set achieved on epoch {} and {}'.format(1 + np.argmax(validation_accuracies),
                                                                                    best_validation_accuracy))
            plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=title)
            best_validation_accuracies.append(best_validation_accuracy)
            best_train_accuracies.append(history[-1]['train accuracy'])

        plt.figure()
        train_handle, = plt.plot(weight_decays, best_validation_accuracies, label='Train accuracy')
        validation_handle, = plt.plot(weight_decays, best_train_accuracies, label='Validation accuracy')
        plt.title('Accuracies with {} regularization after {} epochs'.format(regularization, epochs))
        plt.xlabel('Weight decay')
        plt.ylabel('Accuracy')
        plt.legend(handles=[train_handle, validation_handle], loc='upper right')
        plt.savefig('graphs/accuracies with {} regularization'.format(regularization))
        plt.close()

    plt.show()
