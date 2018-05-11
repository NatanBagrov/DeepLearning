from data_preparation import *
from plotting import *
from mydnn import mydnn

number_of_classes = 10

if "__main__" == __name__:
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

    for batch_size in (128, 1024, 60000):
        print('Batch size is {}'.format(batch_size))
        dnn = mydnn(architecture, 'cross-entropy')
        history = dnn.fit(train_x, train_y, 100, batch_size, 5e-4, x_val=validation_x, y_val=validation_y)
        plot_iteration_to_loss_accuracy_from_history(history, train_x.shape[0], batch_size, 'Batch size is {}'.format(batch_size))
