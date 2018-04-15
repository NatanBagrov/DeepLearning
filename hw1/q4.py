import pickle, gzip, urllib.request, json
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-5s %(name)-5s %(threadName)-5s %(filename)s:%(lineno)s - %(funcName)s() '
                           '%(''levelname)s : %(message)s',
                    datefmt="%H:%M:%S")
logger = logging.getLogger()


def timed(func):
    def func_wrapper(*args, **kwargs):
        start = time.time()
        logger.info("-" * 75)
        logger.info(str(func))
        res = func(*args, **kwargs)
        logger.info("{0} - Total running time: {1} seconds".format(str(func), time.time() - start))
        return res

    return func_wrapper


def get_data():
    path = os.path.join(os.getcwd(), 'mnist.pkl.gz')
    if not os.path.isfile(path):
        data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        return train_set, valid_set, test_set


def train_analytic_ridge_regressor(X, y, lmbda):
    m = X.shape[0]
    I = np.eye(X.shape[1])

    w = np.linalg.inv(X.transpose().dot(X) * (1.0 / m) + I * 2.0 * lmbda).dot(1.0 / m * X.transpose().dot(y))
    b = np.mean(y)
    # This value we are trying to minimize
    loss = (np.linalg.norm(np.dot(X, w) + b - y) ** 2) / 2.0 / m + lmbda * np.linalg.norm(w)

    logging.debug('Analytical for lambda={} loss value is {}'.format(lmbda, loss))

    return w, b


def train_gd_rigde_regressor(X, y, regularization_coefficient,
                             learning_rate, number_of_steps,
                             initial_w=None, initial_b=0.0):
    m, n = X.shape

    if initial_w is None:
        initial_w = np.zeros((n,))

    w = initial_w
    b = initial_b

    x_transpose_dot_x = X.transpose().dot(X)
    x_transpose_dot_y = X.transpose().dot(y)

    for step in range(number_of_steps):
        # TODO: should I really calculate b in this way?
        dl_by_db = b - (1.0 / m) * np.sum(y)
        b -= learning_rate * dl_by_db

        dl_by_dw = 1.0 / m * (x_transpose_dot_x.dot(w) - x_transpose_dot_y) + 2.0 * regularization_coefficient * w
        w -= learning_rate * dl_by_dw

    # This value we are trying to minimize
    loss = (np.linalg.norm(np.dot(X, w) + b - y) ** 2) / 2.0 / m + regularization_coefficient * np.linalg.norm(w)

    logging.debug('Gradient descent for lambda={} loss value is {}'.format(regularization_coefficient, loss))

    return w, b


def zero_one_loss(X, y, w, b):
    y_predicted = np.sign(X.dot(w) + b)

    return np.count_nonzero(y != y_predicted) / y.size


def squared_loss(X, y, w, b):
    y_predicted = X.dot(w) + b
    error = y_predicted - y

    return np.linalg.norm(error) ** 2 / y.size


def calculate_learning_losses(train_x, train_y,
                              regularization_coefficient,
                              learning_rate, number_of_steps,
                              test_xs, test_ys,
                              loss_functions):
    n = train_x.shape[1]
    w = np.zeros((n,))
    b = 0.0
    learning_losses = [
        [np.zeros((number_of_steps,))
         for loss_index in range(len(loss_functions))]
        for test_index in range(len(test_xs))
    ]
    steps = range(number_of_steps)

    for current_step in steps:
        w, b = train_gd_rigde_regressor(train_x, train_y, regularization_coefficient, learning_rate, 1,
                                        initial_w=w, initial_b=b)

        for test_index, (current_test_x, current_test_y) in enumerate(zip(test_xs, test_ys)):
            for loss_index, loss in enumerate(loss_functions):
                learning_losses[test_index][loss_index][current_step] = loss(current_test_x, current_test_y, w, b)

    return learning_losses


def plot_learning_losses(train_x, train_y, test_x, test_y, regularization_coefficient, learning_rate, number_of_steps):
    # they really mean test?! not validation? -> I think so, since we used the validation to find the best coef,
    # and the performance of the model is actually evaluated with respect to the test-set
    learning_losses = calculate_learning_losses(train_x, train_y,
                                                regularization_coefficient,
                                                learning_rate, number_of_steps,
                                                (train_x, test_x), (train_y, test_y),
                                                (zero_one_loss, squared_loss))

    plt.rc('text', usetex=False)
    train_zero_one, = plt.plot(learning_losses[0][0], label='train $L_{0-1}$')
    train_squared, = plt.plot(learning_losses[0][1], label='train $L_{squared}$')
    test_zero_one, = plt.plot(learning_losses[1][0], label='test $L_{0-1}$')
    test_squared, = plt.plot(learning_losses[1][1], label='test $L_{squared}$')
    plt.legend(handles=[train_zero_one, train_squared, test_zero_one, test_squared])
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.show()


def plot_train_validation_losses_with_respect_to_lambdas(lambdas, analytical_models,gradient_descent_models):
    zero_one_analytical_losses_on_training_set = \
        [zero_one_loss(train_x, train_y, *analytical_models[i]) for i in range(len(analytical_models))]
    zero_one_analytical_losses_on_validation_set = \
        [zero_one_loss(valid_x, valid_y, *analytical_models[i]) for i in range(len(analytical_models))]
    squared_analytical_losses_on_training_set = \
        [squared_loss(train_x, train_y, *analytical_models[i]) for i in range(len(analytical_models))]
    squared_analytical_losses_on_validation_set = \
        [squared_loss(valid_x, valid_y, *analytical_models[i]) for i in range(len(analytical_models))]

    print('analytical:')

    for i in range(len(analytical_models)):
        print("lambda {0:e} -> train: 0-1: {1:.3f}, squared: {2:.3f}; validation: 0-1: {3:.3f}, squared: {4:.3f}".format(
            lambdas[i],
            zero_one_analytical_losses_on_training_set[i],
            squared_analytical_losses_on_training_set[i],
            zero_one_analytical_losses_on_validation_set[i],
            squared_analytical_losses_on_validation_set[i],
        ))

    zero_one_gd_losses_on_training_set = \
        [zero_one_loss(train_x, train_y, *gradient_descent_models[i]) for i in range(len(gradient_descent_models))]
    zero_one_gd_losses_on_validation_set = \
        [zero_one_loss(valid_x, valid_y, *gradient_descent_models[i]) for i in range(len(gradient_descent_models))]
    squared_gd_losses_on_training_set = \
        [squared_loss(train_x, train_y, *gradient_descent_models[i]) for i in range(len(gradient_descent_models))]
    squared_gd_losses_on_validation_set = \
        [squared_loss(valid_x, valid_y, *gradient_descent_models[i]) for i in range(len(gradient_descent_models))]

    print('gradient descend:')

    for i in range(len(gradient_descent_models)):
        print("lambda {0:e} -> train: 0-1: {1:.3f}, squared: {2:.3f}; validation: 0-1: {3:.3f}, squared: {4:.3f}".format(
            lambdas[i],
            zero_one_gd_losses_on_training_set[i],
            squared_gd_losses_on_training_set[i],
            zero_one_gd_losses_on_validation_set[i],
            squared_gd_losses_on_validation_set[i],
        ))

    f, shared = plt.subplots(2, sharex=True)
    plt.rc('text')
    shared[0].semilogx(lambdas, zero_one_analytical_losses_on_training_set, 'r', label="a.train $L_{0-1}$")
    shared[0].semilogx(lambdas, zero_one_analytical_losses_on_validation_set, 'g', label="a.validation $L_{0-1}$")
    shared[0].semilogx(lambdas, squared_analytical_losses_on_training_set, 'b', label="a.train $L_{squared}$")
    shared[0].semilogx(lambdas, squared_analytical_losses_on_validation_set, 'y', label="a.validation $L_{squared}$")

    shared[1].semilogx(lambdas, zero_one_gd_losses_on_training_set, 'r', label="gd.train $L_{0-1}$")
    shared[1].semilogx(lambdas, zero_one_gd_losses_on_validation_set, 'g', label="gd.validation $L_{0-1}$")
    shared[1].semilogx(lambdas, squared_gd_losses_on_training_set, 'b', label="gd.train $L_{squared}$")
    shared[1].semilogx(lambdas, squared_gd_losses_on_validation_set, 'y', label="gd.validation $L_{squared}$")

    shared[0].legend()
    shared[1].legend()
    plt.xlabel("Lambdas")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    train, valid, test = get_data()

    train_x, train_y = np.array(train[0]), np.array(train[1])
    valid_x, valid_y = np.array(valid[0]), np.array(valid[1])
    test_x, test_y = np.array(test[0]), np.array(test[1])

    # normalizing
    train_mean = np.mean(train_x, axis=0)
    train_x, valid_x, test_x = train_x - train_mean, valid_x - train_mean, test_x - train_mean

    # reformatting labels
    even_odd = np.vectorize(lambda x: 1 if x % 2 == 0 else -1)
    train_y, valid_y, test_y = even_odd(train_y), even_odd(valid_y), even_odd(test_y)

    # Train the analytical model with 8 lambdas, and get 8 models
    lambdas = np.logspace(-5, 2, num=8)

    analytical_models = list(map(
        lambda current_lambda: train_analytic_ridge_regressor(train_x, train_y, current_lambda),
        lambdas
    ))

    # Train the gradient descent model with 8 lambdas, and get 8 models
    learning_rate = 0.001
    number_of_steps = 100

    gradient_descent_models = list(map(
        lambda current_lambda: train_gd_rigde_regressor(train_x, train_y, current_lambda, learning_rate,
                                                        number_of_steps),
        lambdas
    ))

    plot_train_validation_losses_with_respect_to_lambdas(lambdas, analytical_models, gradient_descent_models)

    # Test best analytical model
    zero_one_analytical_losses_on_validation_set = \
        [zero_one_loss(valid_x, valid_y, *analytical_models[i]) for i in range(len(analytical_models))]
    best_analytical_model = analytical_models[np.argmin(zero_one_analytical_losses_on_validation_set)]

    zero_one_gd_losses_on_validation_set = \
        [zero_one_loss(valid_x, valid_y, *gradient_descent_models[i]) for i in range(len(gradient_descent_models))]
    best_gd_model_index = np.argmin(zero_one_gd_losses_on_validation_set)
    best_gd_model, best_gd_lambda = gradient_descent_models[best_gd_model_index], lambdas[best_gd_model_index]

    logger.debug("The distance between weights of 2 models: {}, the distance between biases of 2 models: {}"
                 .format(np.linalg.norm(best_analytical_model[0] - best_gd_model[0]),
                         np.linalg.norm(best_analytical_model[1] - best_gd_model[1])))

    test_analytical_zero_one_loss = zero_one_loss(test_x, test_y, *best_analytical_model)
    test_analytical_squared_loss = squared_loss(test_x, test_y, *best_analytical_model)

    print('analytical: test: 0-1 {0}, squared: {1}'.format(
        test_analytical_zero_one_loss,
        test_analytical_squared_loss
    ))

    test_gd_zero_one_loss = zero_one_loss(test_x, test_y, *best_gd_model)
    test_gd_squared_loss = squared_loss(test_x, test_y, *best_gd_model)

    print('gradient descent: test: 0-1 {0}, squared: {1}'.format(
        test_gd_zero_one_loss,
        test_gd_squared_loss
    ))

    plot_learning_losses(train_x, train_y, test_x, test_y, best_gd_lambda, learning_rate, number_of_steps)
