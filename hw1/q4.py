import pickle, gzip, urllib.request, json
import numpy as np
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


@timed
def train_analytic_linear_regressor(X, y):
    lmbdas = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10, 100]
    m = X.shape[0]
    I = np.eye(X.shape[1])
    # TODO: Copied from Q3 solution, what about the 1*b?
    return lmbdas, [np.linalg.inv(X.transpose().dot(X) * (1 / m) + I * 2 * lmbda).dot(X.transpose().dot(y)) * (1 / m)
                    for lmbda in lmbdas]


def train_gd_linear_regressor(X, y):
    pass


@timed
def zero_one_loss(X, y, w, b):
    m = X.shape[0]
    Ib = np.ones(X.shape[0]) * b
    x_classified = np.sign(X.dot(w) + Ib)
    return np.sum(y - x_classified) / m


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
    lmbdas, analytical_models = train_analytic_linear_regressor(train_x, train_y)
    zero_one_analytical_losses = \
        [zero_one_loss(train_x, train_y, analytical_models[i], 0) for i in range(len(analytical_models))]
    [print("{0} -> {1}".format(lmbdas[i], zero_one_analytical_losses[i])) for i in range(len(analytical_models))]
