import numpy as np
import itertools
import logging
from mydnn import mydnn
from plotting import plot_iteration_to_loss_accuracy_from_history
from mpl_toolkits.mplot3d import Axes3D  # Without this line I am getting Unknown projection '3d'
import matplotlib.pyplot as plt
from matplotlib import cm

low = -2.0
high = 2.0
batch_size = 128


def f(x):
    return x[:, 0] * np.exp(-np.square(x[:, 0]) - np.square(x[:, 1]))


def get_train_data(m):
    train_x = np.random.uniform(low, high, (m, 2))
    train_y = f(train_x).reshape(-1, 1)

    return train_x, train_y


def get_validation_data(m):
    return get_train_data(m)


def get_test_data(test_step):
    x1 = np.linspace(low, high, test_step) * np.linspace(low, high, test_step)
    x2 = x1
    x1, x2 = np.meshgrid(x1, x2)
    test_x = np.column_stack((np.reshape(x1, (-1,)), np.reshape(x2, (-1,))))
    test_y = f(test_x).reshape(-1, 1)

    return test_x, test_y


def log_information_about_features(set_name, x):
    logger.info('%s set with shape (%s), mean %f and std %f', set_name, ','.join(map(str, x.shape)), np.mean(x), np.std(x))


if "__main__" == __name__:
    np.random.seed(42)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-5s %(name)-5s %(threadName)-5s %(filename)s:%(lineno)s - %(funcName)s() '
                               '%(''levelname)s : %(message)s',
                        datefmt="%H:%M:%S")
    fh = logging.FileHandler('logs/regression.log')
    ch = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    test_x, test_y = get_test_data(1000)  # TODO: is it?

    log_information_about_features('test', test_x)

    for m in (100, 1000):
        logger.debug('m=%d', m)
        train_x, train_y = get_train_data(m)
        log_information_about_features('train', train_x)
        validation_x, validation_y = get_validation_data(int(0.4 * m))
        log_information_about_features('validation', validation_x)
        best_model = None
        best_model_mse = float("inf")
        best_activation, best_units = None, None

        for activation in ('relu', 'sigmoid', 'none'):
            for units in np.logspace(1, 9, num=5, base=2).astype(int):
                logger.info('activation %s with %d hidden units', activation, units)

                model = mydnn([
                    {
                        'input': train_x.shape[1],
                        'output': units,
                        'nonlinear': activation,
                        'regularization': 'l2'
                    },
                    {
                        'input': units,
                        'output': 1,
                        'nonlinear': 'none',
                        'regularization': 'l2'
                    }
                ],
                    'MSE'
                )

                history = model.fit(train_x, train_y, 3 * m, batch_size, 0.01, x_val=validation_x, y_val=validation_y)
                plot_iteration_to_loss_accuracy_from_history(
                    history, m, batch_size,
                    title='Regression with {} hidden units and {} activation for {} training samples'.format(
                        units,
                        activation,
                        m
                    )
                )
                validation_losses = np.array([history_entry['validation loss'] for history_entry in history])  #  TODO: should I measure it on validaiton
                logger.info('Final: train loss: %f. validation loss %f. best validation loss %f on %d',
                            history[-1]['train loss'],
                            validation_losses[-1],
                            np.min(validation_losses),
                            np.argmin(validation_losses)
                )

                if best_model_mse > validation_losses[-1]:
                    logger.info('Using it (model with %s activation and %d hidden units) as best', activation, units)
                    best_model_mse = validation_losses[-1]
                    best_model = model
                    best_activation, best_units = activation, units

        test_loss, = best_model.evaluate(test_x, test_y)
        logger.info('Validation loss was %f. Test loss is %f. Model was with %s and %d', best_model_mse, test_loss, best_activation, best_units)
        test_y_predicted = best_model.predict(test_x)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(test_x[:, 0], test_x[:, 1], test_y_predicted, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('graphs/Regression predicted y from {} train samples with {} activation and {} units.png'.format(m, best_activation, best_units))
        plt.close()