import numpy as np
from time import time
from utils.LossFunctions import CrossEntropy
from utils.RegularizationMethods import regularization_method_name_to_class
from utils.ActivationFunctions import activation_function_name_to_class
from utils.LossFunctions import loss_name_to_class
from utils.Metrics import accuracy
from FullyConnectedLayer import FullyConnectedLayer
from graph.Variable import Variable
from graph.BinaryOperations import Add, Multiply


class mydnn:
    def __init__(self, architecture: list, loss, weight_decay=0.0):
        weight_decay = np.float64(weight_decay)
        self._x_variable = Variable(None)  # TODO: call it placeholder
        self._y_variable = Variable(None)
        self._architecture, self._prediction_variable, regularization_cost = \
            mydnn._build_architecture_get_prediction_and_regularization_cost(
                architecture,
                weight_decay,
                self._x_variable
            )
        loss_class = loss_name_to_class[loss]
        self._is_classification = loss_class == CrossEntropy
        self._loss_variable = Add(loss_class(self._y_variable, self._prediction_variable), regularization_cost)

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None):
        number_of_samples = x_train.shape[0]

        assert x_train.shape[0] == number_of_samples
        assert y_train.shape[0] == number_of_samples

        history = list()

        for epoch_index in range(epochs):
            permutation = np.random.permutation(number_of_samples)
            # Other dimensions are left
            # TODO: should we avoid copying the data?
            x_train = x_train[permutation]
            y_train = y_train[permutation]

            seconds = time()
            train_loss_and_accuracy = self._do_epoch(x_train, y_train, batch_size, learning_rate)
            seconds = time() - seconds

            history_entry = {
                'epoch': 1 + epoch_index,
                'seconds': seconds,
            }

            # TODO: calculate average, we do need to include regularization here
            train_validation_loss_accuracy = [
                string.format(number)
                # Exploit the fact that length of zip is minimum
                for string, number in zip(['loss: {:.2f}', 'acc: {:.2f}'], train_loss_and_accuracy)
            ]

            history_entry.update(dict(zip(['train loss', 'train accuracy'], train_loss_and_accuracy)))

            if x_val is not None and y_val is not None:
                validation_loss_and_accuracy = self.evaluate(x_val, y_val)
                train_validation_loss_accuracy.extend([
                    string.format(number)
                    for string, number in zip(['val_loss: {:.2f}', 'val_acc: {:.2f}'], validation_loss_and_accuracy)
                ])
                history_entry.update(dict(zip(['validation loss', 'validation accuracy'], validation_loss_and_accuracy)))

            print(' - '.join(['Epoch {}/{}'.format(1 + epoch_index, epochs),  # TODO: is it one based?
                              '{:.2f} seconds'.format(seconds),] # TODO: how many digits after second
                             + train_validation_loss_accuracy
            ))

            history.append(history_entry)

        return history

    def predict(self, X, batch_size=None):
        """
        
        :param X: 
        :param batch_size: 
        :return: Returns number_of_samplesxnumber_of_classes numpy array output of last network layer 
        """
        number_of_samples = X.shape[0]

        if batch_size is None:
            batch_size = number_of_samples

        batch_index_to_y = list()

        for batch_offset in range(0, number_of_samples, batch_size):
            self._x_variable.set_value(X[batch_offset:batch_offset + batch_size])
            y_batch = self._prediction_variable.forward()
            batch_index_to_y.append(y_batch)

        y = np.concatenate(batch_index_to_y)

        assert y.shape[0] == number_of_samples

        return y

    def evaluate(self, X, y, batch_size=None):
        number_of_samples = X.shape[0]

        if batch_size is None:
            batch_size = number_of_samples

        total_loss = 0.0
        total_correctly_predicted = 0

        for batch_offset in range(0, number_of_samples, batch_size):
            actual_batch_size = min(batch_offset + batch_size, number_of_samples)
            self._x_variable.set_value(X[batch_offset:actual_batch_size])
            self._y_variable.set_value(y[batch_offset:actual_batch_size])
            loss = self._loss_variable.forward()  # TODO: should I include regularization
            total_loss += loss * actual_batch_size

            if self._is_classification:
                total_correctly_predicted += \
                    accuracy(y[batch_offset:actual_batch_size],
                             self._prediction_variable.get_value()) * actual_batch_size

        return_list = [total_loss / number_of_samples, ]

        if self._is_classification:
            computed_accuracy = 1.0 * total_correctly_predicted / number_of_samples
            return_list.append(computed_accuracy)

        return return_list

    @staticmethod
    def _build_architecture_get_prediction_and_regularization_cost(architecture, weight_decay, current_input):
        architecture_built = list()
        regularization_cost = Variable(0.0)
        weight_decay_variable = Variable(weight_decay)  # TODO: constant
        previous_layer_output = architecture[0]['input']

        for layer_dictionary in architecture:
            assert previous_layer_output == layer_dictionary["input"], \
                'Inconsistent architecture: can not feed {} outputs to {} inputs'.format(
                    previous_layer_output,
                    layer_dictionary['input']
                )
            activation_function = activation_function_name_to_class[layer_dictionary["nonlinear"]]
            regularization_method = regularization_method_name_to_class[layer_dictionary["regularization"]]
            layer = FullyConnectedLayer(layer_dictionary["input"], layer_dictionary["output"],
                                        activation_function,
                                        current_input)
            regularization_cost = Add(regularization_cost,
                                      Multiply(weight_decay_variable, regularization_method(layer.get_weight())))
            architecture_built.append(layer)
            current_input = layer
            previous_layer_output = layer_dictionary['output']

        return architecture_built, current_input, regularization_cost

    def _do_epoch(self, x_train, y_train, batch_size, learning_rate):
        number_of_samples = x_train.shape[0]

        if self._is_classification:
            total_loss_accuracy = np.zeros((2,))
        else:
            total_loss_accuracy = np.zeros((1,))

        for batch_offset in range(0, number_of_samples, batch_size):
            # Other dimensions are left
            actual_batch_size = min(number_of_samples - batch_offset, batch_size)
            x_batch = x_train[batch_offset:batch_offset + actual_batch_size]
            y_batch = y_train[batch_offset:batch_offset + actual_batch_size]

            total_loss_accuracy += np.array(self._do_iteration(x_batch, y_batch, learning_rate)) * actual_batch_size

        return total_loss_accuracy / number_of_samples

    def _do_iteration(self, x_batch, y_batch, learning_rate):
        self._x_variable.set_value(x_batch)
        self._y_variable.set_value(y_batch)

        mini_batch_loss_accuracy = list()
        mini_batch_loss = self._loss_variable.forward()
        mini_batch_loss_accuracy.append(mini_batch_loss)

        if self._is_classification:
            mini_batch_accuracy = accuracy(y_batch, self._prediction_variable.get_value())
            mini_batch_loss_accuracy.append(mini_batch_accuracy)

        self._loss_variable.backward()

        for current_layer in self._architecture:
            current_layer.update_grad(learning_rate)

        self._loss_variable.reset()

        return mini_batch_loss_accuracy

