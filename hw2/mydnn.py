import numpy as np
from timeit import timeit
from utils.LossFunctions import CrossEntropy
from FullyConnectedLayer import FullyConnectedLayer
from graph.Variable import Variable


class mydnn():
    def __init__(self, architecture, loss, weight_decay=0):
        self.prepare_dictionaries()  # TODO: this should create the string->func dict of
                                    # activation/regularization/loss from the utils

        self._x_variable = Variable(None) # TODO: call it placeholder
        self._y_variable = Variable(None)
        self._architecture, self._prediction_variable = \
            mydnn._build_architecture(architecture, weight_decay, self._x_variable)
        self._loss = loss_name_to_class[loss]
        self._loss_variable = self._loss(self._prediction_variable, self._y_variable)

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None):
        number_of_samples = x_train.shape[0]

        assert x_train.shape[0] == number_of_samples
        assert y_train.shape[0] == number_of_samples

        for epoch_index in range(epochs):
            permutation = np.random.permutation(number_of_samples)
            # Other dimensions are left
            # TODO: should we avoid copying the data?
            x_train = x_train[permutation]
            y_train = y_train[permutation]

            seconds = timeit(lambda : self._do_epoch(x_train, y_train, batch_size, learning_rate), number=1)

            train_loss_and_accuracy = self.evaluate(x_train, y_train)
            train_validation_loss_accuracy = [
                string.format(number)
                # Exploit the fact that length of zip is minimum
                for string, number in zip(['loss: {:.1f}', 'acc: {:.1f}'], train_loss_and_accuracy)
            ]

            if x_val is not None and y_val is not None:
                validation_loss_and_accuracy = self.evaluate(x_train, y_train)
                train_validation_loss_accuracy.extend([
                    string.format(number)
                    for string, number in zip(['val_loss: {:.1f}', 'val_acc: {:.1f}'], validation_loss_and_accuracy)
                ])

            print(' - '.join(['Epoch {}/{}'.format(1 + epoch_index, epochs),  # TODO: is it one based?
                              '{} seconds'.format(seconds),]
                             + train_validation_loss_accuracy
            ))

    def predict(self, X, batch_size=None):
        # TODO: forward
        # TODO: should we make argmax in classification? (if yes fix evaluate)
        pass

    def evaluate(self, X, y, batch_size=None):
        prediction = self.predict(X, batch_size=batch_size)
        loss = self._loss(prediction, y)
        return_list = [loss, ]

        if self._is_classification():
            accuracy = 0.0 #
            return_list.append(accuracy)

        return return_list

    def _is_classification(self):
        return isinstance(self._loss, CrossEntropy)

    @staticmethod
    def _build_architecture(architecture, weight_decay, current_input):
        architecture_built = list()

        for layer_dictionary in architecture:
            activation_function = activation_function_name_to_class[layer_dictionary["nonlinear"]]
            regularization_method = regularization_method_to_class[layer_dictionary["regularization"]]
            layer = FullyConnectedLayer(layer_dictionary["input"], layer_dictionary["output"],
                                        activation_function,
                                        regularization_method, weight_decay,
                                        current_input)

            architecture_built.append(layer)
            current_input = layer

        return architecture_built, current_input

    def _do_epoch(self, x_train, y_train, batch_size, learning_rate):
        number_of_samples = x_train.shape[0]

        for batch_offset in range(0, number_of_samples, batch_size):
            # Python does not mind positive overflows
            # Other dimensions are left
            x_batch = x_train[batch_offset:batch_offset+batch_size]
            y_batch = y_train[batch_offset:batch_offset+batch_size]

            self._do_iteration(x_batch, y_batch, learning_rate)

    def _do_iteration(self, x_batch, y_batch, learning_rate):
        self._x_variable.set_value(x_batch)
        self._y_variable.set_value(y_batch)

        self._architecture[0].forward()

        for current_layer in self._architecture:
            current_layer.update_grad(learning_rate)

