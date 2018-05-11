import numpy as np
import matplotlib.pyplot as plt


def plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=None):
    if title is None:
        title='{} epochs with batch size {}'.format(len(history), batch_size)

    is_classification = 'train accuracy' in history[0]
    has_validation = 'validation loss' in history[0]
    number_of_epochs = len(history)
    number_of_iterations_per_epoch = (number_of_samples + batch_size - 1) // batch_size
    epoch_to_number_of_iterations_cumulative = np.arange(1, 1 + number_of_epochs) * number_of_iterations_per_epoch

    def plot_entry(name):
        epoch_to_value = [history_entry[name] for history_entry in history]
        handle, = plt.plot(epoch_to_number_of_iterations_cumulative, epoch_to_value, label=name)

        return handle

    plt.figure()
    train_loss_handle = plot_entry('train loss')
    handles = [train_loss_handle,]

    if has_validation:
        validation_loss_handle = plot_entry('validation loss')
        handles.append(validation_loss_handle)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)

    if is_classification:
        plt.figure()
        train_accuracy_handle = plot_entry('train accuracy')
        handles.append(train_accuracy_handle)

        if has_validation:
            validation_accuracy_handle = plot_entry('validation accuracy')
            handles.append(validation_accuracy_handle)

        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(title)

    plt.legend(handles=handles)
    plt.show()