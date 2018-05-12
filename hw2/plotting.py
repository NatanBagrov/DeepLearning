import numpy as np
import matplotlib.pyplot as plt
import os

def plot_iteration_to_loss_accuracy_from_history(history, number_of_samples, batch_size, title=None,
                                                 show=False, close_figures=True, save_directory='graphs'):
    if title is None:
        title = '{} epochs with batch size {}'.format(len(history), batch_size)

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

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
    loss_handles = [train_loss_handle, ]

    if has_validation:
        validation_loss_handle = plot_entry('validation loss')
        loss_handles.append(validation_loss_handle)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(handles=loss_handles)
    plt.title(title)
    plt.savefig(os.path.join(save_directory, '{} loss.png'.format(title)))

    if close_figures:
        plt.close()

    if is_classification:
        plt.figure()
        train_accuracy_handle = plot_entry('train accuracy')
        accuracy_handles = [train_accuracy_handle, ]

        if has_validation:
            validation_accuracy_handle = plot_entry('validation accuracy')
            accuracy_handles.append(validation_accuracy_handle)

        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend(handles=accuracy_handles, loc='upper left')
        plt.savefig(os.path.join(save_directory, '{} accuracy.png'.format(title)))
        plt.close()

    if show:
        plt.show()
