import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from transfer_learning.cifar100vgg import cifar100vgg


def normalize(x):
    mean = 121.936
    std = 68.389
    return (x - mean) / (std + 1e-7)


def visualize_image(sample, probabilities, X, y):
    top_5_labels = probabilities.argsort()[-5:][::-1]
    top_5_probs = probabilities[top_5_labels]
    top_5_cifar100_samples = list()
    for label in top_5_labels:
        idxs = np.where(y == label)[0]
        top_5_cifar100_samples.append(X[idxs[0]])
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("CIFAR-10")
    ax1.imshow(sample)
    for i, sample in enumerate(top_5_cifar100_samples):
        ax = fig.add_subplot(2, 3, i+2)
        ax.set_title("{} ({:.2f})". format(top_5_labels[i], top_5_probs[i]))
        ax.imshow(sample)
    fig.show()
    fig.clf()
    fig.clear()


def bayesian_vgg(pretrained_vgg_model, num_visualizations=0):
    (x_100, y_100), (x_test_100, y_test_100) = cifar100.load_data()
    new_num_labels = 10
    old_num_labels = 100
    (orig_x_train, orig_y_train), (x_test, y_test) = cifar10.load_data()
    test_size = y_test.size
    print("The test size: %d" % test_size)
    test_probabilities = pretrained_vgg_model.predict(normalize(x_test), verbose=1, batch_size=256)
    accuracies = list()
    for idx, num_samples in enumerate([100, 1000, 10000]):
        x_train, _, y_train, _ = train_test_split(orig_x_train, orig_y_train,
                                                  train_size=num_samples, random_state=42, stratify=orig_y_train)

        not_normalized_x_train = np.copy(x_train)
        x_train = normalize(x_train)

        train_probabilities = pretrained_vgg_model.predict(x_train, verbose=1, batch_size=256)

        for i in range(num_visualizations):
            label_i_idx = np.where(y_train == i)[0]
            visualize_image(not_normalized_x_train[label_i_idx[0]], train_probabilities[label_i_idx[0]], x_100, y_100)

        p_x_given_ck = np.array([
            np.average(train_probabilities[np.where(y_train == i)[0], :], axis=0) for i in range(new_num_labels)
        ])
        # A sanity check
        np.testing.assert_allclose(np.sum(p_x_given_ck, axis=1), np.ones(new_num_labels), atol=1e-4)

        y_pred = np.array([np.argmax(p_x_given_ck[:, np.argmax(test_probabilities[i])]) for i in range(test_size)])
        test_score = accuracy_score(y_test, y_pred)
        y_pred = np.array([np.argmax(p_x_given_ck[:, np.argmax(train_probabilities[i])]) for i in range(num_samples)])
        train_score = accuracy_score(y_train, y_pred)
        accuracies.append((test_score, train_score))
        print("P(Ck|X) = ACCURACY IS {} over {} samples of test"
              .format(test_score, test_size))
        print("P(Ck|X) = ACCURACY IS {} over {} samples of train"
              .format(train_score, num_samples))

        # A sanity to confirm that we do better there Naive Bayes
        # train_one_hot_features = np.eye(old_num_labels)[np.argmax(train_probabilities, axis=1)]
        # test_one_hot_features = np.eye(old_num_labels)[np.argmax(test_probabilities, axis=1)]
        # model = MultinomialNB().fit(train_one_hot_features, y_train)
        # y_pred = model.predict(test_one_hot_features)
        # print(accuracy_score(y_test, y_pred))

    return accuracies


if __name__ == '__main__':
    cifar_100_vgg = cifar100vgg(train=False)
    sample_accuracies = bayesian_vgg(cifar_100_vgg.model, num_visualizations=0)
    print(sample_accuracies)
