import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datasetlib


def show_loss_curves(history):
    """
        Shows loss curves.

        Parameters:
            history (History): keras History from previous training using keras.Model.fit().
    """

    # Show loss curves
    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss + error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()


def confusion_matrix(cm, title='', cmap=plt.cm.Blues, labels=[]):
    """
        Plot confusion matrix using matplotlib.

        Parameters:
            cm (xxx): confusion matrix.
            title (string): title for the box.
            cmap (?): colormap for matplotlib.
            labels (List): labels for axis.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(model, title, x_test, y_test, batch_size, classes):
    """
        Makes a prediction, then builds the confusion matrix and plots it.

        Parameters:
            model (Model): keras.Model used for prediction.
            x_test (List): training set data.
            y_test (List): training set labels.
            batch_size (int): batch dimension.
            classes (List): labels for axis.
    """

    # Plot confusion matrix
    test_y_hat = model.predict(x_test, batch_size=batch_size)

    confusion_m = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])

    for i in range(0, x_test.shape[0]):
        j = list(y_test[i, :]).index(1)
        k = int(np.argmax(test_y_hat[i, :]))
        confusion_m[j, k] = confusion_m[j, k] + 1

    for i in range(0, len(classes)):
        confnorm[i, :] = confusion_m[i, :] / np.sum(confusion_m[i, :])

    confusion_matrix(confnorm, title, labels=classes)


def plot_double_input_confusion_matrix(model, title, iq_test, transformed_test, y_test, batch_size, classes):
    """
        Makes a prediction, then builds the confusion matrix and plots it.

        Parameters:
            model (Model): keras.Model used for prediction.
            y_test (List): training set labels.
            batch_size (int): batch dimension.
            classes (List): labels for axis.
    """

    # Plot confusion matrix
    test_y_hat = model.predict((iq_test, transformed_test), batch_size=batch_size)

    confusion_m = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])

    for i in range(0, iq_test.shape[0]):
        j = list(y_test[i, :]).index(1)
        k = int(np.argmax(test_y_hat[i, :]))
        confusion_m[j, k] = confusion_m[j, k] + 1

    for i in range(0, len(classes)):
        confnorm[i, :] = confusion_m[i, :] / np.sum(confusion_m[i, :])

    confusion_matrix(confnorm, title, labels=classes)


def plot_confusion_matrix_each_snr(model, neural_network_name, snrs, dataset_df, X_test, Y_test, test_index, classes):
    # Plot confusion matrix
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        all_snrs = datasetlib.snrs(dataset_df)
        all_snrs = np.array(all_snrs)

        test_SNRs = list(all_snrs[test_index])
        this_snr_indexes = np.where(np.array(test_SNRs) == snr)

        test_X_i = X_test[this_snr_indexes]
        test_Y_i = Y_test[this_snr_indexes]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])

        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1

        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

        plt.figure()
        confusion_matrix(confnorm, labels=classes, title=neural_network_name + " (SNR=%d)" % (snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)

    return acc


def plot_double_input_confusion_matrix_each_snr(model, neural_network_name, snrs, dataset_df, iq_test, transformed_test, Y_test, test_index, classes):
    # Plot confusion matrix
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        all_snrs = datasetlib.snrs(dataset_df)
        all_snrs = np.array(all_snrs)

        test_SNRs = list(all_snrs[test_index])
        this_snr_indexes = np.where(np.array(test_SNRs) == snr)

        test_X_i = (iq_test[this_snr_indexes], transformed_test[this_snr_indexes])
        test_Y_i = Y_test[this_snr_indexes]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])

        for i in range(0, len(test_X_i[0])):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1

        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

        plt.figure()
        confusion_matrix(confnorm, labels=classes, title=neural_network_name + " (SNR=%d)" % (snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)

    return acc


def accuracy_dataframe(acc):
    """
        Makes a prediction, then builds the confusion matrix and plots it.

        Parameters:
            acc (Matrix): first column is SNR, second column is accuracy.
        Returns:
            .* (DataFrame): pandas.DataFrame containing SNR and accuracy.
    """

    accuracy_perc = {}

    for el in acc.items():
        accuracy_perc[el[0]] = int(el[1] * 100)

    return pd.DataFrame(data=accuracy_perc, index=["Accuracy %"])


def accuracy_curve(snrs, acc, neural_network_name):
    """
        Makes a prediction, then builds the confusion matrix and plots it.

        Parameters:
            snrs (List): unique list of SNRs.
            acc (Matrix): first column is SNR, second column is accuracy.
            neural_network_name (string): name of the used neural network.
    """

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(neural_network_name)
