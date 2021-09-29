import numpy as np

import datasetlib

# seeding the random numpy object (Interstella 5555 - Daft Punk)
np.random.seed(5555)


def train_test_random_indexes(n_elements, training_set_percentage):
    """
        Builds a random index for train and test set.

                Parameters:
                        n_elements (int): number of datapoints in the dataset.
                        training_set_percentage (double): percentage of test set dimension.
                Returns:
                        train_index, test_index (List, List): These are the indexes to extract random datapoints from
                        dataset, respecting the specified training set dimension.
    """

    # take a portion of the dataset as training set
    n_train = int(n_elements * training_set_percentage)

    # extract datapoints indexes randomly
    # train_index = [I_1, I_2, ..., I_N_TRAIN]
    train_index = np.random.choice(range(0, n_elements), size=n_train, replace=False)

    # obtain the remaining datapoints indexes
    # test_index = [I_1, I_2, ..., I_HALF_N_EXAMPLES]
    test_index = list(set(range(0, n_elements)) - set(train_index))

    return train_index, test_index


def train_test_random_indexes_high_snr(dataset_df, training_set_percentage, snr_lower_bound):

    high_snr_dataset_df = datasetlib.filter_dataset_for_high_snr_only(dataset_df, snr_lower_bound)
    high_snr_signals = datasetlib.signals(high_snr_dataset_df)
    total_high_snr_datapoints = len(high_snr_signals)

    total_train_index = high_snr_dataset_df.index.values.tolist()

    # take a portion of the dataset as training set
    n_train = int(total_high_snr_datapoints * training_set_percentage)

    # extract datapoints indexes randomly
    # train_index = [I_1, I_2, ..., I_N_TRAIN]
    train_index = np.random.choice(total_train_index, size=n_train, replace=False)

    # obtain the remaining datapoints indexes
    # test_index = [I_1, I_2, ..., I_HALF_N_EXAMPLES]
    test_index = list(set(range(0, len(dataset_df))) - set(train_index))

    return train_index, test_index


def split_x_train_test(signals, train_index, test_index):
    """
            Uses train_index and test_index to extract datapoints from the I/Q signal column, and build the Training Set
            and Test Set.

                    Parameters:
                            signals (List): datapoints from the signals column in dataset.
                            train_index (List): index for Training Set.
                            test_index (List): index for Test Set.
                    Returns:
                            x_train, x_test (List, List): These two lists contain Training Set and Test Set.
    """

    signals = np.array(signals)

    x_train = signals[train_index]
    x_test = signals[test_index]

    return x_train, x_test


def split_y_train_test(labels, mods, train_index, test_index):
    """
            Uses train_index and test_index to extract datapoints from the I/Q signal column, and build the Training Set
            and Test Set.
            Labels are one-hot encoded.

                    Parameters:
                            labels (List): labels from the label column in dataset.
                            mods (List): ordered list of unique labels.
                            train_index (List): index for Training Set.
                            test_index (List): index for Test Set.
                    Returns:
                            y_train, y_test (List, List): These two lists contain Training Set and Test Set one-hot
                            encoded.
    """

    def to_onehot(x):
        input_list = list(x)
        vectors_number = len(list(input_list))
        number_of_elements_for_each_vector = max(input_list, default=0) + 1

        # one hot encoding is a vector of zeros, and only a 1 that identifies the class
        # producing <vectors_number> vectors of <number_of_elements_for_each_vector> elements
        result = np.zeros([vectors_number, number_of_elements_for_each_vector])

        # placing the 1 in the correct place
        for i in range(0, vectors_number):
            result[i][input_list[i]] = 1

        return result

    y_train = to_onehot(map(lambda x: mods.index(labels[x]), train_index))
    y_test = to_onehot(map(lambda x: mods.index(labels[x]), test_index))

    return y_train, y_test


def split_x_y_train_test(signals, labels, mods, train_index, test_index):
    """
            Uses train_index and test_index to extract datapoints from the I/Q signal column, and build the Training Set
            and Test Set.
            Labels are one-hot encoded.

                    Parameters:
                            signals (List): datapoints from the signals column in dataset.
                            mods (List): list of unique labels.
                            labels (List): labels from the label column in dataset.
                            train_index (List): index for Training Set.
                            test_index (List): index for Test Set.
                    Returns:
                            x_train, x_test (List, List): These two lists contain Training Set and Test Set.
    """

    x_train, x_test = split_x_train_test(signals, train_index, test_index)
    y_train, y_test = split_y_train_test(labels, mods, train_index, test_index)

    return x_train, x_test, y_train, y_test
