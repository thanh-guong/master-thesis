import numpy as np


def randomize_elements_to_transform(signals, labels, increment_percentage=1):
    signals_length = len(signals)
    number_of_signals_to_transform = int(signals_length * increment_percentage)

    signals_to_transform_index = np.random.choice(range(0, signals_length), size=number_of_signals_to_transform,
                                                  replace=True)

    return signals[signals_to_transform_index], labels[signals_to_transform_index]


def rotate(signals, labels, theta=0, increment_percentage=1):
    """

        Function for rotating a portion (proportional to scale_factor) of signals and adding the rotated signals to original
        signals list.

        Rotated signal matrix B is obtained by multiplication of T transformation matrix with A original signal matrix.

        B = TxA

        Args:
            signals: numpy.array of 2x128 matrix of I/Q signals.
            labels: labels for each signal.
            theta: angle of rotation in radian (using pi notation).
            increment_percentage: how many signals to rotate and add to original signals.

        Returns:
            numpy.array of 2x128 matrix of I/Q signals concatenated with a portion (proportional to scale_factor) of signals
            rotated.

        """

    # T matrix is the multiplication matrix used for doing the transformation
    #
    # | cos(theta)  -sin(theta) |
    # | sin(theta)   cos(theta) |

    T = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # list containing rotated signals
    B_list = []

    A_list, new_labels = randomize_elements_to_transform(signals, labels, increment_percentage)

    # for each signal, represented by A matrix
    for A in A_list:
        # B = TxA
        B = np.matmul(T, A)
        B_list.append(B)

    B_list = np.array(B_list)

    return B_list, new_labels


def rotate_and_concatenate_with_signals(signals, labels, theta=0, increment_percentage=1):
    rotated_signals, rotated_labels = rotate(signals, labels, theta, increment_percentage)

    return np.concatenate((signals, rotated_signals)), np.concatenate((labels, rotated_labels))


def flip(signals, labels, direction, increment_percentage=1):
    flipped = []

    signals_to_flip, new_labels = randomize_elements_to_transform(signals, labels, increment_percentage)

    for signal in signals_to_flip:
        I = signal[0]
        Q = signal[1]

        if direction == "horizontal":
            flipped.append([-I, Q])

        if direction == "vertical":
            flipped.append([I, -Q])

    flipped = np.array(flipped)

    return flipped, new_labels


def horizontal_flip(signals, labels, increment_percentage=1):
    return flip(signals, labels, "horizontal", increment_percentage)


def horizontal_flip_and_concatenate_with_signals(signals, labels, increment_percentage=1):
    hflipped_signals, hflipped_labels = horizontal_flip(signals, labels, increment_percentage)

    return np.concatenate((signals, hflipped_signals)), np.concatenate((labels, hflipped_labels))


def vertical_flip(signals, labels, increment_percentage=1):
    return flip(signals, labels, "vertical", increment_percentage)


def vertical_flip_and_concatenate_with_signals(signals, labels, increment_percentage=1):
    vflipped_signals, vflipped_labels = vertical_flip(signals, labels, increment_percentage)

    return np.concatenate((signals, vflipped_signals)), np.concatenate((labels, vflipped_labels))


def add_gaussian_noise(signals, labels, standard_deviation=0, increment_percentage=1):
    disturbed_with_noise_signals = []

    signals_to_disturb_with_noise, new_labels = randomize_elements_to_transform(signals, labels, increment_percentage)

    for signal in signals_to_disturb_with_noise:
        noise = np.random.normal(0, standard_deviation, (signal.shape[1:]))
        disturbed_with_noise_signals.append(signal + noise)

    disturbed_with_noise_signals = np.array(disturbed_with_noise_signals)

    return disturbed_with_noise_signals, new_labels


def add_gaussian_noise_and_concatenate_with_signals(signals, labels, standard_deviation=0, increment_percentage=1):
    gnoised_signals, gnoised_labels = add_gaussian_noise(signals, labels, standard_deviation, increment_percentage)

    return np.concatenate((signals, gnoised_signals)), np.concatenate((labels, gnoised_labels))


def rotate_flip_add_gaussian_noise_and_concatenate_with_signals(signals, labels, standard_deviation=0, theta=0,
                                                                increment_percentage=1):
    increment_percentage = 0.25 * increment_percentage

    rotated_signals, rotated_new_labels = rotate(signals, labels, theta, increment_percentage)
    hflipped_signals, hflipped_new_labels = horizontal_flip(signals, labels, increment_percentage)
    vflipped_signals, vflipped_new_labels = vertical_flip(signals, labels, increment_percentage)
    gnoised_signals, gnoised_new_labels = add_gaussian_noise(signals, labels, standard_deviation, increment_percentage)

    signals_result, labels_result = np.concatenate((signals, rotated_signals)), np.concatenate(
        (labels, rotated_new_labels))
    signals_result, labels_result = np.concatenate((signals_result, hflipped_signals)), np.concatenate(
        (labels_result, hflipped_new_labels))
    signals_result, labels_result = np.concatenate((signals_result, vflipped_signals)), np.concatenate(
        (labels_result, vflipped_new_labels))
    signals_result, labels_result = np.concatenate((signals_result, gnoised_signals)), np.concatenate(
        (labels_result, gnoised_new_labels))

    return signals_result, labels_result
