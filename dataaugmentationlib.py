import math

import numpy as np

THETA_1 = math.pi/2
THETA_2 = math.pi
THETA_3 = (3*math.pi)/2

SIGMA_1 = 0.0005
SIGMA_2 = 0.001
SIGMA_3 = 0.002


def rotate(signals, labels):
    """

        Function for rotating a portion (proportional to scale_factor) of signals and adding the rotated signals to original
        signals list.

        Rotated signal matrix B is obtained by multiplication of T transformation matrix with A original signal matrix.

        B = TxA

        Args:
            signals: numpy.array of 2x128 matrix of I/Q signals.
            labels: labels for each signal.

        Returns:
            numpy.array of 2x128 matrix of I/Q signals concatenated with a portion (proportional to scale_factor) of signals
            rotated.

        """

    # T matrix is the multiplication matrix used for doing the transformation
    #
    # | cos(theta)  -sin(theta) |
    # | sin(theta)   cos(theta) |

    def T(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    T_1 = T(THETA_1)
    T_2 = T(THETA_2)
    T_3 = T(THETA_3)

    # list containing rotated signals
    rotated_signals = []
    rotated_labels = []

    # for each signal B = TxA
    for i in range(0, len(signals)):
        A = signals[i]
        l = labels[i]

        # rotate by THETA_1
        B = np.matmul(T_1, A)
        rotated_signals.append(B)
        rotated_labels.append(l)

        # rotate by THETA_2
        B = np.matmul(T_2, A)
        rotated_signals.append(B)
        rotated_labels.append(l)

        # rotate by THETA_3
        B = np.matmul(T_3, A)
        rotated_signals.append(B)
        rotated_labels.append(l)

    return rotated_signals, rotated_labels


def rotate_and_concatenate_with_signals(signals, labels):
    rotated_signals, rotated_labels = rotate(signals, labels)

    return np.concatenate((signals, rotated_signals)), np.concatenate((labels, rotated_labels))


def flip(signals, labels, direction):
    flipped = []

    for signal in signals:
        I = signal[0]
        Q = signal[1]

        if direction == "horizontal":
            flipped.append([-I, Q])

        if direction == "vertical":
            flipped.append([I, -Q])

    flipped = np.array(flipped)

    return flipped, labels


def horizontal_flip(signals, labels):
    return flip(signals, labels, "horizontal")


def horizontal_flip_and_concatenate_with_signals(signals, labels):
    hflipped_signals, hflipped_labels = horizontal_flip(signals, labels)

    return np.concatenate((signals, hflipped_signals)), np.concatenate((labels, hflipped_labels))


def vertical_flip(signals, labels):
    return flip(signals, labels, "vertical")


def vertical_flip_and_concatenate_with_signals(signals, labels):
    vflipped_signals, vflipped_labels = vertical_flip(signals, labels)

    return np.concatenate((signals, vflipped_signals)), np.concatenate((labels, vflipped_labels))


def horizontal_and_vertical_flip_and_concatenate_with_signals(signals, labels):
    hflipped_signals, hflipped_new_labels = horizontal_flip(signals, labels)
    vflipped_signals, vflipped_new_labels = vertical_flip(signals, labels)

    return np.concatenate((signals, hflipped_signals, vflipped_signals)),\
           np.concatenate((labels, hflipped_new_labels, vflipped_new_labels))


def add_gaussian_noise(signals, labels):
    disturbed_with_noise_signals = []
    disturbed_labels = []

    for i in range(0, len(signals)):
        signal = signals[i]
        label = labels[i]

        # noise with SIGMA_1
        noise = np.random.normal(0, SIGMA_1, (signal.shape[1:]))
        disturbed_with_noise_signals.append(signal + noise)
        disturbed_labels.append(label)

        # noise with SIGMA_2
        noise = np.random.normal(0, SIGMA_2, (signal.shape[1:]))
        disturbed_with_noise_signals.append(signal + noise)
        disturbed_labels.append(label)

        # noise with SIGMA_3
        noise = np.random.normal(0, SIGMA_3, (signal.shape[1:]))
        disturbed_with_noise_signals.append(signal + noise)
        disturbed_labels.append(label)

    disturbed_with_noise_signals = np.array(disturbed_with_noise_signals)

    return disturbed_with_noise_signals, disturbed_labels


def add_gaussian_noise_and_concatenate_with_signals(signals, labels):
    gnoised_signals, gnoised_labels = add_gaussian_noise(signals, labels)

    return np.concatenate((signals, gnoised_signals)), np.concatenate((labels, gnoised_labels))


def rotate_flip_add_gaussian_noise(signals, labels):
    rotated_signals, rotated_new_labels = rotate(signals, labels)
    hflipped_signals, hflipped_new_labels = horizontal_flip(signals, labels)
    vflipped_signals, vflipped_new_labels = vertical_flip(signals, labels)
    gnoised_signals, gnoised_new_labels = add_gaussian_noise(signals, labels)

    signals_result, labels_result = np.concatenate((rotated_signals, hflipped_signals)), np.concatenate(
        (rotated_new_labels, hflipped_new_labels))
    signals_result, labels_result = np.concatenate((signals_result, vflipped_signals)), np.concatenate(
        (labels_result, vflipped_new_labels))
    signals_result, labels_result = np.concatenate((signals_result, gnoised_signals)), np.concatenate(
        (labels_result, gnoised_new_labels))

    return signals_result, labels_result


def rotate_flip_add_gaussian_noise_and_concatenate_with_signals(signals, labels):
    signals_result, labels_result = rotate_flip_add_gaussian_noise(signals, labels)

    return np.concatenate((signals, signals_result)), np.concatenate((labels, labels_result))
