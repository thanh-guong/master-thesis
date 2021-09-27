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

    return np.concatenate((signals, B_list)), np.concatenate((labels, new_labels))


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

    return np.concatenate((signals, flipped)), np.concatenate((labels, new_labels))


def horizontal_flip(signals, labels, increment_percentage=1):
    return flip(signals, labels, "horizontal", increment_percentage)


def vertical_flip(signals, labels, increment_percentage=1):
    return flip(signals, labels, "vertical", increment_percentage)


def add_gaussian_noise(signals, labels, standard_deviation=0, increment_percentage=1):
    disturbed_with_noise_signals = []

    signals_to_disturb_with_noise, new_labels = randomize_elements_to_transform(signals, labels, increment_percentage)

    for signal in signals_to_disturb_with_noise:
        noise = np.random.normal(0, standard_deviation, (signal.shape[1:]))
        disturbed_with_noise_signals.append(signal + noise)

    disturbed_with_noise_signals = np.array(disturbed_with_noise_signals)

    return np.concatenate((signals, disturbed_with_noise_signals)), np.concatenate((labels, new_labels))
