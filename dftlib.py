import numpy as np
import datasetlib

DFT_SIGNALS_COLUMN_DATAFRAME_NAME = 'DFT signals'


def single_I_Q_to_DFT(signal):
    complex_IQ_representation = []

    # a signal is
    I = signal[0]
    Q = signal[1]

    # for each <i,q> couple
    for j in range(0, len(I)):
        cmplx = complex(I[j], Q[j])  # <i,q> can be represented as a complex number (i = real part, q = complex part)
        complex_IQ_representation.append(cmplx)

    # numpy fft(arr) transforms an array 'arr' of complex numbers using Discrete Fourier Transform
    ffted_cplx = np.fft.fft(complex_IQ_representation)

    return [ffted_cplx.real, ffted_cplx.imag]


def all_I_Q_to_DFT(signals):
    transformed_signals = []

    for signal in signals:
        transformed_signal = single_I_Q_to_DFT(signal)

        # transformed_signal is a train of values
        transformed_signals.append(transformed_signal)

    return transformed_signals


def transform_and_add_I_Q_to_DFT(dataset_df):
    # add DFT signals to pandas Dataframe
    signals = datasetlib.signals(dataset_df)
    dataset_df[DFT_SIGNALS_COLUMN_DATAFRAME_NAME] = all_I_Q_to_DFT(signals)

    return dataset_df


def signals(dataset_df):
    return dataset_df[DFT_SIGNALS_COLUMN_DATAFRAME_NAME].tolist()
