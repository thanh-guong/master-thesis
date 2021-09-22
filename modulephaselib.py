import cmath
import datasetlib

MODULE_PHASE_SIGNALS_COLUMN_DATAFRAME_NAME = 'MP signals'


def single_I_Q_to_module_phase(signal):
    modules = []
    phases = []

    # a signal is
    I = signal[0]
    Q = signal[1]

    # for each <i,q> couple
    for j in range(0, len(I)):
        cmplx = complex(I[j], Q[j])  # <i,q> can be represented as a complex number (i = real part, q = complex part)

        modules.append(abs(cmplx))
        phases.append(cmath.phase(cmplx))

    # return [module_1, module_2, ..., module_n], [phase_1, phase_2, ..., phase_n]
    return modules, phases


def all_I_Q_to_module_phase(signals):
    transformed_signals = []

    for signal in signals:
        transformed_signal = single_I_Q_to_module_phase(signal)

        # transformed_signal is a <modules, phases> couple
        transformed_signals.append(transformed_signal)

    return transformed_signals


def transform_and_add_signals_to_dataframe(dataset_df):
    # add Module/Phase signals to pandas Dataframe
    signals = datasetlib.signals(dataset_df)
    dataset_df[MODULE_PHASE_SIGNALS_COLUMN_DATAFRAME_NAME] = all_I_Q_to_module_phase(signals)

    return dataset_df


def signals(dataset_df):
    return dataset_df[MODULE_PHASE_SIGNALS_COLUMN_DATAFRAME_NAME].tolist()
