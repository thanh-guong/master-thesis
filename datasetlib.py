import pickle
import pandas as pd

IQ_SIGNALS_COLUMN_DATAFRAME_NAME = 'IQ signals'
MODULATION_LABEL_COLUMN_DATAFRAME_NAME = 'Modulation_Label'
SNR_COLUMN_DATAFRAME_NAME = 'SNR'


def load_dataset(dataset_filename):
    """
        Opens the dataset file, and returns it as a list.

                Parameters:
                        dataset_filename (string): relative/absolute path to RML2016.10a_dict.pkl dataset.

                Returns:
                        dataset (List): list of tuples (signal, modulation, snr) with data from dataset.
    """

    dataset = []

    with (open(dataset_filename, "rb")) as dataset_file:
        data = dataset_file.read()
        data_dict = pickle.loads(data, encoding='bytes')  # unpickle data
        keys = data_dict.keys()

        # for each key in dataset keys
        for key in keys:
            # extract modulation label and snr
            modulation, snr = key[0].decode("utf-8"), key[1]

            # for each I/Q signal couple sample
            for signal in data_dict[key]:
                # save the tuple (signal, modulation_label, snr) in the list
                signal_tuple = (signal, modulation, snr)
                dataset.append(signal_tuple)

    return dataset


def load_dataset_dataframe(dataset_filename, dataset=None):
    """
        Opens the dataset file, and returns it as a pandas DataFrame.

                Parameters:
                        dataset_filename (string): relative/absolute path to RML2016.10a_dict.pkl dataset.
                        dataset (List): list of tuples (signal, modulation, snr) with data from dataset.
                Returns:
                        dataset (DataFrame): pandas DataFrame containing dataset.
    """

    if dataset is None:
        dataset = load_dataset(dataset_filename)

    dataset_dataframe = pd.DataFrame(data=dataset)

    # pandas aesthetics
    dataset_dataframe.columns = [
        IQ_SIGNALS_COLUMN_DATAFRAME_NAME,
        MODULATION_LABEL_COLUMN_DATAFRAME_NAME,
        SNR_COLUMN_DATAFRAME_NAME
    ]

    return dataset_dataframe


def filter_dataset_for_high_snr_only(dataset_df, snr_lower_bound):
    return dataset_df[dataset_df[SNR_COLUMN_DATAFRAME_NAME] >= snr_lower_bound]


def signals(dataset_dataframe):
    """
        All the elements of I/Q signal column in dataset, as a List.

                Parameters:
                        dataset_dataframe (DataFrame): pandas DataFrame containing dataset.
                Returns:
                        .* (List): I/Q signals.
    """

    return dataset_dataframe[IQ_SIGNALS_COLUMN_DATAFRAME_NAME].tolist()


def labels(dataset_dataframe):
    """
        All the elements of Label column in dataset, as a List.

                Parameters:
                        dataset_dataframe (DataFrame): pandas DataFrame containing dataset.
                Returns:
                        .* (List): Labels.
    """

    return dataset_dataframe[MODULATION_LABEL_COLUMN_DATAFRAME_NAME].tolist()


def snrs(dataset_dataframe):
    """
        All the elements of SNR column in dataset, as a List.

                Parameters:
                        dataset_dataframe (DataFrame): pandas DataFrame containing dataset.
                Returns:
                        .* (List): SNRs.
    """

    return dataset_dataframe[SNR_COLUMN_DATAFRAME_NAME].tolist()


def mods(dataset_dataframe):
    """
        Unique elements of Label column in dataset, as a List.

                Parameters:
                        dataset_dataframe (DataFrame): pandas DataFrame containing dataset.
                Returns:
                        result (List): Labels.
    """

    result = dataset_dataframe[MODULATION_LABEL_COLUMN_DATAFRAME_NAME].unique().tolist()
    result.sort()

    return result


def unique_snrs(dataset_dataframe):
    """
        Unique elements of SNR column in dataset, as a List.

                Parameters:
                        dataset_dataframe (DataFrame): pandas DataFrame containing dataset.
                Returns:
                        result (List): SNRs.
    """

    result = dataset_dataframe[SNR_COLUMN_DATAFRAME_NAME].unique().tolist()
    result.sort()

    return result
