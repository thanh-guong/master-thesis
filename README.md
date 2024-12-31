# Master's Degree Thesis

This is my Master's Degree thesis repository. It's about Signal Modulation Classification using Deep Learning.

Inbound signals are encoded using different kind of modulations, so it's mandatory to know (or predict) the modulation used in order to decode the signal content. AMR (Automatic Modulation Recognition) can be used to predict the modulation of an inbound signal. Deep Learning tecniques can be used for AMR, and that's what I did with my Master Degree thesis.

# Paper: An Effective Convolution Neural Network for Automatic Recognition of Analog and Digital Signal Modulations for Cognitive SDR Applications

I collaborated at writing a paper (with Luca Pallotta and Gaetano Giunta from Roma Tre University in Rome) which resumes all the work done in this repository. In that work, architecture shown and explained is called FreeHand, but it's corresponding to the FreeHand v3 implemented in this repository. The paper is available at https://doi.org/10.1109/IWSDA50346.2022.9870449.

# Project structure description

The project contains different neural network trainings. Each **directory** contains the trainings for a different **data version**. Inside each directory there are the **.ipynb** files which are the **neural network trainings** for the data version specified by the directory where they are. The schema below describes an example of the directory tree structure used in this project.

	.
	├── ...
	│
	├── DatasetTransformation 1
	│	│
	│	├── neural_network_1_training.ipynb
	│	│ ...
	│	└── neural_network_M_training.ipynb
	│
	├── ...
	│
	├── DatasetTransformation N
	│	│
	│	├── neural_network_1_training.ipynb
	│	│ ...
	│	└── neural_network_M_training.ipynb
	│
	└── ...

## Directories

The project contains different neural network trainings. Inside the **IQ** directory there are the ones with the **raw dataset**, the other ones are using a **pre-processing of the dataset** before the training. All the directories content is shown in the table below.

| Directory              | Content description                                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| DFT + IQ               | Datapoints transformed using DFT (Discrete Fourier Transform) concatenated with raw datapoints                          |
| DFT                    | Datapoints transformed using DFT (Discrete Fourier Transform)                                                           |
| IQ - Data Augmentation | Raw dataset enlarged using data augmentation tecniques                                                                  |
| IQ                     | Raw dataset                                                                                                             |
| MP + IQ                | Datapoints transformed in Module/Phase (signals can be represented as complex numbers) concatenated with raw datapoints |
| MP                     | Datapoints transformed in Module/Phase (signals can be represented as complex numbers)                                  |

## Libraries

I wrote some libraries to employ in the neural network trainings I did. A short description of those libraries content is shown in the list below:

- _dataaugmentationlib.py_ enlarging the dataset using data augmentation tecniques
- _datasetlib.py_ reading the dataset
- _dftlib.py_ transform the dataset using DFT (Discrete Fourier Transform)
- _evaluationlib.py_ evaluating models after training
- _modulephaselib.py_ transforming the datapoints in Module/Phase
- _neural_networks.py_ neural network implementations
- _trainlib.py_ training neural networks
- _traintestsplitlib.py_ split the dataset in training set and test set

# Prerequisites

Follow those steps to get all running.

## Dataset

Dataset can be downloaded from [here](https://www.deepsig.ai/datasets). Extract the dataset in a directory named _data/_. Dataset file should has to be named _RML2016.10a_dict.pkl_.

## Python and pip

You need to have Python (at least 3.9.x) and pip (included with Python) installed on your system.

## Dependencies provisioning

You need to run `pip install -r requirements.txt` to install this project dependencies.

## CUDA

Check if your GPU is CUDA enabled [here](https://developer.nvidia.com/cuda-gpus).

I had a problem with CUDA in the beginning, problem/solution couple available [here on StackOverflow](https://stackoverflow.com/questions/68546140/cant-train-with-gpu-in-tensorflow).

### CUDA Toolkit

You need to install CUDA Toolkit (works with version 11.2) in your System: [download](https://developer.nvidia.com/cuda-toolkit).

### cuDNN Library

You need to install cuDNN (works with version 8.1) Library in your System: [download](https://developer.nvidia.com/rdp/cudnn-download) and follow NVDIA [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

# References & useful material

| Name                                                                                           | Ref                                                                                                                                      |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| The paper which resumes the work done in this repository                                       | [link](https://doi.org/10.1109/IWSDA50346.2022.9870449)                                                                                     |
| Background, history and an introduction to Automatic Modulation Recognition with Deep Learning | [link](https://www.researchgate.net/publication/332241963_A_Robust_Modulation_Classification_Method_Using_Convolutional_Neural_Networks) |
| Dataset RadioML2016.10a                                                                        | [link](https://pubs.gnuradio.org/index.php/grcon/article/view/11)                                                                        |
| CNN2 neural network                                                                            | [link](https://arxiv.org/pdf/1602.04105.pdf)                                                                                             |
| RadioML2018.01a, an extended version of the dataset used in my thesis                          | [link](https://ieeexplore.ieee.org/document/8267032)                                                                                     |
| Baseline neural network, relabeling and ensamble learning                                      | [link](https://ieeexplore.ieee.org/document/8935775)                                                                                     |
| Batch normalization                                                                            | [link](https://arxiv.org/abs/1502.03167)                                                                                                 |
