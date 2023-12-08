# Anomaly Detector for Pedestrian Areas

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project structure](#2-project-structure)
3. [Requirements to run the project](#3-requirements-to-run-the-project)
4. [Future Development](#4-future-developments)

## 1. Introduction

The proposed system leverages autoencoder-based anomaly detection in pedestrian zones, detecting anomalies such as bikers, skaters, vans, etc. Utilizing the architecture developed by _Yong Shean Chong et al_., the algorithm enhances anomaly identification in surveillance video streams, employing Convolutional LSTM to analyze 10 frames simultaneously for improved detection of irregular patterns in pedestrian activities.

## 2. Project Structure

The project is structured as follows:

```
|–– output
|    |–– Test001
|    |–– Test002
|    |      .
|    |      .
|    |      .
|    |–– Test035
|–– parameters
|    |–– autoencoder_ucsd_convLSTMAE.params
|--UCSD_Anomaly_Dataset
|–– .gitignore
|-- Documentazione.pdf
|-- LICENSE
|–– README.md
|-- convLSTMAE.py
|-- main.py
|–– utils.py
```

Below are the roles of the different components:

- **output**: folder containing the output frames of the 35 tests with anomalies highlighted in red;
- **parameters**: folder containing the .params file saved after completing the neural network training;
- **UCSD_Anomaly_Dataset**: dataset used for system development;
- **main.py**: source file used as the project's main file;
- **convLSTMAE.py**: source file used to define the neural network and its training;
- **utils.py**: source file used to define the creation of the dataloader and the plot of frames with highlighted anomalies;
- **.gitignore**: file specifying all files to be excluded from the version control system;
- **Documentation.pdf**: documentation of the case study.

## 3. Requirements to run the project

To run the project, it is necessary to install the following programs:
- `Python 3.6.0`
- `Mxnet 1.6.0`
- `Mxnet-cu101 1.5.0`
- `Matplotlib 3.3.4`
- `Pillow 8.2.0`
- `Scipy 1.5.4`
- `Numpy 1.16.6`

To run the code on GPU, install the following programs:
- `Cuda 10.1`
- `cudnn for Cuda 10.1 - v.7.6.5.32`

If you don't have a GPU, change line 13 of the `main.py` file to `ctx = mx.cpu()` in order to perform training with CPU only.

After setting up the Python environment correctly, you can run the code in `main.py`. `Line 21` is commented out so that the neural network parameters in the parameters folder do not need to be retrained. When the execution of `main.py` is finished, the system will have created an output folder in the project directory, containing 35 Test folders, each with 200 frames of individual videos with their respective anomalies highlighted.

## 4. Future Developments

In the future, the developed system can be widely used by multiple cities and metropolises for identifying anomalies in pedestrian zones, as well as in various other contexts of use. One example is to retrain the software for anomalies in urban and rural roads. To do this, it will be necessary to use a different dataset from the one used so far.
