# Generating High-Quality Adversarial Examples in Pattern-Based Autoencoder

## About the project
Contain source code of adversarial example generation method, namely HPBA.
This is a quick guide for running experiment. For detail configuration, please visit [HPBA_tool_manual.pdf](https://drive.google.com/file/d/1qBqTy0K1yBUJLE4gbbwgG_EvzA3DXVvr/view?usp=sharing)

<!-- **Note**: Current version of HPBA only supports 2D image classifier. The 3D image classifier will be supported in the next version. -->

## Table of Content 

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Data Preparation](#Data-Preparation)
    * [Training Data](#Training-data)
    * [Pre-trained DNN](#Pre-trained-DNN)
  * [Input Configuration](#Input-Configuration)

* [Run experiment](#Run-experiment)
  * [Run](#run)
  * [View Result](#View-Result)


## Getting started

### Prerequisites

* Python >= 3.7
* Tensorflow >= 2.4.0
### Installation
```sh
git clone https://github.com/testingforAI-vnuuet/HPBA.git
cd HPBA
pip install -r requirements.txt
```
**Note**: You may optionally wish to create a [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to prevent conflicts with your system's Python environment.
### Data Preparation

#### Training data
- Use training set and label set after being pre-processed. All files must be provided as '.npy' file.
- Download MNIST handwritten digits or MNIST fashion dataset for testting.

| Dataset                  | Download Link                                                       | Ref                                   |
|--------------------------|---------------------------------------------------------------------|---------------------------------------|
| MNIST handwritten digits | [handwritten_mnist_training.npy](https://drive.google.com/file/d/1R7gvFYTrtH75cV7qDg_zaQJ5J8ccIZCV/view?usp=sharing) <br/> [handwritten_mnist_label.npy](https://drive.google.com/file/d/1miFdEi1X8Fr6hZx9_9UbWOmC8MPS27AJ/view?usp=sharing) | [MNIST handwritten digits database](http://yann.lecun.com/exdb/mnist/) |
| MNIST Fashion            | [handwritten_mnist_training.npy](https://drive.google.com/file/d/1rEDOowWbCvKFPphJMtoSW0UHvEZYAmxV/view?usp=sharing)  <br/> [fashion_mnist_label.npy](https://drive.google.com/file/d/1miFdEi1X8Fr6hZx9_9UbWOmC8MPS27AJ/view?usp=sharing)     | [Fashion MNIST on Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)           |


#### Pre-trained DNN
- Use pre-trained DNN that has high accuracy.
- All files must be provided as '.h5' file.
  
  Example:

| Dataset                  | Download Link            | Accuracy |
|--------------------------|--------------------------|----------|
| MNIST handwritten digits | [Alexnet.h5](https://drive.google.com/file/d/1eBmWjM3HPp2Ci3e6dhd7iMNYCik2Se8q/view?usp=sharing)           | ~ 99%    |
| MNIST Fashion            | [fashionMnistModel.h5](https://drive.google.com/file/d/1aVk4oMzOSqsh7qzF_zXUC0Qy2ftHmP_B/view?usp=sharing) | ~ 90%    |

[comment]: <> (  - Hand-written digit MNIST: [handwritten_mnist_model]&#40;https://drive.google.com/file/d/1eBmWjM3HPp2Ci3e6dhd7iMNYCik2Se8q/view?usp=sharing&#41;)

[comment]: <> (  - Fashion MNIST: [fashion_mnist_model]&#40;https://drive.google.com/file/d/1aVk4oMzOSqsh7qzF_zXUC0Qy2ftHmP_B/view?usp=sharing&#41;)
### Input Configuration
- Input required argument in file [config.ini](config.ini).
- An example of legitimate of configuration in [config_tmp.ini](config_tmp.ini)


**Note**: You must provide argument that has default of **None**

[comment]: <> (- Example: Please open [config.ini]&#40;config.ini&#41;)
### Run experiment
#### Run
```sh
python main.py
```
#### View Result
- To view summary: open folder `results/hpba/result_summary`
- To access generated advs: open folder `results/hpba/data`
- To access trained autoencoder: open folder `resuts/hpba/autoencoder`
- To view some random results: open folder `results/hpbd/image`

**Note**: Result file name contains time stamp, i.e. {file_name}_{time_stamp}.{file_format}

**Developers**: Rd320 room, E3 building, 144 Xuanthuy str., Caugiay dist., Hanoi, Vietnam. 

If you have any questions, please contact us via Kha Do Minh <khadm@vnu.edu.vn> .

This project welcomes contributions and suggestions.
