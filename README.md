[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Projet de Deep Learning

## Installation

### Standard installation

pip install -r requirements.txt

### If installation fails (especially on pyarrow)

Some environments may fail when installing `datasets` with pip due to `pyarrow`.

In that case, run:

conda install -c conda-forge pyarrow datasets

# Fashion-MNIST MLP

"MLP and normalization" trains (or use trained weights if available) a simple artificial neuron network on the Fashion MNIST datatset, displays the architecture, displays the curves of train and test loss and train and test accuracy. The user can select the number of hidden layers (it is a simple MLP), the level of dropout, the number of epochs, whether to add a Batch Normalization layer or not, and can compare different
learning rates.
If trained weights for the same combination of hyperparameters are found, they are used. If not, a model is trained and then the weights and metrics are saved. In the former case, a button allows the user to delete the trained weights and metrics and start a new training.


## Effect of Batch Normalization

We compare the test loss and test accuracy with and without Batch Normalization (BN). 
We observe that the metrics are generally better when a BN layer is used.

## Effect of Learning Rate with and without Batch Normalization

We compare the impact of the learning rate on model performance, both with and without Batch Normalization.

In the paper *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* [Ioffe & Szegedy, 2015], the authors show that Batch Normalization allows the use of higher learning rates. 

However, in our experiments on FashionMNIST, we observe a different behavior:
the model without BN performs better when the learning rate increases, while the model with BN performs better with a smaller learning rate.

One possible explanation is the choice of activation function.
The paper uses sigmoid activations, which can easily lead to saturation and vanishing gradients, making training more sensitive to the learning rate. In contrast, our implementation uses ReLU activations, which do not have this issues.


# "RNN Sentiment analysis" mode

The RNN page compares Batch Normalization and Layer Normalization on an IMDb sentiment classification task. For a fixed architecture and training setup, both models are trained and their loss and accuracy curves are displayed on the same graphs.

## Maximum sequence length

The maximum sequence length controls how many words from each review are given as input to the model.

If a review is longer than this value, it is truncated (only the first words are kept).
If a review is shorter, it is padded with special tokens so that all inputs have the same length.


## Train subset size and Test subset size : 
Train subset size controls how many training examples are used to train the RNN.
Test subset size controls how many examples are used to evaluate its performance.
Using smaller subsets makes training much faster. 

# Retraining a model from scratch

1. Check the “Force retrain” option.
2. Click “Delete local copy (will be re-downloaded)” to remove any existing local files.
3. Click “Train / Load model”.

The model will then be trained from scratch instead of being loaded or downloaded.



## Miscellaneous

### docstrings

This repo uses the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for its docstrings.