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

MLP Comparison Modes

The MLP page allows users to explore the impact of normalization and hyperparameter tuning on a model trained on FashionMNIST. Three comparison modes are available:

1. No comparison: Train a single model with chosen hyperparameters (number of layers, dropout rate, learning rate, number of epochs, and optional BatchNorm). Observe that the metrics are better when a BN layer is used.

2. Compare learning rates: Fix the architecture and dropout rate, and compare several learning rates (e.g. 
1e−3,1e-2,1e−1). This experiment was designed to illustrate that Batch Normalization can enable the use of higher learning rates. In our experiments, however, we observed a different effect: BatchNorm did not necessarily allow strictly higher learning rates, but rather made training more robust to the choice of learning rate. In particular, all learning rates performed well with BatchNorm, whereas without BatchNorm the performance was more sensitive and only a narrower range of learning rates yielded good results.

3. Compare dropout rates: Fix the architecture and learning rate, and compare several dropout values. While this experiment is intended to illustrate that Batch Normalization can reduce the need for strong dropout regularization, our results show a different behavior. In our setting, the lowest dropout rate yields the best performance both with and without BatchNorm, suggesting that the model does not strongly benefit from dropout. However, we observe that with BatchNorm the model becomes less sensitive to the choice of dropout rate earlier during training, as the performance gap between different dropout values is smaller in the first epochs.

In all modes, users can choose to compare BatchNorm vs no BatchNorm, or focus on a single normalization setting.


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