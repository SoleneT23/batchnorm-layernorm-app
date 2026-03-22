"""This module contains the function used for visualization of an image dataset.

Notes
-----
Inspired by https://nextjournal.com/gkoehler/pytorch-mnist.
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np


def mnist_like_viz(data, classes, model=None):
    """Selects randomly 6 images from an MNIST-like dataset and plot them.

    Also plots the label as "GT" (ground truth) and, if a model is given,
    the prediction of the model on the given images.

    Parameters
    ----------
    data: torchvision.datasets.MNIST
        An MNIST-like dataset of grayscale labeled images.
    classes: list
        The list of the labale classes.
    model: torch.nn.module, default=None
        A torch model used to predict labels from the selected
        images. If None, there is simply no prediction.

    Returns
    -------
    None

    """
    if model is not None:
        model.eval()

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        sample_idx = torch.randint(len(data), size=(1,)).item()
        x, y = data[sample_idx][0], data[sample_idx][1]
        actual_class = classes[y]
        if model is not None:
            with torch.no_grad():
                pred = model(x)
                predicted = classes[pred[0].argmax(0)]
        plt.imshow(x[0], cmap="gray", interpolation="none")
        if model is None:
            plt.title("Ground Truth: {}".format(actual_class))
        else:
            plt.title(f"GT: {actual_class}, \nPred: {predicted}")
        plt.xticks([])
        plt.yticks([])
    st.pyplot(fig)


def training_curves(model, mode=None):
    """Plot training and test metrics over epochs.

    This function displays two plots:
    - training and test loss over epochs,
    - training and test accuracy over epochs.

    Parameters
    ----------
    model : nn.Module
        Model containing a `metrics` DataFrame with columns
        ["train_loss", "train_acc", "test_loss", "test_acc"].
    mode : str, optional
        Either "script" for console execution (matplotlib display)
        or "st" for Streamlit display.

    Returns
    -------
    None
    """
    fig = plt.figure()
    fig.set_figheight(10)
    padding = 2
    model.metrics.index = np.array(model.metrics.index) + 1
    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=padding)
    plt.plot(model.metrics.index, "train_loss", data=model.metrics)
    plt.plot(model.metrics.index, "test_loss", data=model.metrics)
    plt.legend()
    plt.title("Train and test loss during training")
    plt.xlabel("Epoch")
    plt.subplot(2, 1, 2)
    plt.tight_layout(pad=padding)
    plt.plot(model.metrics.index, "train_acc", data=model.metrics)
    plt.plot(model.metrics.index, "test_acc", data=model.metrics)
    plt.legend()
    plt.title("Train and test accuracy during training")
    plt.xlabel("Epoch")
    if mode == "script":
        plt.show()
    elif mode == "st":
        st.pyplot(fig)


def compare_training_curves(models, mode=None):
    """Compare test loss and test accuracy for several models.

    Parameters
    ----------
    models: list
        A list of tuples (label, model), where label identifies each model
        (for example a learning rate or a normalization type).
    mode: str
        Either "script" or "st".
    """
    fig = plt.figure()
    fig.set_figheight(10)
    padding = 2

    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=padding)
    for label, model in models:
        epochs = np.array(model.metrics.index) + 1
        plt.plot(epochs, model.metrics["test_loss"], label=str(label))
    plt.legend()
    plt.title("Test loss comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2, 1, 2)
    plt.tight_layout(pad=padding)
    for label, model in models:
        epochs = np.array(model.metrics.index) + 1
        plt.plot(epochs, model.metrics["test_acc"], label=str(label))
    plt.legend()
    plt.title("Test accuracy comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    if mode == "script":
        plt.show()
    elif mode == "st":
        st.pyplot(fig)
