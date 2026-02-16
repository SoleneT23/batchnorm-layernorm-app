# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from IPython.display import display

# iowa_file_path = "./home-data-for-ml-course/train.csv"

# home_data = pd.read_csv(iowa_file_path)
# # Create target object and call it y
# y = home_data.SalePrice
# # Create X
# features = [
#     "LotArea",
#     "YearBuilt",
#     "1stFlrSF",
#     "2ndFlrSF",
#     "FullBath",
#     "BedroomAbvGr",
#     "TotRmsAbvGrd",
# ]
# X = home_data[features]

# display(X.head())

# # Split into validation and training data
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# # Specify Model
# iowa_model = DecisionTreeRegressor(random_state=1)
# # Fit Model
# iowa_model.fit(train_X, train_y)
# # Make validation predictions and calculate mean absolute error
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(val_predictions, val_y)
# print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# # Using best value for max_leaf_nodes
# iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
# iowa_model.fit(train_X, train_y)
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(val_predictions, val_y)
# print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# # Define the model. Set random_state to 1
# rf_model = RandomForestRegressor(random_state=1)
# rf_model.fit(train_X, train_y)
# rf_val_predictions = rf_model.predict(val_X)
# rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
# print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# # Import the necessary modules and libraries
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt

# # Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))

# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_1.fit(X, y)
# regr_2.fit(X, y)

# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)

# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


class RunConfig:
    """
    Stocker les hyperparamètres d'un entraînement 
    """
    def __init__(
        self,
        use_batchnorm: bool,
        hidden_layers: int,
        hidden_dim: int,
        dropout: float,
        lr: float,
        batch_size: int,
        epochs: int,
        seed: int,
        device: str,
    ):
        self.use_batchnorm = use_batchnorm
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.device = device

device = "cpu"





        """
Streamlit page for Batch Normalization experiments.

This module defines a Streamlit "mode" that compares training of the same MLP
architecture with and without Batch Normalization, on FashionMNIST (by default).

The training is ONLY triggered by a button click to avoid heavy recomputation
on every Streamlit rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bn_models import MLPNoBN, MLPWithBN
from train_utils import train_loop, evaluate


@dataclass
class RunConfig:
    """Configuration for a training run.

    Parameters
    ----------
    use_batchnorm : bool
        Whether to use BatchNorm in the model.
    hidden_layers : int
        Number of hidden layers.
    hidden_dim : int
        Width of each hidden layer.
    dropout : float
        Dropout probability.
    lr : float
        Learning rate.
    batch_size : int
        Batch size.
    epochs : int
        Number of epochs.
    seed : int
        Random seed.
    device : str
        Torch device string (e.g., "cpu", "cuda", "mps").
    """

    use_batchnorm: bool
    hidden_layers: int
    hidden_dim: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    seed: int
    device: str


@st.cache_resource
def _get_device() -> str:
    """Select an available device.

    Returns
    -------
    str
        "cuda" if available, else "mps" if available (Apple Silicon), else "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource
def _get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders for FashionMNIST.

    Parameters
    ----------
    batch_size : int
        Batch size.

    Returns
    -------
    (DataLoader, DataLoader)
        Train and test dataloaders.
    """
    train_ds = FashionMNIST("data", train=True, download=True, transform=ToTensor())
    test_ds = FashionMNIST("data", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: RunConfig) -> torch.nn.Module:
    """Build the MLP model (with or without BatchNorm)."""
    if cfg.use_batchnorm:
        return MLPWithBN(
            input_dim=28 * 28,
            num_classes=10,
            hidden_layers=cfg.hidden_layers,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
    return MLPNoBN(
        input_dim=28 * 28,
        num_classes=10,
        hidden_layers=cfg.hidden_layers,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )


def _plot_curves(history: Dict[str, List[float]], title: str) -> None:
    """Plot loss/accuracy curves using matplotlib."""
    fig1 = plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{title} — Loss")
    plt.legend()
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["test_acc"], label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{title} — Accuracy")
    plt.legend()
    st.pyplot(fig2)


def bn_experiment() -> None:
    """Streamlit mode: BatchNorm experiment (BN vs no BN).

    The user chooses hyperparameters, then clicks a button to train either:
    - one model (BN ON or OFF), or
    - both models for a direct comparison.

    Returns
    -------
    None
    """
    st.header("Batch Normalization experiment (BN vs no BN)")
    st.write(
        "This page trains a simple MLP on FashionMNIST with the same architecture, "
        "either **with** or **without** Batch Normalization. "
        "Training starts only when you click **Train**."
    )

    device = _get_device()
    st.caption(f"Device: `{device}`")

    col_a, col_b = st.columns(2)
    with col_a:
        compare_both = st.checkbox("Compare BN vs no BN (train both)", value=True)
        use_bn = st.checkbox("Use BatchNorm (single run only)", value=True, disabled=compare_both)

        hidden_layers = st.slider("Hidden layers", 1, 6, 2)
        hidden_dim = st.slider("Hidden dimension", 32, 512, 128, step=32)

    with col_b:
        dropout = st.slider("Dropout", 0.0, 0.9, 0.0, 0.1)
        lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
        batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=64)
        epochs = st.slider("Epochs", 1, 50, 10)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=1, step=1)

    train_loader, test_loader = _get_dataloaders(batch_size)

    # Clear previous results
    if st.button("Clear results"):
        for k in ["history_bn", "history_nobn", "final_bn", "final_nobn"]:
            st.session_state.pop(k, None)
        st.success("Cleared.")

    # Train
    if st.button("Train"):
        _set_seed(seed)

        runs = []
        if compare_both:
            runs = [
                RunConfig(True, hidden_layers, hidden_dim, dropout, float(lr), batch_size, epochs, seed, device),
                RunConfig(False, hidden_layers, hidden_dim, dropout, float(lr), batch_size, epochs, seed, device),
            ]
        else:
            runs = [
                RunConfig(bool(use_bn), hidden_layers, hidden_dim, dropout, float(lr), batch_size, epochs, seed, device)
            ]

        for cfg in runs:
            label = "BN" if cfg.use_batchnorm else "NoBN"
            st.write(f"### Training: {label}")

            model = _build_model(cfg).to(cfg.device)

            history = train_loop(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=cfg.epochs,
                lr=cfg.lr,
                device=cfg.device,
            )

            test_loss, test_acc = evaluate(model, test_loader, device=cfg.device)
            st.write(f"Final test loss: **{test_loss:.4f}**, Final test accuracy: **{test_acc:.4f}**")

            # Store in session for reruns without retraining
            if cfg.use_batchnorm:
                st.session_state["history_bn"] = history
                st.session_state["final_bn"] = {"test_loss": test_loss, "test_acc": test_acc}
            else:
                st.session_state["history_nobn"] = history
                st.session_state["final_nobn"] = {"test_loss": test_loss, "test_acc": test_acc}

    # Display saved results if available
    if "history_bn" in st.session_state:
        st.subheader("Results: With BatchNorm")
        _plot_curves(st.session_state["history_bn"], title="With BN")
        st.json(st.session_state.get("final_bn", {}))

    if "history_nobn" in st.session_state:
        st.subheader("Results: Without BatchNorm")
        _plot_curves(st.session_state["history_nobn"], title="Without BN")
        st.json(st.session_state.get("final_nobn", {}))

    if ("history_bn" in st.session_state) and ("history_nobn" in st.session_state):
        st.subheader("Quick comparison")
        bn_acc = st.session_state["final_bn"]["test_acc"]
        nobn_acc = st.session_state["final_nobn"]["test_acc"]
        st.write(f"Accuracy gain (BN - NoBN): **{bn_acc - nobn_acc:+.4f}**")