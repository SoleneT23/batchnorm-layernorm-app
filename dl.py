"""Functions for the deep learning mode.

Notes
-----
Inspired from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.
"""

import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import streamlit as st

from viz import training_curves
from utils import paths, paths_rnn

from collections import Counter
import re

from datasets import load_dataset


def get_FashionMNIST_datasets(batch_size=64, only_loader=True):
    """Loads and returns the FashionMNIST dataset.

    The data is returned as DataLoader objects by default or,
    if specified by the user, also as datasets.

    Parameters
    ----------
    batch_size: int, default=64
        The batch size to use for the DataLoader objects.
    only_loader: bool, default=True
        If `True`, returns only the DataLoader objects.
        If `False`, also returns the datasets.

    Returns
    -------
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader of the training data.
    test_dataloader: torch.utils.data.DataLoader
        The DataLoader of the test data.
    training_data: torchvision.datasets.mnist.FashionMNIST
        The training data, only returned if `only_loader==False`.
    test_data: torchvision.datasets.mnist.FashionMNIST
        The test data, only returned if `only_loader==False`.

    """
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if only_loader:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader, training_data, test_data


# Define model
class FMNIST_MLP(nn.Module):
    """The MLP model we train on the FashionMNIST dataset.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Attributes
    ----------
    flatten: nn.Flatten
        A flatten layer.
    linear_relu_stack: nn.Sequential
        A stack of liner layers with ReLU
        activations.
    metrics: pd.DataFrame
        The training metrics dataframe.
    """

    def __init__(self, hidden_layers=2, dropout_rate=0.0,  norm_type="none"):

        super().__init__()
        self.flatten = nn.Flatten()


        def norm_layer():
            if norm_type == "batchnorm":
                return nn.BatchNorm1d(512)
            elif norm_type == "layernorm":
                return nn.LayerNorm(512)
            else:
                return nn.Identity() 
        
        list_hidden = []
        for _ in range(hidden_layers - 1):
            list_hidden.append(nn.Linear(512, 512))
            list_hidden.append(norm_layer())
            list_hidden.append(nn.ReLU())
            list_hidden.append(nn.Dropout(dropout_rate))

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            norm_layer(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *list_hidden,
            nn.Linear(512, 10),
        )
        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.

        Returns
        -------
        logits: Tensor
            The unnormalized logits, of shape `(batch_size, 10)`.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_metrics(self, df):
        """Sets the metrics dataframe.

        Used when a saved model is loaded,
        to also load its past training metrics dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            The training metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = df

    def update_metrics(self, series):
        """Updates the metrics dataframe after one epoch.

        Parameters
        ----------
        series: pd.Series
            The new row of metrics to add at the
            end of the metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)


def train_step(dataloader, model, loss_fn, optimizer, device, mode=None):
    """The training step for one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The training DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    optimizer: torch.optim.optimizer.Optimizer
        The optimizer.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    train_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.
    correct: float
        The accuracy of all the predictions on the epoch,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (mode == "script") & (batch % 100 == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    return train_loss, correct


def test_step(dataloader, model, loss_fn, device, mode=None):
    """The evaluation step after one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The test DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    test_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.
    correct: float
        The accuracy of all the predictions on all the batches,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if mode == "script":
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return test_loss, correct


def get_and_train_model(
    train_dataloader,
    test_dataloader,
    hidden_layers=2,
    dropout_rate=0.0,
    norm_type="none", 
    lr= 1e-3,
    epochs=5,
    mode=None,
    progress_prefix="",
):
    """Creates and trains a model on the given dataset.

    Doesn't train if saved weights are found for the given hyperparameters
    (except the number of epochs). The MLP architecture is displayed.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader of the training data.
    test_dataloader: torch.utils.data.DataLoader
        The DataLoader of the test data.
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.
    norm_type: str, default="none"
        The type of normalization applied after each hidden layer.
        Must be one of {"none", "batchnorm", "layernorm"}.
    lr: float, default=1e-3
        The learning rate used by the SGD optimizer.
    epochs: int, default=5
        The number of epochs used for training.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    model: FMNIST_MLP
        The model.
    """
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "script":
        print(f"Using {device} device")

    # Create the model
    model = FMNIST_MLP(hidden_layers, dropout_rate, norm_type=norm_type) 
    path_weights, path_metrics = paths(hidden_layers, dropout_rate, norm_type=norm_type, lr= lr, epochs=epochs) 

    # Load the weights if they already exist
    if os.path.exists(path_weights):
        if mode == "script":
            print("model already exists, let us just load it")
        elif mode == "st":
            st.write("Found a saved model with given config")
        #model.load_state_dict(torch.load(path_weights))
        state = torch.load(path_weights, weights_only=True, map_location=device) 
        model.load_state_dict(state) 
        metrics = pd.read_csv(path_metrics, index_col=0)
        model.set_metrics(metrics)
    model = model.to(device)
    if mode == "script":
        print(model)
    elif mode == "st":
        st.text("Model architecture:")
        st.text(model)

    # Train the model and save the weights if they don't exist
    if not os.path.exists(path_weights):
        if mode == "script":
            print("No existing model found")
        elif mode == "st":
            st.write("Didn't find an existing model, training a new one")

        loss_fn = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        epoch_progress_bar = None
        epoch_status_text = None
        stop_message_placeholder = None

        if mode == "st":
            if "stop_training" not in st.session_state:
                st.session_state.stop_training = False

            epoch_progress_bar = st.progress(0)
            epoch_status_text = st.empty()

            stop_col, info_col = st.columns([1, 3])

            with stop_col:
                if st.button(
                    "Stop training",
                    key=(
                        f"stop_button_{progress_prefix}"
                        f"{hidden_layers}_{dropout_rate}_{norm_type}_{lr}_{epochs}"
                    ),
                ):
                    st.session_state.stop_training = True

            with info_col:
                st.caption("Training will stop after the current epoch is completed.")

            stop_message_placeholder = st.empty()

        stopped_early = False
        for t in range(epochs):
            if mode == "script":
                print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train_step(
                train_dataloader, model, loss_fn, optimizer, device, mode
            )
            test_loss, test_acc = test_step(
                test_dataloader, model, loss_fn, device, mode
            )

            # Saved the metrics in the model.metrics dataframe
            new_row = pd.Series(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            model.update_metrics(new_row)

            if mode == "st":
                epoch_progress_bar.progress((t + 1) / epochs)
                epoch_status_text.write(
                    f"{progress_prefix}Epoch {t + 1}/{epochs} completed — "
                    f"train loss: {train_loss:.4f}, "
                    f"test loss: {test_loss:.4f}, "
                    f"train acc: {100 * train_acc:.1f}%, "
                    f"test acc: {100 * test_acc:.1f}%"
                )
            
            if st.session_state.stop_training:
                stopped_early = True
                stop_message_placeholder.warning(
                    f"{progress_prefix}Stop requested. Training stops now that "
                    f"epoch {t + 1} is finished."
                )
                break

        if mode == "script":
            print("Done!")

        if stopped_early:
            if mode == "st":
                epoch_status_text.info(
                    f"{progress_prefix}Training was stopped by the user. "
                    f"Partial training was not saved."
                )
            return model

        # Save the weights and the metrics dataframe
        torch.save(model.state_dict(), path_weights)
        model.metrics.to_csv(path_metrics)

        if mode == "script":
            print("Saved PyTorch Model State to " + path_weights)
            print(model.metrics)
        elif mode == "st":
            epoch_status_text.write(f"{progress_prefix}Training finished and model saved.")
            epoch_progress_bar.progress(1.0)

    return model


def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenizer(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=200):
    tokens = simple_tokenizer(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_text(self.texts[idx], self.vocab, self.max_len),
            dtype=torch.long,
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class IMDB_RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout_rate=0.0,
        norm_type="none",
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )

        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        # x shape: (batch, seq_len), containing token indices
        x = self.embedding(x)                 # (batch, seq_len, embedding_dim)
        output, h_n = self.rnn(x)

        # For 1-layer unidirectional RNN:
        # h_n shape = (1, batch, hidden_dim)
        h_last = h_n[-1]                      # (batch, hidden_dim)

        h_last = self.norm(h_last)
        h_last = self.dropout(h_last)

        logits = self.fc(h_last).squeeze(1)   # (batch,)
        return logits

    def set_metrics(self, df):
        self.metrics = df

    def update_metrics(self, series):
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)


def train_step_binary(dataloader, model, loss_fn, optimizer, device, mode=None):
    """Training step for binary classification."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)                    # (batch,)
        loss = loss_fn(logits, y)

        train_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    correct /= size
    return train_loss, correct

def test_step_binary(dataloader, model, loss_fn, device, mode=None):
    """Evaluation step for binary classification.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Test dataloader.
    model : nn.Module
        The binary classification model.
    loss_fn : nn.modules._Loss
        Typically nn.BCEWithLogitsLoss().
    device : str
        "cuda" or "cpu".
    mode : str, optional
        Either "script" or "st".

    Returns
    -------
    test_loss : float
        Mean loss over batches.
    correct : float
        Accuracy over all examples, between 0 and 1.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float()

            logits = model(X)                     # shape: (batch,)
            loss = loss_fn(logits, y)
            test_loss += loss.item()

            probs = torch.sigmoid(logits)         # convert logits to probabilities
            preds = (probs >= 0.5).float()        # threshold at 0.5
            correct += (preds == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    if mode == "script":
        print(
            f"Test Error:\n Accuracy: {(100 * correct):>0.1f}%, "
            f"Avg loss: {test_loss:>8f}\n"
        )

    return test_loss, correct

def get_and_train_rnn_model(
    train_dataloader,
    test_dataloader,
    vocab_size,
    embedding_dim=64,
    hidden_dim=128,
    num_layers=1,
    dropout_rate=0.0,
    norm_type="none",
    lr=1e-3,
    epochs=5,
    max_len=200,
    train_size=None,
    mode=None,
    progress_prefix="",
):
    """Creates, trains, loads, and saves an RNN model for IMDb sentiment analysis.

    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    test_dataloader : torch.utils.data.DataLoader
        Test dataloader.
    vocab_size : int
        Vocabulary size for nn.Embedding.
    embedding_dim : int, default=64
        Embedding dimension.
    hidden_dim : int, default=128
        Hidden dimension of the RNN.
    num_layers : int, default=1
        Number of recurrent layers.
    dropout_rate : float, default=0.0
        Dropout rate after normalization / hidden representation.
    norm_type : str, default="none"
        One of {"none", "batchnorm", "layernorm"}.
    lr : float, default=1e-3
        Learning rate.
    epochs : int, default=5
        Number of epochs.
    max_len : int, default=200
        Maximum sequence length used in preprocessing.
    train_size : int, optional
        Size of the training subset, useful for naming saved files.
    mode : str, optional
        "script" or "st".
    progress_prefix : str, default=""
        Prefix used in Streamlit progress messages.

    Returns
    -------
    model : IMDB_RNN
        The loaded or trained model.
    """
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "script":
        print(f"Using {device} device")

    model = IMDB_RNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        norm_type=norm_type,
    )

    path_weights, path_metrics = paths_rnn(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        norm_type=norm_type,
        epochs=epochs,
        lr=lr,
        max_len=max_len,
        train_size=train_size,
    )

    # Load if saved model exists
    if os.path.exists(path_weights):
        if mode == "script":
            print("RNN model already exists, loading it")
        elif mode == "st":
            st.write("Found a saved RNN model with given config")

        state = torch.load(path_weights, weights_only=True, map_location=device)
        model.load_state_dict(state)

        metrics = pd.read_csv(path_metrics, index_col=0)
        model.set_metrics(metrics)

    model = model.to(device)

    if mode == "script":
        print(model)
    elif mode == "st":
        st.text("RNN model architecture:")
        st.text(model)

    # Train only if weights do not exist
    if not os.path.exists(path_weights):
        if mode == "script":
            print("No existing RNN model found")
        elif mode == "st":
            st.write("Didn't find an existing RNN model, training a new one")

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        epoch_progress_bar = None
        epoch_status_text = None
        stop_message_placeholder = None

        if mode == "st":
            if "stop_training" not in st.session_state:
                st.session_state.stop_training = False

            epoch_progress_bar = st.progress(0)
            epoch_status_text = st.empty()

            stop_col, info_col = st.columns([1, 3])

            with stop_col:
                if st.button(
                    "Stop training",
                    key=(
                        f"stop_rnn_button_{progress_prefix}"
                        f"{embedding_dim}_{hidden_dim}_{num_layers}_"
                        f"{dropout_rate}_{norm_type}_{lr}_{epochs}_{max_len}"
                    ),
                ):
                    st.session_state.stop_training = True

            with info_col:
                st.caption("Training will stop after the current epoch is completed.")

            stop_message_placeholder = st.empty()

        stopped_early = False

        for t in range(epochs):
            if mode == "script":
                print(f"Epoch {t + 1}\n-------------------------------")

            train_loss, train_acc = train_step_binary(
                train_dataloader, model, loss_fn, optimizer, device, mode
            )
            test_loss, test_acc = test_step_binary(
                test_dataloader, model, loss_fn, device, mode
            )

            new_row = pd.Series(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            model.update_metrics(new_row)

            if mode == "st":
                epoch_progress_bar.progress((t + 1) / epochs)
                epoch_status_text.write(
                    f"{progress_prefix}Epoch {t + 1}/{epochs} completed — "
                    f"train loss: {train_loss:.4f}, "
                    f"test loss: {test_loss:.4f}, "
                    f"train acc: {100 * train_acc:.1f}%, "
                    f"test acc: {100 * test_acc:.1f}%"
                )

            if mode == "st" and st.session_state.stop_training:
                stopped_early = True
                stop_message_placeholder.warning(
                    f"{progress_prefix}Stop requested. Training stops now that "
                    f"epoch {t + 1} is finished."
                )
                break

        if mode == "script":
            print("Done!")

        if stopped_early:
            if mode == "st":
                epoch_status_text.info(
                    f"{progress_prefix}Training was stopped by the user. "
                    f"Partial training was not saved."
                )
            return model

        torch.save(model.state_dict(), path_weights)
        model.metrics.to_csv(path_metrics)

        if mode == "script":
            print("Saved PyTorch RNN model state to " + path_weights)
            print(model.metrics)
        elif mode == "st":
            epoch_status_text.write(
                f"{progress_prefix}Training finished and RNN model saved."
            )
            epoch_progress_bar.progress(1.0)

    return model

def get_IMDB_datasets(
    batch_size=64,
    max_len=200,
    train_size=2000,
    test_size=1000,
    min_freq=2,
    only_loader=True,
):
    """Loads and returns a processed IMDb sentiment dataset.

    The raw IMDb reviews are downloaded with Hugging Face datasets,
    then tokenized, converted to integer ids, padded/truncated to a
    fixed length, and wrapped in PyTorch Dataset / DataLoader objects.

    Parameters
    ----------
    batch_size : int, default=64
        Batch size for the DataLoader objects.
    max_len : int, default=200
        Maximum sequence length. Reviews longer than max_len are truncated,
        shorter reviews are padded.
    train_size : int, default=2000
        Number of training reviews to keep from the IMDb train split.
        Useful to keep the app fast.
    test_size : int, default=1000
        Number of test reviews to keep from the IMDb test split.
    min_freq : int, default=2
        Minimum word frequency required to enter the vocabulary.
    only_loader : bool, default=True
        If True, returns only train/test dataloaders.
        If False, also returns datasets and vocabulary.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    test_dataloader : torch.utils.data.DataLoader
        Test dataloader.
    train_dataset : IMDBDataset
        Returned only if only_loader is False.
    test_dataset : IMDBDataset
        Returned only if only_loader is False.
    vocab : dict
        Returned only if only_loader is False.
    """
    # Load raw IMDb dataset
    dataset = load_dataset("imdb")

    # Restrict to a smaller subset for speed
    train_split = dataset["train"].select(range(train_size))
    test_split = dataset["test"].select(range(test_size))

    # Extract texts and labels
    train_texts = [example["text"] for example in train_split]
    train_labels = [example["label"] for example in train_split]

    test_texts = [example["text"] for example in test_split]
    test_labels = [example["label"] for example in test_split]

    # Build vocabulary only on training texts
    vocab = build_vocab(train_texts, min_freq=min_freq)

    # Build PyTorch datasets
    train_dataset = IMDBDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        max_len=max_len,
    )

    test_dataset = IMDBDataset(
        texts=test_texts,
        labels=test_labels,
        vocab=vocab,
        max_len=max_len,
    )

    # Build dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    if only_loader:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader, train_dataset, test_dataset, vocab



if __name__ == "__main__":

    mode = "script"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument(
        "--norm",
        type=str,
        default="none",
        choices=["none", "batchnorm", "layernorm"],
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_dataloader, test_dataloader = get_FashionMNIST_datasets(64)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = get_and_train_model(
        train_dataloader,
        test_dataloader,
        hidden_layers=args.hidden,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        norm_type=args.norm,
        lr=args.lr,
        mode=mode,
    )
    training_curves(model, mode)


