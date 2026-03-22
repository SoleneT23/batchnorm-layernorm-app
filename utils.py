"""This module contains useful functions for other modules.
"""
from pathlib import Path
import requests

def ensure_saved_models_dir(saved_models_dir):
    """Create saved_models directory if it does not exist."""
    Path(saved_models_dir).mkdir(parents=True, exist_ok=True)
    
def download_file(url, destination, chunk_size=8192):
    """Download a file from a URL to a local destination."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                
BASE_URL = "https://github.com/SoleneT23/methodes-ia/releases/download/v1/"

def download_if_missing(path):
    """
    Download a file from BASE_URL if it does not exist locally.
    """
    path = Path(path)

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    filename = path.name
    url = BASE_URL + filename

    try:
        download_file(url, path)
    except Exception:
        raise FileNotFoundError(
            f"File not found locally and not available online:\n{url}"
        )



def paths(hidden_layers=2, dropout_rate=0.0, norm_type="none", lr=None, epochs=None):
    """File paths for model weights and metrics from model parameters.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.
    norm_type: str, default="none"
        The type of normalization applied after each hidden layer.
        Must be one of {"none", "batchnorm", "layernorm"}.
    lr: float or None, default=None
        The learning rate used to train the model.
    epochs: int or None, default=None
        The number of training epochs used to generate the model.


    Returns
    -------
    path_model: string
        The file path of the model weights.
    path_metrics: string
        The file path of the model metrics computed during training.

    """
    base_name = (
        "saved_models/fmnist_mlp"
        + "_hidden_" + str(hidden_layers)
        + "_dropout_rate_" + str(dropout_rate)
        + "_norm_" + str(norm_type)
    )

    if epochs is not None:
        base_name += "_epochs_" + str(epochs)

    if lr is not None:
        base_name += "_lr_" + str(lr)

    path_weights = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"
    return path_weights, path_metrics


def paths_rnn(
    embedding_dim=64,
    hidden_dim=128,
    num_layers=1,
    dropout_rate=0.0,
    norm_type="none",
    epochs=None,
    lr=None,
    max_len=200,
    train_size=2000,
):
    base_name = (
        "saved_models/imdb_rnn"
        + f"_emb_{embedding_dim}"
        + f"_hid_{hidden_dim}"
        + f"_layers_{num_layers}"
        + f"_dropout_{dropout_rate}"
        + f"_norm_{norm_type}"
        + f"_maxlen_{max_len}"
        + f"_ntrain_{train_size}"
    )

    if epochs is not None:
        base_name += f"_epochs_{epochs}"
    if lr is not None:
        base_name += f"_lr_{lr}"

    path_weights = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"
    return path_weights, path_metrics