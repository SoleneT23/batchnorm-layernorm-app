"""This module contains useful functions for other modules.
"""


def paths(hidden_layers=2, dropout_rate=0.0, norm_type="none", lr=None, epochs = None):
    """File paths for model weights and metrics from model parameters.

    The input vector is evaluated element-wise
    to the power 1, 2, ..., `order`. The resulting vectors
    are then concatenated and returned.

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
        "saved_models/fmnist_mlp_hidden="
        + str(hidden_layers)
        + "_dropout_rate="
        + str(dropout_rate)
        + "_norm="
        + str(norm_type)

    )

    if epochs is not None:
        base_name += "_epochs=" + str(epochs)

    if lr is not None:
        base_name += "_lr=" + str(lr)

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
        + f"_emb={embedding_dim}"
        + f"_hid={hidden_dim}"
        + f"_layers={num_layers}"
        + f"_dropout={dropout_rate}"
        + f"_norm={norm_type}"
        + f"_maxlen={max_len}"
        + f"_ntrain={train_size}"
    )

    if epochs is not None:
        base_name += f"_epochs={epochs}"
    if lr is not None:
        base_name += f"_lr={lr}"

    path_weights = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"
    return path_weights, path_metrics