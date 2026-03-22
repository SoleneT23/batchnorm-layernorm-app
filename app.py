"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""
import os
import streamlit as st
from viz import mnist_like_viz, training_curves, compare_training_curves
from utils import paths, paths_rnn
import dl


def main():
    """The main function of the app.

    Returns
    -------
    None
    """
    
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Show instructions",           
            "MLP and normalization",
            "RNN sentiment analysis",
        ],
    )  
    if app_mode == "Show instructions":
        show_instructions()
    elif app_mode == "MLP and normalization":
        fashionmnist()
    elif app_mode == "RNN sentiment analysis":
        imdb_rnn_page()


def show_instructions():
    st.title("Instructions")

    st.write("""
    This app allows you to explore normalization techniques in deep learning.

    - MLP page: comparison of normalization methods on FashionMNIST
    - RNN page: comparison of BatchNorm and LayerNorm on IMDb sentiment analysis
    """)

def fashionmnist():
    """Training a simple MLP on the FashionMNIST dataset and displaying the metrics evolution during the training.

    The user can decide the number of hidden layers of the MLP. They can also choose the number of epochs
    for training. Once a model with given hyperparameters is trained, it is saved and used
    again the next times without new training, unless the user clicks the button to delete
    the saved model and train again. The MLP architecture is displayed.
    Then 2 figures that are the evolution
    of, respectively, the losses (train and test) and accuracies (train and test)
    with respect to the epoch, are displayed. Finally 6 random images of the test dataset are
    displayed, along with their ground truth and predicted labels.

    Returns
    -------
    None

    Notes
    -----
    Inspired by https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.

    """
    st.title("MLP Training on FashionMNIST with Normalization Techniques")
    
    st.info("This section allows you to compare BatchNorm and no BatchNorm simultaneously across multiple learning rates.")
    st.info("Pretrained models are automatically downloaded if not available locally."
)
    hidden_layers = st.slider("Choose the number of hidden layers", 1, 5, 2)

    dropout_rate = st.slider("Choose the dropout rate", 0.0, 0.9, 0.0, 0.1)

    compare_bn = st.checkbox("Compare BatchNorm vs no BatchNorm", value=True)
    
    compare_lr = st.checkbox("Compare multiple learning rates", value=True)

    if not compare_bn:
        use_bn = st.checkbox("Use BatchNorm", value=True)
        norm_type = "batchnorm" if use_bn else "none"
    else:
        norm_type = None

    if not compare_lr:
        lr = st.selectbox(
            "Learning rate",
            [1e-4, 1e-3, 1e-2],
            index=1
        )
    else:
        st.write("Comparing learning rates: 1e-4, 1e-3, 1e-2")
        lr = None

    epochs = st.slider("Choose the number of epochs to train", 1, 50, 10) 
    force_train = st.checkbox("Force retrain (ignore local and remote pretrained model)", value=False)
    st.caption("This only deletes the local copy. The model can still be downloaded again.")
    if st.button("Delete local copy (will be re-downloaded)"):
        if compare_bn and compare_lr:
            for norm_i in ["none", "batchnorm"]:
                for lr_i in [1e-4, 1e-3, 1e-2]:
                    path_weights, path_metrics = paths(
                        hidden_layers,
                        dropout_rate,
                        norm_type=norm_i,
                        epochs=epochs,
                        lr=lr_i,
                    )
                    try:
                        os.remove(path_weights)
                        os.remove(path_metrics)
                    except FileNotFoundError:
                        pass

        elif compare_bn and not compare_lr:
            for norm_i in ["none", "batchnorm"]:
                path_weights, path_metrics = paths(
                    hidden_layers,
                    dropout_rate,
                    norm_type=norm_i,
                    epochs=epochs,
                    lr=lr,
                )
                try:
                    os.remove(path_weights)
                    os.remove(path_metrics)
                except FileNotFoundError:
                    pass

        elif not compare_bn and compare_lr:
            for lr_i in [1e-4, 1e-3, 1e-2]:
                path_weights, path_metrics = paths(
                    hidden_layers,
                    dropout_rate,
                    norm_type=norm_type,
                    epochs=epochs,
                    lr=lr_i,
                )
                try:
                    os.remove(path_weights)
                    os.remove(path_metrics)
                except FileNotFoundError:
                    pass

        else:
            path_weights, path_metrics = paths(
                hidden_layers,
                dropout_rate,
                norm_type=norm_type,
                epochs=epochs,
                lr=lr,
            )
            try:
                os.remove(path_weights)
                os.remove(path_metrics)
            except FileNotFoundError:
                pass

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    
    train_clicked = st.button("Train / Load model") 

    if train_clicked:
        st.session_state.stop_training = False

    # If user didn't click train, do nothing
    if not train_clicked:
        st.stop()

    train_dataloader, test_dataloader, _, test_data = dl.get_FashionMNIST_datasets(
        64, only_loader=False
    )

    if not compare_bn and not compare_lr:
    # One single model
        with st.spinner("Training or loading model..."):
            model = dl.get_and_train_model(
                train_dataloader,
                test_dataloader,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                lr=lr,
                epochs=epochs,
                mode="st",
                force_train=force_train,
            )

        training_curves(model, "st")
        mnist_like_viz(test_data, classes, model)

    elif not compare_bn and compare_lr:
        # One normalization setting, compared across several learning rates
        models = []
        lr_list = [1e-4, 1e-3, 1e-2]

        compare_bar = st.progress(0)
        compare_status = st.empty()

        for i, lr_i in enumerate(lr_list):
            compare_status.write(
                f"Training / loading model {i + 1}/{len(lr_list)} "
                f"with learning rate = {lr_i}"
            )

            model_i = dl.get_and_train_model(
                train_dataloader,
                test_dataloader,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                lr=lr_i,
                epochs=epochs,
                mode="st",
                progress_prefix=f"[lr={lr_i}] ",
                force_train=force_train,
            )

            models.append((f"lr={lr_i}", model_i))
            compare_bar.progress((i + 1) / len(lr_list))

        compare_status.write("All models loaded / trained.")
        compare_training_curves(models, "st")

    elif compare_bn and not compare_lr:
        # Compare no BN vs BN for one fixed learning rate
        models = []

        compare_bar = st.progress(0)
        compare_status = st.empty()

        norm_list = [("No BatchNorm", "none"), ("BatchNorm", "batchnorm")]

        for i, (label, norm_i) in enumerate(norm_list):
            compare_status.write(
                f"Training / loading model {i + 1}/{len(norm_list)} "
                f"with normalization = {label}"
            )

            model_i = dl.get_and_train_model(
                train_dataloader,
                test_dataloader,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                norm_type=norm_i,
                lr=lr,
                epochs=epochs,
                mode="st",
                progress_prefix=f"[{label}] ",
                force_train=force_train,
            )

            models.append((label, model_i))
            compare_bar.progress((i + 1) / len(norm_list))

        compare_status.write("All models loaded / trained.")
        compare_training_curves(models, "st")

    elif compare_bn and compare_lr:
        # Compare learning rates separately for no BN and BN
        lr_list = [1e-4, 1e-3, 1e-2]
        total_models = 6
        done = 0

        compare_bar = st.progress(0)
        compare_status = st.empty()

        models_no_bn = []
        models_bn = []

        for lr_i in lr_list:
            compare_status.write(f"Training / loading No BatchNorm model with lr = {lr_i}")

            model_i = dl.get_and_train_model(
                train_dataloader,
                test_dataloader,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                norm_type="none",
                lr=lr_i,
                epochs=epochs,
                mode="st",
                progress_prefix=f"[no BN, lr={lr_i}] ",
                force_train=force_train,
            )

            models_no_bn.append((f"lr={lr_i}", model_i))
            done += 1
            compare_bar.progress(done / total_models)

        for lr_i in lr_list:
            compare_status.write(f"Training / loading BatchNorm model with lr = {lr_i}")

            model_i = dl.get_and_train_model(
                train_dataloader,
                test_dataloader,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                norm_type="batchnorm",
                lr=lr_i,
                epochs=epochs,
                mode="st",
                progress_prefix=f"[BN, lr={lr_i}] ",
                force_train=force_train,
            )

            models_bn.append((f"lr={lr_i}", model_i))
            done += 1
            compare_bar.progress(done / total_models)

        compare_status.write("All models loaded / trained.")

        st.subheader("Without BatchNorm: effect of learning rate")
        compare_training_curves(models_no_bn, "st")

        st.subheader("With BatchNorm: effect of learning rate")
        compare_training_curves(models_bn, "st")

    
def imdb_rnn_page():
    st.title("RNN Sentiment Analysis with Normalization (IMDb)")
    st.header("RNN sentiment analysis on IMDb")
    
    st.info("""
            This page compares BatchNorm and LayerNorm in an RNN for sentiment analysis.

            The normalization is applied to the final hidden state of the RNN.
            You can observe that LayerNorm leads to better performance than BatchNorm in fewer steps.
            """)
    st.caption("LayerNorm is typically better suited for sequence models because it does not rely on batch statistics.")
    st.info("Pretrained models are automatically downloaded if not available locally.")
    
    embedding_dim = st.slider("Embedding dimension", 16, 128, 64, 16)
    hidden_dim = st.slider("Hidden dimension", 32, 256, 128, 32)
    dropout_rate = st.slider("Dropout rate", 0.0, 0.9, 0.0, 0.1)
    lr = st.selectbox("Learning rate", [1e-4, 1e-3, 1e-2], index=1)
    epochs = st.slider("Epochs", 1, 20, 5)
    max_len = st.slider("Maximum sequence length", 50, 400, 200, 50)

    train_size = st.slider("Train subset size", 500, 5000, 2000, 500)
    test_size = st.slider("Test subset size", 200, 2000, 1000, 200)

    force_train = st.checkbox("Force retrain (ignore local and remote pretrained model)", value=False)

    st.caption("This only deletes the local copy. The model can still be downloaded again.")

    if st.button("Delete local copy (will be re-downloaded)"):
        for norm_type in ["batchnorm", "layernorm"]:
            path_weights, path_metrics = paths_rnn(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                epochs=epochs,
                lr=lr,
                max_len=max_len,
                train_size=train_size,
            )
            try:
                os.remove(path_weights)
                os.remove(path_metrics)
            except FileNotFoundError:
                pass

    train_clicked = st.button("Train / Load RNN models")
    if not train_clicked:
        st.stop()

    train_dataloader, test_dataloader, _, _, vocab = dl.get_IMDB_datasets(
        batch_size=64,
        max_len=max_len,
        train_size=train_size,
        test_size=test_size,
        only_loader=False,
    )

    models = []

    for norm_type in ["batchnorm", "layernorm"]:
        st.write(f"Training / loading model with {norm_type}")
        model = dl.get_and_train_rnn_model(
            train_dataloader,
            test_dataloader,
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            lr=lr,
            epochs=epochs,
            max_len=max_len,
            train_size=train_size,
            mode="st",
            progress_prefix=f"[{norm_type}] ",
            force_train=force_train,
        )
        models.append((norm_type, model))

    compare_training_curves(models, "st")

if __name__ == "__main__":
    main()
