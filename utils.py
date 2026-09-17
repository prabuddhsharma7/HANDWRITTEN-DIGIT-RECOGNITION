"""
utils.py
--------
Shared helper functions for loading data and plotting results.

The project uses scikit-learn's built-in `load_digits` dataset:
1,797 images of handwritten digits (0-9), each 8x8 pixels.
It ships with scikit-learn, so no external download is needed --
the project works immediately after `pip install -r requirements.txt`.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    """Load the digits dataset and split it into train/test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Flattened 64-pixel feature vectors and integer labels (0-9).
    images_test : array
        The un-flattened 8x8 test images, kept around purely so we can
        visualize predictions later.
    """
    digits = load_digits()
    X = digits.data          # shape (n_samples, 64) -- flattened 8x8 images
    y = digits.target        # shape (n_samples,)    -- labels 0-9
    images = digits.images   # shape (n_samples, 8, 8) -- for plotting

    X_train, X_test, y_train, y_test, _, images_test = train_test_split(
        X, y, images, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, images_test


def plot_sample_digits(images, labels, save_path=None, n=10):
    """Plot the first `n` digit images with their labels."""
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray_r")
        ax.set_title(str(labels[i]), fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample digits from the dataset", y=1.05)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_confusion_matrix(cm, save_path=None):
    """Plot a confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_predictions(images, y_true, y_pred, save_path=None, n=10):
    """Plot predicted vs. true labels for the first `n` test images.
    Wrong predictions are highlighted in red.
    """
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.8))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray_r")
        correct = y_true[i] == y_pred[i]
        color = "green" if correct else "red"
        ax.set_title(f"P:{y_pred[i]}\nT:{y_true[i]}", fontsize=9, color=color)
        ax.axis("off")
    fig.suptitle("Predictions (P) vs True labels (T)", y=1.08)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
