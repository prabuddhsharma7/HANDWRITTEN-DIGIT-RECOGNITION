"""
train.py
--------
Trains two simple, classic ML models on the handwritten digits dataset
and compares them:

1. Support Vector Machine (SVM)      -- strong classical baseline
2. Multi-Layer Perceptron (MLP)      -- a small neural network

This is meant to demonstrate core AI/ML fundamentals end-to-end:
    load data -> split -> train -> evaluate -> visualize -> save model

Run with:
    python src/train.py
"""

import os
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import load_data, plot_sample_digits, plot_confusion_matrix, plot_predictions

# ---- paths -----------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    """Train a model, time it, and print/return evaluation metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Training time : {elapsed:.2f}s")
    print(f"Test accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    return model, y_pred, acc


def main():
    print("Loading the handwritten digits dataset (1,797 images, 8x8 pixels)...")
    X_train, X_test, y_train, y_test, images_test = load_data()
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Visualize a handful of raw samples before training anything
    plot_sample_digits(
        images_test, y_test,
        save_path=os.path.join(IMAGES_DIR, "sample_digits.png"),
    )

    # ---- Model 1: SVM --------------------------------------------------
    svm_model = SVC(kernel="rbf", gamma=0.001, C=10, probability=False)
    svm_model, svm_pred, svm_acc = train_and_evaluate(
        "SVM (RBF kernel)", svm_model, X_train, y_train, X_test, y_test
    )

    # ---- Model 2: Small Neural Network (MLP) ----------------------------
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    mlp_model, mlp_pred, mlp_acc = train_and_evaluate(
        "Neural Network (MLP)", mlp_model, X_train, y_train, X_test, y_test
    )

    # ---- Pick the better model to save & visualize in detail -----------
    if svm_acc >= mlp_acc:
        best_name, best_model, best_pred, best_acc = "SVM", svm_model, svm_pred, svm_acc
    else:
        best_name, best_model, best_pred, best_acc = "MLP", mlp_model, mlp_pred, mlp_acc

    print(f"\nBest model: {best_name} (accuracy: {best_acc:.4f})")

    cm = confusion_matrix(y_test, best_pred)
    plot_confusion_matrix(cm, save_path=os.path.join(IMAGES_DIR, "confusion_matrix.png"))
    plot_predictions(
        images_test, y_test, best_pred,
        save_path=os.path.join(IMAGES_DIR, "sample_predictions.png"),
    )

    model_path = os.path.join(MODELS_DIR, "digit_classifier.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to: {model_path}")
    print("Run `python src/predict.py` to test it on new samples.")


if __name__ == "__main__":
    main()
