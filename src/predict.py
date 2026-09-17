"""
predict.py
----------
Loads the trained model saved by train.py and uses it to predict
digits from the test set, printing a few example predictions.

Run with:
    python src/predict.py
"""

import os
import sys
import joblib
import numpy as np

from utils import load_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "digit_classifier.joblib")


def main():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run `python src/train.py` first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Reuse the same split so we predict on unseen test data
    _, X_test, _, y_test, images_test = load_data()

    n = min(10, len(X_test))
    sample_X = X_test[:n]
    sample_y = y_test[:n]

    predictions = model.predict(sample_X)

    print(f"\nPredicting {n} sample digits:\n")
    correct = 0
    for i in range(n):
        is_correct = predictions[i] == sample_y[i]
        correct += is_correct
        status = "OK" if is_correct else "WRONG"
        print(f"  Sample {i+1:2d}: predicted={predictions[i]}  actual={sample_y[i]}  [{status}]")

    print(f"\n{correct}/{n} correct on this sample batch.")


if __name__ == "__main__":
    main()
