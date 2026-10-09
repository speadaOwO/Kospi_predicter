import numpy as np
import pandas as pd
import torch



X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_valid = np.load("data/processed/X_valid.npy")
y_valid = np.load("data/processed/y_valid.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")


for name, y in [
    ("train", y_train),
    ("valid", y_valid),
    ("test", y_test),
]:
    print(
        name,
        "mean:", y.astype(np.float64).mean().item(),
        "std:", y.astype(np.float64).std().item(),
        "min:", y.astype(np.float64).min().item(),
        "max:", y.astype(np.float64).max().item(),
        "positive:", (y > 0).astype(np.float64).mean().item(),
        "negative:", (y < 0).astype(np.float64).mean().item(),
        "zero:", (y == 0).astype(np.float64).mean().item(),
    )

    


