
"""KOSPI next-day return - 1D CNN.
Run: python src/train_cnn.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "training_run_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
seed = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "seed": seed,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "early_stopping_patience": 14,
    "scheduler_patience": 4,
    "min_learning_rate": 1e-6,
    "gradient_clip_norm": 1.0,
    "channels": 64,
    "kernel_size": 3,
    "dropout": 0.15,
    "huber_delta": 1.0,
    "num_workers": 0,
    "use_amp": True,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(DATA_DIR / f"X_{name}.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"y_{name}.npy").astype(np.float32).reshape(-1)
    direction = np.load(
        DATA_DIR / f"y_{name}_direction.npy"
    ).astype(np.float32).reshape(-1)

    print(f"{name}: X={X.shape}, y={y.shape}, up={direction.mean():.2%}")

    if len(X) != len(y) or len(y) != len(direction):
        raise ValueError(f"{name}: X, y, direction 길이가 다릅니다.")

    return X, y, direction


def load_zero_threshold() -> float:
    path = DATA_DIR / "scaler_y.pkl"

    if not path.exists():
        return 0.0

    scaler = joblib.load(path)
    return -float(scaler.mean_[0]) / float(scaler.scale_[0])


def make_loader(X, y, direction, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(direction),
    )

    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
    )


class CNNRegressor(nn.Module):
    def __init__(self, input_dim: int, channels: int, dropout: float):
        super().__init__()

        # Conv1d 입력 형식: (batch, features, time)
        self.features = nn.Sequential(
            nn.Conv1d(
                input_dim, channels,
                kernel_size=3, padding=1
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                channels, channels,
                kernel_size=5, padding=2
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            # 더 넓은 시간 구간의 패턴도 포착
            nn.Conv1d(
                channels, channels,
                kernel_size=3, padding=2, dilation=2
            ),
            nn.GELU(),
        )

        # 평균 풀링 + 최대 풀링
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.features(x)   # (B, C, T)

        avg = self.avg_pool(x).squeeze(-1)  # (B, C)
        mx = self.max_pool(x).squeeze(-1)   # (B, C)

        x = torch.cat([avg, mx], dim=1)     # (B, 2C)
        return self.head(x).squeeze(-1)     # (B,)


@torch.no_grad()
def evaluate(model, loader, criterion, zero_th: float) -> Dict:
    model.eval()

    loss_sum = 0.0
    n = 0
    ys, preds, dirs = [], [], []

    for X, y, d in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(X)

        loss_sum += criterion(pred, y).item() * len(y)
        n += len(y)

        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
        dirs.append(d.numpy().astype(np.int64))

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    actual_dir = np.concatenate(dirs)

    pred_dir = (y_pred > zero_th).astype(np.int64)

    return {
        "loss": loss_sum / max(n, 1),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "sign_acc": float(accuracy_score(actual_dir, pred_dir)),
        "y_true": y_true,
        "y_pred": y_pred,
        "actual_dir": actual_dir,
        "pred_dir": pred_dir,
    }


def save_plots(history, best_epoch):
    df = pd.DataFrame(history)
    df.to_csv(OUTPUT_DIR / f"training_history120{seed}.csv", index=False)

    epochs = df["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(epochs, df["train_loss"], label="Train")
    ax.plot(epochs, df["valid_loss"], label="Validation")
    ax.axvline(best_epoch, linestyle="--", alpha=0.7,
               label=f"Best epoch: {best_epoch}")
    ax.set_title("CNN: Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, df["valid_sign_acc"] * 100)
    ax.set_title("Validation Direction Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, df["lr"])
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, df["valid_mae"])
    ax.set_title("Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"training_curves120{seed}.png", dpi=160)
    plt.close(fig)

    print(f"Plots saved: {OUTPUT_DIR / f'training_curves120{seed}.png'}")


def main():
    set_seed(CONFIG["seed"])
    print(f"Device: {DEVICE}")

    X_tr, y_tr, d_tr = load_split("train")
    X_va, y_va, d_va = load_split("valid")
    X_te, y_te, d_te = load_split("test")

    input_dim = X_tr.shape[-1]
    window = X_tr.shape[1]

    print(f"Features={input_dim}, window={window}")

    # Ensure the same input dimensions across splits.
    if X_va.shape[1:] != X_tr.shape[1:]:
        raise ValueError("Validation 입력 shape이 train과 다릅니다.")
    if X_te.shape[1:] != X_tr.shape[1:]:
        raise ValueError("Test 입력 shape이 train과 다릅니다.")

    train_loader = make_loader(X_tr, y_tr, d_tr, True)
    valid_loader = make_loader(X_va, y_va, d_va, False)
    test_loader = make_loader(X_te, y_te, d_te, False)

    model = CNNRegressor(
        input_dim=input_dim,
        channels=CONFIG["channels"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["min_learning_rate"],
    )

    amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    zero_th = load_zero_threshold()
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    t0 = time.time()

    best_path = OUTPUT_DIR / "best_cnn_model.pth"

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        n = 0

        for X, y, _ in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                pred = model(X)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG["gradient_clip_norm"],
            )

            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * len(y)
            n += len(y)

        train_loss = loss_sum / max(n, 1)

        valid = evaluate(model, valid_loader, criterion, zero_th)
        scheduler.step(valid["loss"])
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid["loss"],
            "valid_mae": valid["mae"],
            "valid_sign_acc": valid["sign_acc"],
            "lr": lr,
        })

        print(
            f"Ep {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val={valid['loss']:.4f} "
            f"MAE={valid['mae']:.4f} "
            f"Sign={valid['sign_acc'] * 100:.1f}% | "
            f"lr={lr:.1e}"
        )

        if valid["loss"] < best_loss - 1e-5:
            best_loss = valid["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
            print("  -> best CNN saved")
        else:
            stale += 1

            if stale >= CONFIG["early_stopping_patience"]:
                print("Early stopping.")
                break

    save_plots(history, best_epoch)

    




    print(
        f"\nDone in {(time.time() - t0) / 60:.1f} min "
        f"| best ep={best_epoch} loss={best_loss:.5f}"
    )

    # Evaluate only the best validation-loss checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test = evaluate(model, test_loader, criterion, zero_th)

    print("\n===== CNN TEST =====")
    print(f"Samples     : {len(test['y_true'])}")
    print(f"Huber loss  : {test['loss']:.5f}")
    print(f"MAE         : {test['mae']:.5f}")
    print(f"seed = {seed}")
    print(f"Sign acc    : {test['sign_acc'] * 100:.2f}%")
    print(confusion_matrix(
        test["actual_dir"], test["pred_dir"], labels=[0, 1]
    ))
    print(f"Always-up   : {(test['actual_dir'] == 1).mean() * 100:.2f}%")

    pd.DataFrame({
        "actual_return_scaled": test["y_true"],
        "predicted_return_scaled": test["y_pred"],
        "actual_direction": test["actual_dir"],
        "predicted_direction": test["pred_dir"],
    }).to_csv(OUTPUT_DIR / f"test_predictions120{seed}.csv", index=False)

    with open(OUTPUT_DIR / f"metrics120{seed}.json", "w") as f:
        json.dump({
            "model": "CNN",
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "test_loss": test["loss"],
            "test_mae": test["mae"],
            "test_sign_acc": test["sign_acc"],
            "config": CONFIG,
        }, f, indent=2)

    print(f"\nSaved -> {OUTPUT_DIR}")

plt.show()
if __name__ == "__main__":
    main()
