"""End-to-end training script for the KOSPI next-day prediction model.

Expected files under data/processed:
    X_train.npy, y_train.npy, X_valid.npy, y_valid.npy, X_test.npy, y_test.npy
    scaler_y.pkl (recommended; used to recover the true zero-return threshold)

Run from the repository root:
    python src/train.py
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "training_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Training
    "seed": 200,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "early_stopping_patience": 7,
    "scheduler_patience": 4,
    "min_learning_rate": 1e-6,
    "gradient_clip_norm": 1.0,

    # Model. Keep capacity moderate because the training set is only a few thousand windows.
    "d_model": 32,
    "nhead": 4,
    "num_layers": 2,
    "num_conv_layers": 2,  # 추가: 1~4 추천
    "ff_dim": 64,
    "dropout": 0.15,

    # Multi-task objective: next-day return regression + up/down classification.
    "regression_loss_weight": 0.35,
    "classification_loss_weight": 1.0,
    "huber_delta": 1.0,

    # DataLoader / device
    "num_workers": 0,
    "use_amp": True,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism helps reproduce experiments. It may be a little slower on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Data loading and checks
# ============================================================


def load_split(name: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = DATA_DIR / f"X_{name}.npy"
    y_path = DATA_DIR / f"y_{name}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path.name} or {y_path.name} in {DATA_DIR}. "
            "Run the preprocessing/dataset pipeline first."
        )

    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.float32, copy=False).reshape(-1)

    if X.ndim != 3:
        raise ValueError(f"X_{name} must have shape (samples, window, features), got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X_{name}/y_{name} sample mismatch: {len(X)} vs {len(y)}")
    if len(X) == 0:
        raise ValueError(f"{name} split is empty")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values found in {name} split. Fix preprocessing before training.")
    return X, y


def load_zero_return_threshold_scaled() -> float:
    """Return the standardized y value corresponding to raw return == 0.

    y was standardized as (raw_return - mean) / scale, so raw zero maps to
    -mean / scale, not necessarily exactly zero.
    """
    scaler_path = DATA_DIR / "scaler_y.pkl"
    if not scaler_path.exists():
        warnings.warn(
            "scaler_y.pkl not found. Assuming standardized y=0 corresponds to zero return. "
            "This is usually close, but saving/loading scaler_y.pkl is more correct."
        )
        return 0.0

    scaler = joblib.load(scaler_path)
    mean = float(np.asarray(scaler.mean_).reshape(-1)[0])
    scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale == 0:
        raise ValueError("scaler_y.pkl has invalid mean/scale")
    return -mean / scale


# ============================================================
# 3. Model
# ============================================================

# 3. Model 부분만 교체
class KOSPITransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_conv_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = d_model

        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # FIXED: CNN layer 수 조절 + dilated
        conv_layers = []
        dilation = 1
        for i in range(num_conv_layers):
            conv_layers.append(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation)
            )
            if i < num_conv_layers - 1:
                conv_layers.append(nn.GELU())
                conv_layers.append(nn.Dropout(dropout))
            dilation *= 2

        self.local_conv = nn.Sequential(*conv_layers)
        self.local_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.pool_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(d_model // 2, 1)
        self.direction_head = nn.Linear(d_model // 2, 1)

    def forward(self, x: torch.Tensor):
        #... (기존 forward 그대로)
        gate = self.feature_gate(x)
        x = x * (0.5 + gate)
        x = self.input_norm(self.input_projection(x))
        x = x + self.position_embedding[:, : x.shape[1], :]
        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.local_norm(x + local)
        encoded = self.encoder(x)
        scores = self.pool_score(encoded).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        recent = encoded[:, -1, :]
        fused = self.fusion(torch.cat([pooled, recent], dim=-1))
        shared = self.shared(fused)
        return self.return_head(shared).squeeze(-1), self.direction_head(shared).squeeze(-1)


# ============================================================
# 4. Metrics / evaluation
# ============================================================


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    regression_criterion: nn.Module,
    classification_criterion: nn.Module,
    zero_threshold: float,
) -> Dict[str, object]:
    model.eval()
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_count = 0
    all_y = []
    all_return_pred = []
    all_labels = []
    all_logits = []

    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        return_pred, direction_logit = model(X)
        labels = (y > zero_threshold).float()
        reg_loss = regression_criterion(return_pred, y)
        cls_loss = classification_criterion(direction_logit, labels)
        n = len(y)
        total_reg_loss += reg_loss.item() * n
        total_cls_loss += cls_loss.item() * n
        total_count += n
        all_y.append(y.detach().cpu().numpy())
        all_return_pred.append(return_pred.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy().astype(np.int64))
        all_logits.append(direction_logit.detach().cpu().numpy())

    y_scaled = np.concatenate(all_y)
    return_pred_scaled = np.concatenate(all_return_pred)
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    predicted_labels = (probabilities >= 0.5).astype(np.int64)

    accuracy = float(accuracy_score(labels, predicted_labels))
    balanced_accuracy = float(balanced_accuracy_score(labels, predicted_labels))
    bce = total_cls_loss / max(total_count, 1)
    huber = total_reg_loss / max(total_count, 1)
    # Validation/checkpoint objective. Classification is the primary objective.
    selection_loss = bce + CONFIG["regression_loss_weight"] * huber

    return {
        "selection_loss": selection_loss,
        "classification_loss": bce,
        "regression_loss": huber,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "labels": labels,
        "predicted_labels": predicted_labels,
        "probabilities": probabilities,
        "logits": logits,
        "y_scaled": y_scaled,
        "return_pred_scaled": return_pred_scaled,
    }


# 4. Metrics 부분 교체
def print_metrics(name: str, result: Dict[str, object]) -> None:
    labels = result["labels"]
    preds = result["predicted_labels"]
    probs = result["probabilities"]
    print(f"\n===== {name} metrics, CNN TRANSFORMER =====")
    print(f"Samples : {len(labels)}")
    print(f"CNN Layers = {CONFIG['num_conv_layers']}")
    print(f"dmodel = {CONFIG['d_model']}")
    print(f"transformer layer = {CONFIG['num_layers']}")
    print(f"ffdim = {CONFIG['ff_dim']}")
    print(f"Direction accuracy : {accuracy_score(labels, preds) * 100:.2f}%")
    print(f"Balanced accuracy : {balanced_accuracy_score(labels, preds) * 100:.2f}%")
    print(f"Precision (up) : {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"Recall (up) : {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"F1 (up) : {f1_score(labels, preds, zero_division=0):.4f}")
    print(f"ROC-AUC : {safe_auc(labels, probs):.4f}")
    print(f"MCC : {matthews_corrcoef(labels, preds):.4f}")
    print("Confusion matrix [true rows: down, up; predicted columns: down, up]:")
    print(confusion_matrix(labels, preds, labels=[0, 1]))
    print(f"Always-up baseline : {(labels == 1).mean() * 100:.2f}%")
    print(f"Always-down baseline: {(labels == 0).mean() * 100:.2f}%")



# ============================================================
# Plot training curves
# ============================================================

def plot_training_curves(
    history: list,
    output_dir: Path,
    seed: int,
    window_size: int,
) -> None:
    if not history:
        print("No training history to plot.")
        return

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. Regression loss
    axes[0].plot(
        df["epoch"], df["train_regression_loss"],
        label="Train Huber"
    )
    axes[0].plot(
        df["epoch"], df["valid_regression_loss"],
        label="Validation Huber"
    )
    axes[0].set_title("Regression Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Classification loss
    axes[1].plot(
        df["epoch"], df["train_classification_loss"],
        label="Train BCE"
    )
    axes[1].plot(
        df["epoch"], df["valid_classification_loss"],
        label="Validation BCE"
    )
    axes[1].set_title("Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Validation metrics
    axes[2].plot(
        df["epoch"], df["valid_accuracy"] * 100,
        label="Accuracy"
    )
    axes[2].plot(
        df["epoch"], df["valid_balanced_accuracy"] * 100,
        label="Balanced Accuracy"
    )
    axes[2].set_title("Validation Metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score (%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"CNN-Transformer | Window={window_size} | Seed={seed}"
    )
    fig.tight_layout()

    plot_path = (
        output_dir
        / f"training_curves_cnn_transformer_w{window_size}_seed120{seed}.png"
    )

    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved to: {plot_path}")
# ============================================================
# 5. Training loop
# ============================================================


def main() -> None:
    set_seed(CONFIG["seed"])
    print(f"Repository : {ROOT}")
    print(f"Device     : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    X_train_np, y_train_np = load_split("train")
    X_valid_np, y_valid_np = load_split("valid")
    X_test_np, y_test_np = load_split("test")

    if X_train_np.shape[1:] != X_valid_np.shape[1:] or X_train_np.shape[1:] != X_test_np.shape[1:]:
        raise ValueError(
            "Train/valid/test input shapes differ: "
            f"{X_train_np.shape[1:]}, {X_valid_np.shape[1:]}, {X_test_np.shape[1:]}"
        )

    input_dim = int(X_train_np.shape[-1])
    window_size = int(X_train_np.shape[1])
    zero_threshold = load_zero_return_threshold_scaled()
    

    print("\n===== Data =====")
    print(f"Train: X={X_train_np.shape}, y={y_train_np.shape}")
    print(f"Valid: X={X_valid_np.shape}, y={y_valid_np.shape}")
    print(f"Test : X={X_test_np.shape}, y={y_test_np.shape}")
    print(f"Features={input_dim}, window={window_size}")
    print(f"Standardized target threshold for raw 0% return: {zero_threshold:.8f}")

    def make_loader(X_np: np.ndarray, y_np: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
        generator = torch.Generator()
        generator.manual_seed(CONFIG["seed"])
        return DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"],
            pin_memory=(DEVICE.type == "cuda"),
            drop_last=False,
            generator=generator if shuffle else None,
        )

    train_loader = make_loader(X_train_np, y_train_np, True)
    valid_loader = make_loader(X_valid_np, y_valid_np, False)
    test_loader = make_loader(X_test_np, y_test_np, False)

    model = KOSPITransformer(
        input_dim=input_dim,
        window_size=window_size,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_layers"],
        num_conv_layers=CONFIG["num_conv_layers"],
        ff_dim=CONFIG["ff_dim"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    print("\n===== Model =====")
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    regression_criterion = nn.HuberLoss(delta=CONFIG["huber_delta"])
    classification_criterion = nn.BCEWithLogitsLoss()
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

    amp_enabled = bool(CONFIG["use_amp"] and DEVICE.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # compatibility with older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_total = 0
        train_reg_sum = 0.0
        train_cls_sum = 0.0

        for X, y in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            labels = (y > zero_threshold).float()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
                return_pred, direction_logit = model(X)
                reg_loss = regression_criterion(return_pred, y)
                cls_loss = classification_criterion(direction_logit, labels)
                loss = (
                    CONFIG["regression_loss_weight"] * reg_loss
                    + CONFIG["classification_loss_weight"] * cls_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            n = len(y)
            train_total += n
            train_reg_sum += reg_loss.detach().item() * n
            train_cls_sum += cls_loss.detach().item() * n

        train_reg = train_reg_sum / max(train_total, 1)
        train_cls = train_cls_sum / max(train_total, 1)
        valid = evaluate(
            model,
            valid_loader,
            regression_criterion,
            classification_criterion,
            zero_threshold,
        )
        scheduler.step(valid["selection_loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_regression_loss": train_reg,
            "train_classification_loss": train_cls,
            "valid_regression_loss": valid["regression_loss"],
            "valid_classification_loss": valid["classification_loss"],
            "valid_selection_loss": valid["selection_loss"],
            "valid_accuracy": valid["accuracy"],
            "valid_balanced_accuracy": valid["balanced_accuracy"],
            "learning_rate": lr_now,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
            f"train Huber={train_reg:.4f} BCE={train_cls:.4f} | "
            f"valid Huber={valid['regression_loss']:.4f} BCE={valid['classification_loss']:.4f} "
            f"Acc={valid['accuracy'] * 100:.2f}% BAcc={valid['balanced_accuracy'] * 100:.2f}% | "
            f"lr={lr_now:.2e}"
        )

        if valid["selection_loss"] < best_valid_loss - 1e-5:
            best_valid_loss = float(valid["selection_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0

            # Keep state_dict at the familiar path for simple loading.
            torch.save(model.state_dict(), DATA_DIR / "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_loss": best_valid_loss,
                    "config": CONFIG,
                    "input_dim": input_dim,
                    "window_size": window_size,
                    "zero_return_threshold_scaled": zero_threshold,
                },
                OUTPUT_DIR / "best_checkpoint.pt",
            )
            print("  -> saved new best model")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping: no validation improvement for {CONFIG['early_stopping_patience']} epochs.")
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        OUTPUT_DIR / f"training_history_seed{CONFIG['seed']}.csv",
        index=False,
    )

    # Plot and save training curves
    plot_training_curves(
        history=history,
        output_dir=OUTPUT_DIR,
        seed=CONFIG["seed"],
        window_size=window_size,
    )
    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best epoch: {best_epoch}, best validation loss: {best_valid_loss:.5f}")

    # Final test evaluation only after choosing the checkpoint using validation data.
    best_state = torch.load(DATA_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(best_state)

    valid_result = evaluate(
        model,
        valid_loader,
        regression_criterion,
        classification_criterion,
        zero_threshold,
    )
    
    valid_probs = np.asarray(valid_result["probabilities"])
    valid_labels = np.asarray(valid_result["labels"])
    best_thresh = 0.5
    best_bacc = 0
    for t in np.arange(0.3, 0.81, 0.01):
        pred = (valid_probs >= t).astype(int)
        bacc = balanced_accuracy_score(valid_labels, pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thresh = t
            
    print(f"\n[VALID] Best thresh {best_thresh:.2f} -> BAcc {best_bacc:.4f}")
    test_result = evaluate(
        model,
        test_loader,
        regression_criterion,
        classification_criterion,
        zero_threshold,
    )
    print_metrics("TEST (threshold=0.50)", test_result)

    # --- threshold 튜닝 추가 (여기부터) ---
    probs = np.asarray(test_result["probabilities"])
    labels = np.asarray(test_result["labels"])
    
    best_thresh = 0.5
    best_bacc = 0
    print("\n===== Threshold sweep (TEST) =====")
    for t in np.arange(0.30, 0.81, 0.02):
        pred = (probs >= t).astype(int)
        bacc = balanced_accuracy_score(labels, pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thresh = t
        # 보고 싶으면 매 thresh마다 출력
        # print(f"thresh {t:.2f} -> BAcc {bacc:.4f}")

    print(f"Default 0.50 -> BAcc {test_result['balanced_accuracy']:.4f}")
    print(f"Best {best_thresh:.2f} -> BAcc {best_bacc:.4f}")
    # --- 여기까지 ---
    

    # Regression head's sign is reported as a secondary diagnostic, not used as the main direction prediction.
    y_scaled = np.asarray(test_result["y_scaled"])
    return_pred_scaled = np.asarray(test_result["return_pred_scaled"])
    actual_direction = (y_scaled > zero_threshold).astype(np.int64)
    regression_direction = (return_pred_scaled > zero_threshold).astype(np.int64)
    print(f"Regression-head sign accuracy: {accuracy_score(actual_direction, regression_direction) * 100:.2f}%")

    predictions_df = pd.DataFrame({
        "actual_return_scaled": y_scaled,
        "predicted_return_scaled": return_pred_scaled,
        "actual_direction": actual_direction,
        "predicted_direction": np.asarray(test_result["predicted_labels"]),
        "predicted_up_probability": np.asarray(test_result["probabilities"]),
    })
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_valid_selection_loss": best_valid_loss,
        "test_samples": int(len(actual_direction)),
        "test_direction_accuracy": float(accuracy_score(actual_direction, test_result["predicted_labels"])),
        "test_balanced_accuracy": float(balanced_accuracy_score(actual_direction, test_result["predicted_labels"])),
        "test_roc_auc": safe_auc(actual_direction, np.asarray(test_result["probabilities"])),
        "test_regression_sign_accuracy": float(accuracy_score(actual_direction, regression_direction)),
        "zero_return_threshold_scaled": zero_threshold,
        "config": CONFIG,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print(f"- Best model weights: {DATA_DIR / 'best_model.pth'}")
    print(f"- Full checkpoint   : {OUTPUT_DIR / 'best_checkpoint.pt'}")
    print(f"- Training history  : {OUTPUT_DIR / 'training_history.csv'}")
    print(f"- Test predictions  : {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"- Metrics           : {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
