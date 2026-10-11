"""KOSPI next-day return - bare Transformer. Run: python src/train.py"""
from __future__ import annotations
import json, math, random, time, warnings
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error , roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
seed = 100
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "training_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "seed": seed,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "early_stopping_patience": 7,
    "scheduler_patience": 4,
    "min_learning_rate": 1e-6,
    "gradient_clip_norm": 1.0,
    "d_model": 32,
    "nhead": 4,
    "num_layers": 2,
    "ff_dim": 64,
    "dropout": 0.1,
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
    direction = np.load(DATA_DIR / f"y_{name}_direction.npy").astype(np.float32).reshape(-1)
    print(f"{name}: X={X.shape}, y={y.shape}, up={direction.mean():.2%}")
    return X, y, direction


def load_zero_threshold() -> float:
    path = DATA_DIR / "scaler_y.pkl"
    if not path.exists():
        return 0.0
    scaler = joblib.load(path)
    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0])
    return -mean / scale


def make_loader(X, y, direction, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(direction))
    return DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
    )


class BareTransformer(nn.Module):
    def __init__(self, input_dim: int, window_size: int, d_model: int, nhead: int,
                 num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)   # last timestep only


@torch.no_grad()
def evaluate(model, loader, criterion, zero_th: float) -> Dict:
    model.eval()

    loss_sum, n = 0.0, 0
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

    # 연속적인 수익률 예측값으로 방향을 예측
    pred_dir = (y_pred > zero_th).astype(np.int64)

    try:
        auc = float(roc_auc_score(actual_dir, y_pred))
    except ValueError:
        auc = None

    return {
        "loss": loss_sum / max(n, 1),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "sign_acc": float(accuracy_score(actual_dir, pred_dir)),
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "actual_dir": actual_dir,
        "pred_dir": pred_dir,
    }



def main():
    set_seed(CONFIG["seed"])
    print(f"Device: {DEVICE}")

    X_tr, y_tr, d_tr = load_split("train")
    X_va, y_va, d_va = load_split("valid")
    X_te, y_te, d_te = load_split("test")

    input_dim, window = X_tr.shape[-1], X_tr.shape[1]
    print(f"Features={input_dim}, window={window}")

    train_loader = make_loader(X_tr, y_tr, d_tr, True)
    valid_loader = make_loader(X_va, y_va, d_va, False)
    test_loader  = make_loader(X_te, y_te, d_te, False)

    model = BareTransformer(
        input_dim, window,
        CONFIG["d_model"], CONFIG["nhead"], CONFIG["num_layers"],
        CONFIG["ff_dim"], CONFIG["dropout"]
    ).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=CONFIG["scheduler_patience"], min_lr=CONFIG["min_learning_rate"]
    )
    amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
             else torch.cuda.amp.GradScaler(enabled=amp)

    zero_th = load_zero_threshold()
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history, t0 = [], time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for X, y, _ in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(X), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
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
            "valid_auc": valid["auc"],
            "lr": lr,
        })

        
        print(
            f"Ep {epoch:03d} | train={train_loss:.4f} | "
            f"val={valid['loss']:.4f} MAE={valid['mae']:.4f} "
            f"Sign={valid['sign_acc']*100:.1f}% "
            f"AUC={valid['auc'] if valid['auc'] is not None else float('nan'):.4f} "
            f"| lr={lr:.1e}"
        )


        if valid["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = valid["loss"], epoch, 0
            torch.save(model.state_dict(), DATA_DIR / "best_model.pth")
            print("  → best saved")
        else:
            stale += 1
            if stale >= CONFIG["early_stopping_patience"]:
                print("Early stopping.")
                break

    # =========================
    # Training history & plots
    # =========================

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / f"training_history120{seed}.csv", index=False)

    epochs_range = history_df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Train / Validation Loss
    ax = axes[0, 0]
    ax.plot(epochs_range, history_df["train_loss"], label="Train Loss")
    ax.plot(epochs_range, history_df["valid_loss"], label="Validation Loss")
    ax.axvline(best_epoch, color="gray", linestyle="--",
            alpha=0.7, label=f"Best Epoch: {best_epoch}")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Validation Direction Accuracy
    ax = axes[0, 1]
    ax.plot(
        epochs_range,
        history_df["valid_sign_acc"] * 100,
        label="Validation Sign Accuracy"
    )
    ax.axhline(
        history_df["valid_sign_acc"].mean() * 100,
        color="gray", linestyle=":", alpha=0.7,
        label="Mean Validation Accuracy"
    )
    ax.set_title("Validation Direction Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs_range, history_df["lr"])
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # 4. Validation MAE
    ax = axes[1, 1]
    ax.plot(epochs_range, history_df["valid_mae"],
            label="Validation MAE")
    ax.axvline(best_epoch, color="gray", linestyle="--",
            alpha=0.7, label=f"Best Epoch: {best_epoch}")
    ax.set_title("Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("KOSPI Transformer Training Dashboard", fontsize=15)
    fig.tight_layout()

    plot_path = OUTPUT_DIR / f"training_curves120{seed}.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Training plots saved: {plot_path}")

    plt.show()
    plt.close(fig)

    print(
        f"\nDone in {(time.time()-t0)/60:.1f} min "
        f"| best ep={best_epoch} loss={best_loss:.5f}"
    )

    model.load_state_dict(torch.load(DATA_DIR / "best_model.pth", map_location=DEVICE))
    test = evaluate(model, test_loader, criterion, zero_th)

    print(f"\n===== TEST =====")
    print(f"Samples     : {len(test['y_true'])}")
    print(f"Huber loss  : {test['loss']:.5f}")
    print(f"MAE         : {test['mae']:.5f}")
    print(f"Sign acc    : {test['sign_acc']*100:.2f}%")
    print(confusion_matrix(test["actual_dir"], test["pred_dir"], labels=[0, 1]))
    print(f"Always-up   : {(test['actual_dir']==1).mean()*100:.2f}%")
    print(f"ROC-AUC:             {test['auc']}")

    pd.DataFrame({
        "actual_return_scaled": test["y_true"],
        "predicted_return_scaled": test["y_pred"],
        "actual_direction": test["actual_dir"],
        "predicted_direction": test["pred_dir"],
    }).to_csv(OUTPUT_DIR / f"test_predictions120{seed}.csv", index=False)

    with open(OUTPUT_DIR / f"metrics120{seed}.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "test_loss": test["loss"],
            "test_mae": test["mae"],
            "test_sign_acc": test["sign_acc"],
            "config": CONFIG,
        }, f, indent=2)

    print(f"\nSaved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()