
from pathlib import Path
import copy
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score


# ============================================================
# 1. 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "training_run_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
seed = 100
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
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,
    "huber_delta": 1.0,
    "num_workers": 0,
    "use_amp": True,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 데이터
# ============================================================

def load_data():
    data = {}

    for split in ["train", "valid", "test"]:
        x_path = DATA_DIR / f"X_{split}.npy"
        y_path = DATA_DIR / f"y_{split}.npy"
        d_path = DATA_DIR / f"y_{split}_direction.npy"

        for path in [x_path, y_path, d_path]:
            if not path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")

        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32).reshape(-1, 1)
        direction = np.load(d_path).reshape(-1)

        if len(X) != len(y) or len(y) != len(direction):
            raise ValueError(f"{split} 데이터 길이가 일치하지 않음")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{split} 데이터에 NaN 또는 무한대가 있음")

        data[split] = (X, y, direction)

    input_dim = data["train"][0].shape[-1]

    for split, (X, _, _) in data.items():
        if X.ndim != 3:
            raise ValueError(
                f"{split} X의 형태가 (샘플, 시간, 특성)이 아님: {X.shape}"
            )
        if X.shape[-1] != input_dim:
            raise ValueError("데이터 분할 간 특성 개수가 다름")

    return data, input_dim


def load_zero_threshold() -> float:
    path = DATA_DIR / "scaler_y.pkl"

    if not path.exists():
        return 0.0

    scaler = joblib.load(path)
    return -float(scaler.mean_[0]) / float(scaler.scale_[0])


# ============================================================
# 3. LSTM 모델
# ============================================================

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        output_dim = hidden_size * directions

        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, time_steps, features)
        _, (h_n, _) = self.lstm(x)

        # 마지막 LSTM 층의 은닉 상태만 사용
        if self.lstm.bidirectional:
            # 정방향 마지막 상태와 역방향 마지막 상태 결합
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]

        return self.head(h)


# ============================================================
# 4. 평가
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, zero_threshold):
    model.eval()

    total_loss = 0.0
    total_count = 0
    predictions = []
    targets = []

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(X)
        loss = criterion(pred, y)

        total_loss += loss.item() * len(X)
        total_count += len(X)

        predictions.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    y_pred = np.concatenate(predictions).reshape(-1)
    y_true = np.concatenate(targets).reshape(-1)

    pred_dir = y_pred > zero_threshold
    true_dir = y_true > zero_threshold

    sign_acc = float(np.mean(pred_dir == true_dir))
    balanced_acc = float(
        balanced_accuracy_score(true_dir, pred_dir)
    )

    try:
        auc = float(roc_auc_score(true_dir, y_pred))
    except ValueError:
        auc = None

    return {
        "loss": total_loss / max(total_count, 1),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "sign_acc": sign_acc,
        "balanced_acc": balanced_acc,
        "auc": auc,
        "y_pred": y_pred,
        "y_true": y_true,
        "pred_dir": pred_dir,
        "true_dir": true_dir,
    }


# ============================================================
# 5. 그래프 저장
# ============================================================

def save_plots(history):
    df = pd.DataFrame(history)
    df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0, 0].plot(df["epoch"], df["valid_loss"], label="Validation")
    axes[0, 0].set_title("Huber Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        df["epoch"],
        df["valid_sign_acc"] * 100,
        label="Validation",
    )
    axes[0, 1].axhline(50, linestyle="--", label="50%")
    axes[0, 1].set_title("Direction Accuracy (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df["epoch"], df["lr"])
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["epoch"], df["valid_mae"])
    axes[1, 1].set_title("Validation MAE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"training_curves120{seed}.png", dpi=160)
    plt.close(fig)


# ============================================================
# 6. 학습
# ============================================================

def main():
    set_seed(CONFIG["seed"])

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")

    data, input_dim = load_data()
    zero_threshold = load_zero_threshold()

    loaders = {}

    for split, (X, y, _) in data.items():
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=(split == "train"),
            num_workers=CONFIG["num_workers"],
            pin_memory=(DEVICE.type == "cuda"),
        )

        print(f"{split}: X={X.shape}, y={y.shape}")

    model = LSTMModel(
        input_dim=input_dim,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        bidirectional=CONFIG["bidirectional"],
    ).to(DEVICE)

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

    amp_enabled = CONFIG["use_amp"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X, y in loaders["train"]:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=amp_enabled,
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

            train_loss_sum += loss.item() * len(X)
            train_count += len(X)

        train_loss = train_loss_sum / max(train_count, 1)

        valid = evaluate(
            model,
            loaders["valid"],
            criterion,
            zero_threshold,
        )

        scheduler.step(valid["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid["loss"],
            "valid_mae": valid["mae"],
            "valid_sign_acc": valid["sign_acc"],
            "valid_balanced_acc": valid["balanced_acc"],
            "valid_auc": valid["auc"],
            "lr": current_lr,
        })

        print(
            f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
            f"train={train_loss:.5f} | "
            f"valid={valid['loss']:.5f} | "
            f"MAE={valid['mae']:.4f} | "
            f"Direction={valid['sign_acc']:.2%} | "
            f"Balanced={valid['balanced_acc']:.2%} | "
            f"LR={current_lr:.2e}"
            
        )

        if valid["loss"] < best_valid_loss:
            best_valid_loss = valid["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(
                best_state,
                OUTPUT_DIR / f"best_lstm_model120{seed}.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("학습 중 최적 모델이 저장되지 않았음")

    model.load_state_dict(best_state)
    save_plots(history)

    # 테스트 데이터는 학습 도중 모델 선택에 사용하지 않음
    test = evaluate(
        model,
        loaders["test"],
        criterion,
        zero_threshold,
    )

    y_true = test["true_dir"]
    y_pred = test["pred_dir"]

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    always_up = float(np.mean(y_true))
    always_down = float(np.mean(~y_true))

    metrics = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "test_loss": test["loss"],
        "test_mae": test["mae"],
        "test_direction_accuracy": test["sign_acc"],
        "test_balanced_accuracy": test["balanced_acc"],
        "test_auc": test["auc"],
        "always_up_accuracy": always_up,
        "always_down_accuracy": always_down,
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(test["y_true"]),
        "zero_threshold": zero_threshold,
        "config": CONFIG,
    }

    pd.DataFrame({
        "y_true_standardized": test["y_true"],
        "y_pred_standardized": test["y_pred"],
        "true_up": y_true.astype(int),
        "pred_up": y_pred.astype(int),
    }).to_csv(OUTPUT_DIR / f"test_predictions120{seed}.csv", index=False)

    with open(OUTPUT_DIR / f"metrics120{seed}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n========== LSTM TEST RESULTS ==========")
    print(f"Best epoch:          {best_epoch}")
    print(f"Test samples:        {len(test['y_true'])}")
    print(f"Test Huber loss:     {test['loss']:.5f}")
    print(f"Test MAE:            {test['mae']:.5f}")
    print(f"Direction accuracy:  {test['sign_acc']:.2%}")
    print(f"Balanced accuracy:   {test['balanced_acc']:.2%}")
    print(f"ROC-AUC:             {test['auc']}")
    print(f"Always-up baseline:  {always_up:.2%}")
    print(f"Always-down baseline:{always_down:.2%}")
    print(f"seed = {seed}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    


if __name__ == "__main__":
    main()
