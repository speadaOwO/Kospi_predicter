"""KOSPI next-day return - bare Transformer Multi-Seed Runner"""
from __future__ import annotations
import json, random, time
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "training_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 여기서 돌릴 Seed 설정 =====
SEEDS_TO_RUN = [200, 201, 202, 203, 204]

BASE_CONFIG = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "early_stopping_patience": 7,
    "scheduler_patience": 4,
    "min_learning_rate": 1e-6,
    "gradient_clip_norm": 1.0,
    "d_model": 12,
    "nhead": 4,
    "num_layers": 2,
    "ff_dim": 24,
    "dropout": 0.1,
    "huber_delta": 1.0,
    "num_workers": 0,
    "use_amp": True,
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_split(name: str):
    X = np.load(DATA_DIR / f"X_{name}.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"y_{name}.npy").astype(np.float32).reshape(-1)
    direction = np.load(DATA_DIR / f"y_{name}_direction.npy").astype(np.float32).reshape(-1)
    return X, y, direction

def load_zero_threshold():
    path = DATA_DIR / "scaler_y.pkl"
    if not path.exists(): return 0.0
    scaler = joblib.load(path)
    return -float(scaler.mean_[0]) / float(scaler.scale_[0])

def make_loader(X, y, direction, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(direction))
    return DataLoader(ds, batch_size=BASE_CONFIG["batch_size"], shuffle=shuffle, num_workers=0, pin_memory=DEVICE.type == "cuda")

class BareTransformer(nn.Module):
    def __init__(self, input_dim, window_size, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)

@torch.no_grad()
def evaluate(model, loader, criterion, zero_th):
    model.eval()
    loss_sum, n, ys, preds, dirs = 0.0, 0, [], [], []
    for X, y, d in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss_sum += criterion(pred, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy()); preds.append(pred.cpu().numpy()); dirs.append(d.numpy().astype(np.int64))
    y_true = np.concatenate(ys); y_pred = np.concatenate(preds); actual_dir = np.concatenate(dirs)
    pred_dir = (y_pred > zero_th).astype(np.int64)
    try: auc = float(roc_auc_score(actual_dir, y_pred))
    except: auc = None
    return {"loss": loss_sum/max(n,1), "mae": float(mean_absolute_error(y_true, y_pred)), "sign_acc": float(accuracy_score(actual_dir, pred_dir)), "auc": auc, "y_true": y_true, "y_pred": y_pred, "actual_dir": actual_dir, "pred_dir": pred_dir}

def find_best_threshold(y_true_dir, y_pred_score):
    best_th, best_acc = 0, 0
    thresholds = np.linspace(y_pred_score.min(), y_pred_score.max(), 100)
    for th in thresholds:
        acc = accuracy_score(y_true_dir, (y_pred_score > th).astype(int))
        if acc > best_acc: best_acc, best_th = acc, th
    return best_th, best_acc

def run_one_seed(seed: int) -> Dict:
    print(f"\n{'='*20} SEED {seed} START {'='*20}")
    set_seed(seed)
    CONFIG = {**BASE_CONFIG, "seed": seed}

    X_tr, y_tr, d_tr = load_split("train")
    X_va, y_va, d_va = load_split("valid")
    X_te, y_te, d_te = load_split("test")
    print(f"train up={d_tr.mean():.2%} valid up={d_va.mean():.2%} test up={d_te.mean():.2%}")

    input_dim, window = X_tr.shape[-1], X_tr.shape[1]
    train_loader = make_loader(X_tr, y_tr, d_tr, True)
    valid_loader = make_loader(X_va, y_va, d_va, False)
    test_loader = make_loader(X_te, y_te, d_te, False)

    model = BareTransformer(input_dim, window, CONFIG["d_model"], CONFIG["nhead"], CONFIG["num_layers"], CONFIG["ff_dim"], CONFIG["dropout"]).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=CONFIG["scheduler_patience"], min_lr=CONFIG["min_learning_rate"])
    amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp)
    zero_th = load_zero_threshold()

    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, CONFIG["epochs"]+1):
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
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()*len(y); n+=len(y)
        train_loss = loss_sum/max(n,1)
        valid = evaluate(model, valid_loader, criterion, zero_th)
        scheduler.step(valid["loss"])
        if valid["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = valid["loss"], epoch, 0
            torch.save(model.state_dict(), DATA_DIR / f"best_model_{seed}.pth")
        else:
            stale+=1
            if stale >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping at {epoch}")
                break
        print(f"Ep {epoch:03d} | train={train_loss:.4f} val={valid['loss']:.4f} Sign={valid['sign_acc']*100:.1f}% AUC={valid['auc']:.4f}")

    model.load_state_dict(torch.load(DATA_DIR / f"best_model_{seed}.pth", map_location=DEVICE))
    valid_best = evaluate(model, valid_loader, criterion, zero_th)
    best_th, _ = find_best_threshold(valid_best["actual_dir"], valid_best["y_pred"])
    test = evaluate(model, test_loader, criterion, zero_th)
    tuned_acc = accuracy_score(test["actual_dir"], (test["y_pred"] > best_th).astype(int))

    print(f"\n[SEED {seed} RESULT] Best Ep={best_epoch} | Test AUC={test['auc']:.4f} | Sign={test['sign_acc']*100:.2f}% | Tuned={tuned_acc*100:.2f}% | BestTh={best_th:.4f}")

    # 개별 저장
        # 개별 저장 - float로 변환
    with open(OUTPUT_DIR / f"metrics_{seed}.json", "w") as f:
        json.dump({
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "test_auc": float(test["auc"]) if test["auc"] is not None else None,
            "test_sign_acc": float(test["sign_acc"]),
            "tuned_acc": float(tuned_acc),
            "best_th": float(best_th),
            "config": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in CONFIG.items()}
        }, f, indent=2)

    return {
        "seed": int(seed),
        "auc": float(test["auc"]) if test["auc"] else 0.0,
        "sign_acc": float(test["sign_acc"]),
        "tuned_acc": float(tuned_acc),
        "best_th": float(best_th),
        "best_epoch": int(best_epoch),
        "always_up": float((test["actual_dir"]==1).mean())
    }
def main():
    all_results = []
    for seed in SEEDS_TO_RUN:
        res = run_one_seed(seed)
        all_results.append(res)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "multi_seed_summary.csv", index=False)

    print("\n\n" + "="*40)
    print("FINAL MULTI-SEED SUMMARY")
    print("="*40)
    print(df.to_string(index=False))
    print(f"\nAUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
    print(f"Sign Acc: {df['sign_acc'].mean()*100:.2f}% ± {df['sign_acc'].std()*100:.2f}%")
    print(f"Tuned Acc: {df['tuned_acc'].mean()*100:.2f}% ± {df['tuned_acc'].std()*100:.2f}%")
    print(f"Always-up: {df['always_up'].mean()*100:.2f}%")
    print(f"\nSaved → {OUTPUT_DIR / 'multi_seed_summary.csv'}")

if __name__ == "__main__":
    main()