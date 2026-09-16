import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 1. 설정
# =========================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# =========================
# 2. 데이터 불러오기
# =========================

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_valid = np.load("data/processed/X_valid.npy")
y_valid = np.load("data/processed/y_valid.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_valid:", X_valid.shape)
print("y_valid:", y_valid.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# =========================
# 3. NumPy → PyTorch Tensor
# =========================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# 4. Transformer 모델
# =========================

class TransformerModel(nn.Module):

    def __init__(
        self,
        input_dim=28,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        # 28개 feature → 64차원
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 마지막에 수익률 1개 예측
        self.output_layer = nn.Linear(d_model, 1)


    def forward(self, x):

        # x:
        # (batch, 30, 28)

        x = self.input_projection(x)

        # (batch, 30, 64)
        x = self.transformer(x)

        # 마지막 날짜의 representation 사용
        x = x[:, -1, :]

        # (batch, 64) → (batch, 1)
        x = self.output_layer(x)

        return x.squeeze(-1)


# =========================
# 5. 모델 생성
# =========================

model = TransformerModel().to(DEVICE)

print(model)


# =========================
# 6. Loss / Optimizer
# =========================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# 7. Validation 함수
# =========================

def evaluate(loader):

    model.eval()

    total_loss = 0
    total = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(X)

            loss = criterion(prediction, y)

            total_loss += loss.item() * len(y)
            total += len(y)

    return total_loss / total


# =========================
# 8. 학습
# =========================

best_valid_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # 이전 gradient 제거
        optimizer.zero_grad()

        # 예측
        prediction = model(X)

        # Loss
        loss = criterion(prediction, y)

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        total_loss += loss.item() * len(y)
        total += len(y)

    train_loss = total_loss / total

    valid_loss = evaluate(valid_loader)

    print(
        f"Epoch [{epoch + 1:3d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Valid Loss: {valid_loss:.6f}"
    )

    # 가장 좋은 모델 저장
    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(
            model.state_dict(),
            "data/processed/best_model.pth"
        )

        print("  → Best model saved")


# =========================
# 9. Test
# =========================

model.load_state_dict(
    torch.load(
        "data/processed/best_model.pth",
        map_location=DEVICE
    )
)

test_loss = evaluate(test_loader)

print()
print("=========================")
print(f"Test Loss: {test_loss:.6f}")
print("=========================")


# =========================
# 10. 방향성 정확도
# =========================

model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        prediction = model(X)

        predictions.extend(prediction.cpu().numpy())
        actuals.extend(y.numpy())


predictions = np.array(predictions)
actuals = np.array(actuals)

direction_accuracy = (
    np.sign(predictions) == np.sign(actuals)
).mean()

print(f"Direction Accuracy: {direction_accuracy * 100:.2f}%")