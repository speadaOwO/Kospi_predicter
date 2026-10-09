import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 기본 설정
# =========================================================

INPUT_PATH = "data/raw/data.csv"
OUTPUT_DIR = "data/processed"

WINDOW_SIZE = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

df = pd.read_csv(
    INPUT_PATH,
    index_col=0,
    parse_dates=True
)

df.index.name = "Date"

# 날짜순 정렬
df = df.sort_index()


# =========================================================
# 3. KOSPI 기본 지표
# =========================================================

# 일일 수익률
df["KOSPI_return"] = df["Close"].pct_change()

# 거래량 변화율
df["volume_change"] = df["Volume"].pct_change()

# 이동평균
df["KOSPI_MA5"] = df["Close"].rolling(5).mean()
df["KOSPI_MA20"] = df["Close"].rolling(20).mean()
df["KOSPI_MA60"] = df["Close"].rolling(60).mean()

# 이동평균 이격도
df["KOSPI_disparity5"] = df["Close"] / df["KOSPI_MA5"] * 100
df["KOSPI_disparity20"] = df["Close"] / df["KOSPI_MA20"] * 100


# =========================================================
# 4. 해외 증시 / 거시경제 변동률
# =========================================================

df["SP500_return"] = df["^GSPC"].pct_change()
df["NASDAQ_return"] = df["^IXIC"].pct_change()
df["SOX_return"] = df["^SOX"].pct_change()

df["USDKRW_return"] = df["KRW=X"].pct_change()

df["VIX_change"] = df["^VIX"].pct_change()
df["WTI_return"] = df["CL=F"].pct_change()
df["US10Y_change"] = df["^TNX"].pct_change()


# =========================================================
# 5. USD/KRW 이동평균 이격도
# =========================================================

usd_ma5 = df["KRW=X"].rolling(5).mean()
usd_ma20 = df["KRW=X"].rolling(20).mean()

df["USDKRW_disparity5"] = (
    df["KRW=X"] / usd_ma5 * 100
)

df["USDKRW_disparity20"] = (
    df["KRW=X"] / usd_ma20 * 100
)


# =========================================================
# 6. RSI (14)
# =========================================================

delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

# Wilder 방식에 가까운 EMA 사용
avg_gain = gain.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()

avg_loss = loss.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()

rs = avg_gain / avg_loss

df["RSI_14"] = 100 - (100 / (1 + rs))


# =========================================================
# 7. MACD
# =========================================================

ema12 = df["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = df["Close"].ewm(
    span=26,
    adjust=False
).mean()

df["MACD"] = ema12 - ema26

df["MACD_Signal"] = df["MACD"].ewm(
    span=9,
    adjust=False
).mean()

df["MACD_Histogram"] = (
    df["MACD"] - df["MACD_Signal"]
)


# 가격 수준에 따른 영향을 줄이기 위한 정규화 MACD
df["MACD_pct"] = df["MACD"] / df["Close"]
df["MACD_Signal_pct"] = df["MACD_Signal"] / df["Close"]
df["MACD_Histogram_pct"] = (
    df["MACD_Histogram"] / df["Close"]
)


# =========================================================
# 8. Bollinger Bands
# =========================================================

bb_middle = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()

df["BB_Upper"] = bb_middle + 2 * bb_std
df["BB_Lower"] = bb_middle - 2 * bb_std

# %B
df["BB_%B"] = (
    (df["Close"] - df["BB_Lower"])
    / (df["BB_Upper"] - df["BB_Lower"])
)

# BB Width
df["BB_Width"] = (
    (df["BB_Upper"] - df["BB_Lower"])
    / bb_middle
)


# =========================================================
# 9. ATR (14)
# =========================================================

high_low = df["High"] - df["Low"]

high_close = (
    df["High"] - df["Close"].shift(1)
).abs()

low_close = (
    df["Low"] - df["Close"].shift(1)
).abs()

true_range = pd.concat(
    [high_low, high_close, low_close],
    axis=1
).max(axis=1)

df["ATR_14"] = true_range.rolling(14).mean()

# 가격 대비 ATR
df["ATR_14_pct"] = df["ATR_14"] / df["Close"]


# =========================================================
# 10. Stochastic Fast %K / %D
# =========================================================

lowest_low = df["Low"].rolling(14).min()
highest_high = df["High"].rolling(14).max()

denominator = highest_high - lowest_low

df["Stoch_%K"] = (
    (df["Close"] - lowest_low)
    / denominator
    * 100
)

df["Stoch_%D"] = (
    df["Stoch_%K"].rolling(3).mean()
)


# =========================================================
# 11. 수급 데이터
# =========================================================

flow_features = [
    "institution_net_value",
    "retail_net_value",
    "foreign_net_value",

    "institution_net_volume",
    "retail_net_volume",
    "foreign_net_volume"
]


# =========================================================
# 12. Target
# =========================================================

# t일의 정보를 이용하여 t+1일 KOSPI 수익률 예측
df["target"] = df["KOSPI_return"].shift(-1)


# =========================================================
# 13. 사용할 Feature 선택
# =========================================================

features = [

    # KOSPI
    "KOSPI_return",
    "volume_change",

    # KOSPI 기술적 지표
    "KOSPI_disparity5",
    "KOSPI_disparity20",

    "RSI_14",

    "MACD_pct",
    "MACD_Signal_pct",
    "MACD_Histogram_pct",

    "BB_%B",
    "BB_Width",

    "ATR_14_pct",

    "Stoch_%K",
    "Stoch_%D",

    # 해외시장
    "SP500_return",
    "NASDAQ_return",
    "SOX_return",

    # 환율
    "USDKRW_return",
    "USDKRW_disparity5",
    "USDKRW_disparity20",

    # 변동성 / 거시
    "VIX_change",
    "WTI_return",
    "US10Y_change",

    # 투자자 수급
    "institution_net_value",
    "retail_net_value",
    "foreign_net_value",

    "institution_net_volume",
    "retail_net_volume",
    "foreign_net_volume"
]


# =========================================================
# 14. Inf / NaN 처리
# =========================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=features + ["target"]
)


print("\n===== Dataset =====")
print("전체 데이터:", len(df))
print("Feature 개수:", len(features))

print("\nFeatures:")
for i, feature in enumerate(features):
    print(f"{i:2d}. {feature}")


df.to_csv("data/processed/raw_prep.csv")

# =========================================================
# 15. X / y
# =========================================================

X = df[features].copy()
y = df["target"].copy()


# =========================================================
# 16. 시간 순서대로 Train / Valid / Test 분할
# =========================================================

n = len(X)

train_end = int(n * 0.70)
valid_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
X_valid = X.iloc[train_end:valid_end].copy()
X_test = X.iloc[valid_end:].copy()

y_train = y.iloc[:train_end].copy()
y_valid = y.iloc[train_end:valid_end].copy()
y_test = y.iloc[valid_end:].copy()


print("\n===== Split =====")
print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test :", X_test.shape)


# =========================================================
# 17. StandardScaler
# =========================================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 반드시 Train에만 fit
X_train = scaler_X.fit_transform(X_train)
X_valid = scaler_X.transform(X_valid)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).flatten()

y_valid = scaler_y.transform(
    y_valid.to_numpy().reshape(-1, 1)
).flatten()

y_test = scaler_y.transform(
    y_test.to_numpy().reshape(-1, 1)
).flatten()


# =========================================================
# 18. Sequence 생성
# =========================================================


def make_sequences(X, y, window_size):
    X_seq = []
    y_seq = []

    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i + window_size])

        # 입력 시퀀스 마지막 날의 다음 날 수익률
        y_seq.append(y[i + window_size - 1])

    return np.array(X_seq), np.array(y_seq)


X_train_seq, y_train_seq = make_sequences(
    X_train,
    y_train,
    WINDOW_SIZE
)

X_valid_seq, y_valid_seq = make_sequences(
    X_valid,
    y_valid,
    WINDOW_SIZE
)

X_test_seq, y_test_seq = make_sequences(
    X_test,
    y_test,
    WINDOW_SIZE
)


# =========================================================
# 19. 결과 확인
# =========================================================

print("\n===== Sequence =====")

print("X_train:", X_train_seq.shape)
print("y_train:", y_train_seq.shape)

print("X_valid:", X_valid_seq.shape)
print("y_valid:", y_valid_seq.shape)

print("X_test :", X_test_seq.shape)
print("y_test :", y_test_seq.shape)


# =========================================================
# 20. 저장
# =========================================================

np.save(
    f"{OUTPUT_DIR}/X_train.npy",
    X_train_seq
)

np.save(
    f"{OUTPUT_DIR}/y_train.npy",
    y_train_seq
)

np.save(
    f"{OUTPUT_DIR}/X_valid.npy",
    X_valid_seq
)

np.save(
    f"{OUTPUT_DIR}/y_valid.npy",
    y_valid_seq
)

np.save(
    f"{OUTPUT_DIR}/X_test.npy",
    X_test_seq
)

np.save(
    f"{OUTPUT_DIR}/y_test.npy",
    y_test_seq
)

# scaler 저장
import joblib

joblib.dump(
    scaler_X,
    f"{OUTPUT_DIR}/scaler_X.pkl"
)

joblib.dump(
    scaler_y,
    f"{OUTPUT_DIR}/scaler_y.pkl"
)


# 전처리된 전체 DataFrame도 저장
df.to_csv(
    f"{OUTPUT_DIR}/preprocessed.csv"
)


print("\n===== 저장 완료 =====")
print(f"저장 위치: {OUTPUT_DIR}/")

print("Feature count:", len(features))
print("Features:")
for i, feature in enumerate(features, 1):
    print(i, feature)