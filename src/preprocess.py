
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 기본 설정
# =========================================================

INPUT_PATH = "data/raw/data.csv"
OUTPUT_DIR = "data/processed"

WINDOW_SIZE = 250

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

df = pd.read_csv(
    INPUT_PATH,
    index_col=0,
    parse_dates=True
)

df.index = pd.to_datetime(df.index)
df.index.name = "Date"

df = df.sort_index()

# 중복 날짜 검사
if df.index.has_duplicates:
    print("중복 날짜 발견. 마지막 행만 유지합니다.")
    df = df[~df.index.duplicated(keep="last")]

print("원본 데이터:", df.shape)
print("원본 기간:", df.index.min(), "~", df.index.max())


# =========================================================
# 3. 필수 컬럼 검사
# =========================================================

required_columns = [
    "Close", "High", "Low", "Volume",
    "^GSPC", "^IXIC", "^SOX", "KRW=X",
    "^VIX", "CL=F", "^TNX",
    "institution_net_value",
    "retail_net_value",
    "foreign_net_value",
    "institution_net_volume",
    "retail_net_volume",
    "foreign_net_volume",
    "VKOSPI"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}"
    )


# =========================================================
# 4. KOSPI 기본 지표
# =========================================================

df["KOSPI_return"] = df["Close"].pct_change(fill_method=None)

df["volume_change"] = df["Volume"].pct_change(fill_method=None)

df["KOSPI_MA5"] = df["Close"].rolling(5).mean()
df["KOSPI_MA20"] = df["Close"].rolling(20).mean()
df["KOSPI_MA60"] = df["Close"].rolling(60).mean()

df["KOSPI_disparity5"] = (
    df["Close"] / df["KOSPI_MA5"] * 100
)

df["KOSPI_disparity20"] = (
    df["Close"] / df["KOSPI_MA20"] * 100
)


# =========================================================
# 5. 해외 증시 / 거시경제 변동률
# =========================================================

df["SP500_return"] = df["^GSPC"].pct_change(fill_method=None)
df["NASDAQ_return"] = df["^IXIC"].pct_change(fill_method=None)
df["SOX_return"] = df["^SOX"].pct_change(fill_method=None)

df["USDKRW_return"] = df["KRW=X"].pct_change(fill_method=None)

df["VIX_change"] = df["^VIX"].pct_change(fill_method=None)
df["WTI_return"] = df["CL=F"].pct_change(fill_method=None)
df["US10Y_change"] = df["^TNX"].pct_change(fill_method=None)


# =========================================================
# 6. VKOSPI 특성 추가
# =========================================================

# VKOSPI 수준: 시장이 예상하는 변동성 수준
df["VKOSPI"] = pd.to_numeric(
    df["VKOSPI"],
    errors="coerce"
)

# VKOSPI 일일 변화율: 변동성의 급격한 상승/하락
df["VKOSPI_change"] = df["VKOSPI"].pct_change(
    fill_method=None
)


# =========================================================
# 7. USD/KRW 이동평균 이격도
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
# 8. RSI (14)
# =========================================================

delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

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
# 9. MACD
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

df["MACD_pct"] = df["MACD"] / df["Close"]
df["MACD_Signal_pct"] = df["MACD_Signal"] / df["Close"]
df["MACD_Histogram_pct"] = (
    df["MACD_Histogram"] / df["Close"]
)


# =========================================================
# 10. Bollinger Bands
# =========================================================

bb_middle = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()

df["BB_Upper"] = bb_middle + 2 * bb_std
df["BB_Lower"] = bb_middle - 2 * bb_std

df["BB_%B"] = (
    (df["Close"] - df["BB_Lower"])
    / (df["BB_Upper"] - df["BB_Lower"])
)

df["BB_Width"] = (
    (df["BB_Upper"] - df["BB_Lower"])
    / bb_middle
)


# =========================================================
# 11. ATR (14)
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
df["ATR_14_pct"] = df["ATR_14"] / df["Close"]


# =========================================================
# 12. Stochastic Fast %K / %D
# =========================================================

lowest_low = df["Low"].rolling(14).min()
highest_high = df["High"].rolling(14).max()

denominator = highest_high - lowest_low

df["Stoch_%K"] = (
    (df["Close"] - lowest_low)
    / denominator
    * 100
)

df["Stoch_%D"] = df["Stoch_%K"].rolling(3).mean()

# =========================
# 거시경제 파생변수
# =========================

# 1. 금리 변화량 (단위: %p)
df["KR_rate_change"] = df["INTDSRKRM193N"].diff()
df["US_rate_change"] = df["FEDFUNDS"].diff()
df["JP_rate_change"] = df["INTDSRJPM193N"].diff()

# 2. CPI 전년 동월 대비 상승률 (%)
# 일별로 확장된 데이터에서 계산하지 않고,
# 월별 관측값 간 비교가 필요함.
# 아래 방식은 월별 값이 일별 데이터에 forward-fill된 경우에만 사용.
cpi_columns = {
    "KORCPIALLMINMEI": "KR_CPI_inflation",
    "CPIAUCSL": "US_CPI_inflation",
    "JPNCPIALLMINMEI": "JP_CPI_inflation",
}

# for source_col, new_col in cpi_columns.items():
#     monthly = df[source_col].dropna()
#     monthly = monthly[~monthly.index.duplicated(keep="last")]

#     # 월별 관측값만 추출
#     monthly = monthly.groupby(monthly.index.to_period("M")).last()
#     monthly.index = monthly.index.to_timestamp()

#     # 전년 동월 대비 상승률
#     inflation = monthly.pct_change(12, fill_method=None) * 100

#     # 월별 결과를 일별 날짜에 병합
#     inflation.index.name = "Date"
#     df[new_col] = inflation.reindex(
#         df.index.to_period("M").to_timestamp()
#     ).to_numpy()

# =========================
# features 목록에 추가할 변수
# =========================

macro_features = [
    # 금리 수준
    "INTDSRKRM193N",
    "FEDFUNDS",
    "INTDSRJPM193N",

    # 금리 변화량
    "KR_rate_change",
    "US_rate_change",
    "JP_rate_change",

    # CPI 지수
    # "KORCPIALLMINMEI",
    # "CPIAUCSL",
    # "JPNCPIALLMINMEI",

    # CPI 전년 동월 대비 상승률
    # "KR_CPI_inflation",
    # "US_CPI_inflation",
    # "JP_CPI_inflation",
]



# =========================================================
# 13. Target
# =========================================================

# t일 정보로 t+1일 KOSPI 수익률 예측
df["target"] = df["KOSPI_return"].shift(-1)


# =========================================================
# 14. 사용할 Feature 선택
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
    "VKOSPI",
    "VKOSPI_change",
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

features += macro_features


#=========================================================
# HEHE - 중간 꺼내기
#=========================================================

df.to_csv("data/processed/processed.csv")




# =========================================================
# 15. Inf / NaN 처리
# =========================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# 사용할 특성과 target에 결측치가 있는 행 제거
df = df.dropna(
    subset=features + ["target"]
).copy()

print("\n===== Dataset =====")
print("사용 가능한 데이터:", len(df))
print("Feature 개수:", len(features))
print("시작 날짜:", df.index.min())
print("종료 날짜:", df.index.max())

print("\nFeatures:")
for i, feature in enumerate(features, 1):
    print(f"{i:2d}. {feature}")

# 남아 있는 결측치 검사
assert not df[features + ["target"]].isna().any().any(), (
    "Feature 또는 target에 결측치가 남아 있습니다."
)

# VKOSPI가 없는 과거 데이터는 위 dropna 단계에서 제거됨
assert df["VKOSPI"].notna().all()

df.to_csv(
    "data/processed/raw_prep.csv",
    encoding="utf-8-sig"
)


# =========================================================
# 16. X / y 및 분류 라벨
# =========================================================

X = df[features].copy()
y_raw = df["target"].copy()

# 실제 수익률 기준 상승/하락 라벨
y_direction = (y_raw > 0).astype(np.float32)

# 시간순 분할
n = len(X)

train_end = int(n * 0.80)
# valid_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
X_valid = X.iloc[train_end:].copy()
# X_test = X.iloc[valid_end:].copy()

y_train_raw = y_raw.iloc[:train_end].copy()
y_valid_raw = y_raw.iloc[train_end:].copy()
# y_test_raw = y_raw.iloc[valid_end:].copy()

y_train_dir = y_direction.iloc[:train_end].copy()
y_valid_dir = y_direction.iloc[train_end:].copy()
# y_test_dir = y_direction.iloc[valid_end:].copy()

for name, labels in [
    ("Train", y_train_dir),
    ("Valid", y_valid_dir),
    # ("Test", y_test_dir),
]:
    print(
        f"{name}: 상승 {labels.mean():.2%}, "
        f"하락 {1 - labels.mean():.2%}"
    )


# =========================================================
# 17. StandardScaler
# =========================================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 스케일러는 train 데이터에만 fit
X_train = scaler_X.fit_transform(X_train)
X_valid = scaler_X.transform(X_valid)
# X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(
    y_train_raw.to_numpy().reshape(-1, 1)
).flatten()

y_valid = scaler_y.transform(
    y_valid_raw.to_numpy().reshape(-1, 1)
).flatten()

# y_test = scaler_y.transform(
#     y_test_raw.to_numpy().reshape(-1, 1)
# ).flatten()


# =========================================================
# 18. Sequence 생성
# =========================================================

def make_sequences(X, y, window_size):
    X_seq = []
    y_seq = []

    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size - 1])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32)
    )


def make_direction_sequences(labels, window_size):
    labels = np.asarray(labels)

    return np.asarray([
        labels[i + window_size - 1]
        for i in range(len(labels) - window_size + 1)
    ], dtype=np.float32)


y_train_dir_seq = make_direction_sequences(
    y_train_dir, WINDOW_SIZE
)

y_valid_dir_seq = make_direction_sequences(
    y_valid_dir, WINDOW_SIZE
)

# y_test_dir_seq = make_direction_sequences(
#     y_test_dir, WINDOW_SIZE
# )

X_train_seq, y_train_seq = make_sequences(
    X_train, y_train, WINDOW_SIZE
)

X_valid_seq, y_valid_seq = make_sequences(
    X_valid, y_valid, WINDOW_SIZE
)

# X_test_seq, y_test_seq = make_sequences(
#     X_test, y_test, WINDOW_SIZE
# )


# =========================================================
# 19. Sequence 결과 검증
# =========================================================

print("\n===== Sequence =====")

print("X_train:", X_train_seq.shape)
print("y_train:", y_train_seq.shape)
print("Direction train:", y_train_dir_seq.shape)

print("X_valid:", X_valid_seq.shape)
print("y_valid:", y_valid_seq.shape)
print("Direction valid:", y_valid_dir_seq.shape)

# print("X_test :", X_test_seq.shape)
# print("y_test :", y_test_seq.shape)
# print("Direction test:", y_test_dir_seq.shape)

assert X_train_seq.shape[1] == WINDOW_SIZE
assert X_train_seq.shape[2] == len(features)

assert len(X_train_seq) == len(y_train_seq)
assert len(X_valid_seq) == len(y_valid_seq)
# assert len(X_test_seq) == len(y_test_seq)

assert len(y_train_seq) == len(y_train_dir_seq)
assert len(y_valid_seq) == len(y_valid_dir_seq)
# assert len(y_test_seq) == len(y_test_dir_seq)


# =========================================================
# 20. 저장
# =========================================================

arrays = {
    "X_train": X_train_seq,
    "y_train": y_train_seq,
    "X_valid": X_valid_seq,
    "y_valid": y_valid_seq,
    # "X_test": X_test_seq,
    # "y_test": y_test_seq,
    "y_train_direction": y_train_dir_seq,
    "y_valid_direction": y_valid_dir_seq,
    # "y_test_direction": y_test_dir_seq,
}

for name, array in arrays.items():
    np.save(
        os.path.join(OUTPUT_DIR, f"{name}.npy"),
        array
    )

joblib.dump(
    scaler_X,
    os.path.join(OUTPUT_DIR, "scaler_X.pkl")
)

joblib.dump(
    scaler_y,
    os.path.join(OUTPUT_DIR, "scaler_y.pkl")
)

# 학습 코드에서 특성 구성을 확인할 수 있도록 저장
with open(
    os.path.join(OUTPUT_DIR, "feature_names.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(features, f, ensure_ascii=False, indent=2)

# 전체 전처리 데이터 저장
df.to_csv(
    os.path.join(OUTPUT_DIR, "preprocessed.csv"),
    encoding="utf-8-sig"
)

print("\n===== 저장 완료 =====")
print("저장 위치:", OUTPUT_DIR)
print("Feature count:", len(features))
print("Input shape:", X_train_seq.shape)
print("VKOSPI 사용:", True)
