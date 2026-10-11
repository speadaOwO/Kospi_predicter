
from pathlib import Path
import pandas as pd

# CSV 파일이 들어 있는 폴더
DATA_DIR = Path("./data/raw/VKOSPI")

# 폴더 안의 모든 CSV 읽기
files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError("vkospi_data 폴더에 CSV 파일이 없습니다.")

dfs = []



for file in files:
    df = pd.read_csv(file, encoding="cp949")
    print(file.name, df.shape, df.columns.tolist())
    dfs.append(df)

# 전부 합치기
merged = pd.concat(dfs, ignore_index=True)

# 중복 행 제거
merged = merged.drop_duplicates()

# 날짜를 datetime 형식으로 변환
merged["일자"] = pd.to_datetime(
    merged["일자"],
    errors="coerce"
)

# 날짜 오름차순 정렬: 과거 → 현재
merged = (
    merged.dropna(subset=["일자"])
          .drop_duplicates(subset=["일자"], keep="last")
          .sort_values("일자", ascending=True)
          .reset_index(drop=True)
)

# 저장
merged.to_csv(
    "VKOSPI_merged.csv",
    index=False,
    encoding="utf-8-sig"
)

print(merged[["일자"]].head())  # 가장 오래된 날짜
print(merged[["일자"]].tail())  # 가장 최근 날짜
# 결과 저장
merged.to_csv(
    "./data/raw/VKOSPI_merged.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n병합 완료!")
print("총 행 수:", len(merged))
print(merged.head())
print(merged.tail())
