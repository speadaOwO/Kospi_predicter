import pandas as pd
import yfinance as yf
import pykrx as pk
from pykrx import stock
import FinanceDataReader as fdr

start = "2003-01-01"
end= "2026-10-17"

tickers = {
    "KOSPI": "^KS11",
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "SOX": "^SOX",
    "USD/KRW": "KRW=X",
    "VIX": "^VIX",
    "WTI": "CL=F",
    "US10Y": "^TNX",
    "DOW":"^DJI",
    "KOSDAQ":"^KQ11",
    "Nikkei225":"^N225"
}

kospi = yf.download("^KS11" , start=start,end=end, auto_adjust=False, multi_level_index=False)

macro  = yf.download(list(tickers.values())[1:], start=start,end=end , auto_adjust=False, multi_level_index=False)["Close"]

df = kospi.join(macro , how = "inner")

cpi_index = fdr.DataReader('FRED:KORCPIALLMINMEI', start=start, end=end) #소비자물가지수
base_rate = fdr.DataReader('FRED:INTDSRKRM193N', start=start, end=end) #한국 금리

us_rate = fdr.DataReader('FRED:FEDFUNDS',start=start,end=end)
us_cpi = fdr.DataReader('FRED:CPIAUCSL',start=start,end=end)

jp_rate = fdr.DataReader('FRED:INTDSRJPM193N',start=start,end=end)
jp_cpi = fdr.DataReader('FRED:JPNCPIALLMINMEI',start=start,end=end)


# df: KOSPI 및 일별 시장 데이터
df = df.sort_index()
df.index = pd.to_datetime(df.index)
df.index.name = "Date"

# 경제지표 데이터
macro_series = [
    cpi_index, base_rate,
    us_rate, us_cpi,
    jp_rate, jp_cpi
]

for series in macro_series:
    series.index = pd.to_datetime(series.index)
    series.sort_index(inplace=True)

    # 중복 날짜가 있다면 마지막 관측값 사용
    series = series[~series.index.duplicated(keep="last")]

    # 날짜 인덱스를 컬럼으로 변환해 asof 병합
    left = df.reset_index()
    right = series.reset_index()

    # 첫 번째 컬럼을 날짜 컬럼으로 통일
    right = right.rename(columns={right.columns[0]: "Date"})

    # 경제지표 컬럼 이름 충돌 방지
    value_col = right.columns[1]
    if value_col in left.columns:
        right = right.rename(
            columns={value_col: f"{value_col}_macro"}
        )

    left = left.sort_values("Date")
    right = right.sort_values("Date")

    df = (
        pd.merge_asof(
            left,
            right,
            on="Date",
            direction="backward"
        )
        .set_index("Date")
    )

print("행 개수:", len(df))
print("기간:", df.index.min(), "~", df.index.max())


value = stock.get_market_trading_value_by_date(
    "20030101",
    "20261017",
    "KOSPI"
)




volume = stock.get_market_trading_volume_by_date(
    "20030101",
    "20261017",
    "KOSPI"
)


value = value.rename(columns={
    "기관합계": "institution_net_value",
    "개인": "retail_net_value",
    "외국인합계": "foreign_net_value",
    "기타법인" : "other_net_value",
    "전체" : "all_net_value",
})

volume = volume.rename(columns={
    "기관합계": "institution_net_volume",
    "개인": "retail_net_volume",
    "외국인합계": "foreign_net_volume",
    "기타법인" : "other_net_volume",
    "전체" : "all_net_volume",
})


df = df.join(value, how="inner")
df = df.join(volume, how="inner")

vkospi = pd.read_csv(
    "data/raw/VKOSPI_merged.csv",
    encoding="utf-8-sig"
)

vkospi["일자"] = pd.to_datetime(vkospi["일자"])

vkospi = (
    vkospi.rename(columns={
        "일자": "Date",
        "종가": "VKOSPI"
    })
    .set_index("Date")
    .sort_index()
)

df = df.join(vkospi[["VKOSPI"]], how="left")

print("전체 데이터:", df.shape)
print("VKOSPI 시작일:", df["VKOSPI"].first_valid_index())
print("VKOSPI 결측치:", df["VKOSPI"].isna().sum())

print(df.columns )

df.to_csv("data/raw/data.csv")





