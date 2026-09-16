import pandas as pd
import yfinance as yf
import pykrx as pk
from pykrx import stock


start = "2010-01-01"
end= "2026-9-14"

tickers = {
    "KOSPI": "^KS11",
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "SOX": "^SOX",
    "USD/KRW": "KRW=X",
    "VIX": "^VIX",
    "WTI": "CL=F",
    "US10Y": "^TNX",
}

kospi = yf.download("^KS11" , start=start,end=end, auto_adjust=False, multi_level_index=False)

macro  = yf.download(list(tickers.values())[1:], start=start,end=end , auto_adjust=False, multi_level_index=False)["Close"]

df = kospi.join(macro , how = "inner")



value = stock.get_market_trading_value_by_date(
    "20100101",
    "20260914",
    "KOSPI"
)




volume = stock.get_market_trading_volume_by_date(
    "20100101",
    "20260914",
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


print(df.columns )

df.to_csv("data/raw/data.csv")





