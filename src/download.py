import pandas as pd
import yfinance as yf
start = "2010-01-01"
end= "2026-8-16"
kospi = yf.download("^KS11" , start=start,end=end)
SP500 = yf.download("^GSPC",start=start , end=end)
NASDAQ = yf.download("^IXIC",start=start , end=end)
SOX = yf.download("^SOX",start=start , end=end)
VIX = yf.download("^VIX",start=start , end=end)
USDKRW = yf.download("KRW=X",start=start , end=end)
y10us = yf.download("^TNX",start=start , end=end)

kospi.to_csv("data/raw/kospi.csv")
SP500.to_csv("data/raw/SP500.csv")
NASDAQ.to_csv("data/raw/NASDAQ.csv")
SOX.to_csv("data/raw/SOX.csv")
VIX.to_csv("data/raw/VIX.csv")
USDKRW.to_csv("data/raw/USDKRW.csv")
y10us.to_csv("data/raw/y10us.csv")
