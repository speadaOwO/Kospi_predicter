import pandas as pd
import yfinance as yf
start = "2010-01-01"
end= "2026-8-16"
kospi = yf.download("^KS11" , start=start,end=end, auto_adjust=False)
SP500 = yf.download("^GSPC",start=start , end=end, auto_adjust=False)
NASDAQ = yf.download("^IXIC",start=start , end=end, auto_adjust=False)
SOX = yf.download("^SOX",start=start , end=end, auto_adjust=False)
VIX = yf.download("^VIX",start=start , end=end, auto_adjust=False)
USDKRW = yf.download("KRW=X",start=start , end=end, auto_adjust=False)
y10us = yf.download("^TNX",start=start , end=end, auto_adjust=False)



kospi.columns = kospi.columns.get_level_values(0)
SP500.columns = SP500.columns.get_level_values(0)
NASDAQ.columns = NASDAQ.columns.get_level_values(0)
SOX.columns = SOX.columns.get_level_values(0)
VIX.columns = VIX.columns.get_level_values(0)
USDKRW.columns = USDKRW.columns.get_level_values(0)
y10us.columns = y10us.columns.get_level_values(0)



kospi.to_csv("data/raw/kospi.csv")
SP500.to_csv("data/raw/SP500.csv")
NASDAQ.to_csv("data/raw/NASDAQ.csv")
SOX.to_csv("data/raw/SOX.csv")
VIX.to_csv("data/raw/VIX.csv")
USDKRW.to_csv("data/raw/USDKRW.csv")
y10us.to_csv("data/raw/y10us.csv")
