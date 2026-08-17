import pandas as pd 
import numpy as np 

#가져오기
kospi = pd.read_csv("data/raw/kospi.csv")
sp500 = pd.read_csv("data/raw/SP500.csv")
nasdaq = pd.read_csv("data/raw/NASDAQ.csv")
sox = pd.read_csv("data/raw/SOX.csv")
vix = pd.read_csv("data/raw/VIX.csv")
usdkrw = pd.read_csv("data/raw/USDKRW.csv")
us10y = pd.read_csv("data/raw/y10us.csv")

#date(문자열) 을 pandas 가 읽도록 datetime 로 
kospi["Date"] = pd.to_datetime(kospi["Date"])
sp500["Date"] = pd.to_datetime(sp500["Date"])
nasdaq["Date"] = pd.to_datetime(nasdaq["Date"])
sox["Date"] = pd.to_datetime(sox["Date"])
vix["Date"] = pd.to_datetime(vix["Date"])
usdkrw["Date"] = pd.to_datetime(usdkrw["Date"])
us10y["Date"] = pd.to_datetime(us10y["Date"])

#return 은 수익률임 
kospi["return"] = kospi["Close"].pct_change()

#Volume 은 거래량임
kospi["volume_change"] = kospi["Volume"].pct_change()

sp500["return"] = sp500["Close"].pct_change()
nasdaq["return"] = nasdaq["Close"].pct_change()
sox["return"] = sox["Close"].pct_change()
usdkrw["return"] = usdkrw["Close"].pct_change()
vix["change"] = vix["Close"].pct_change()
us10y["change"] = us10y["Close"].diff()


kospi = kospi.sort_values("Date")
usdkrw = usdkrw.sort_values("Date")


merged = pd.merge_asof(
    kospi,
    usdkrw,
    on="Date",
    direction="backward"
)