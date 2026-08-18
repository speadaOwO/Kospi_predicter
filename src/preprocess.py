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



#간단하게 수익률만 사용할 것이므로 다 날리기 
kospi = kospi[["Date", "return", "volume_change"]]
sp500 = sp500[["Date", "return"]]
nasdaq = nasdaq[["Date", "return"]]
sox = sox[["Date", "return"]]
vix = vix[["Date", "change"]]
usdkrw = usdkrw[["Date", "return"]]
us10y = us10y[["Date", "change"]]



#나중에 Merge 하면 return 이 여러개가 되어 꼬임 
kospi = kospi.rename(columns={"return": "KOSPI_return"})
sp500 = sp500.rename(columns={"return": "SP500_return"})
nasdaq = nasdaq.rename(columns={"return": "NASDAQ_return"})
sox = sox.rename(columns={"return": "SOX_return"})
vix = vix.rename(columns={"change": "VIX_change"})
usdkrw = usdkrw.rename(columns={"return": "USDKRW_return"})
us10y = us10y.rename(columns={"change": "US10Y_change"})



#미국장 시차 고려 데이터 누수 방지 , 하루씩 밀기 
USA = [sp500 , nasdaq , sox , vix , us10y]
for i in USA:
    i["Date"] = i["Date"] + pd.Timedelta(days=1)



#data sorting
kospi = kospi.sort_values("Date")
usdkrw = usdkrw.sort_values("Date")
for i in USA:
    i = i.sort_values("Date")



merged = pd.merge_asof(kospi,sp500,on="Date",direction="backward")
merged = pd.merge_asof(merged,nasdaq,on="Date",direction="backward")
merged = pd.merge_asof(merged,sox,on="Date",direction="backward")
merged = pd.merge_asof(merged,vix,on="Date",direction="backward")
merged = pd.merge_asof(merged,usdkrw,on="Date",direction="backward")
merged = pd.merge_asof(merged,us10y,on="Date",direction="backward")



#수익률을 ‰(퍼밀) 단위(*1000) 로 조정 아무래도 숫자가 너무 작음 
t1000 = ["KOSPI_return","SP500_return","NASDAQ_return","SOX_return","VIX_change","USDKRW_return","US10Y_change"]
merged[t1000] = merged[t1000]*1000



#Kospi 정답 레이블 (다음날꺼)
merged["target"] = merged["KOSPI_return"].shift(-1)



#결측치 제거 (NaN)
merged = merged.dropna()



#저장하기!
merged.to_csv("data/processed/dataset.csv" , index=False)



























