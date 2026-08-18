import numpy as np
import pandas as pd

merged = pd.read_csv("data/processed/dataset.csv")



#Train , Valid , Test 나누기!
mgLenth = len(merged)
TL = int(mgLenth * 0.7)
VL = int(mgLenth * 0.85)

Train = merged.iloc[:TL]
Valid = merged.iloc[TL:VL]
Test = merged.iloc[VL:]



#X , Y 분리 (정답 분리)
shouldtrain = ["KOSPI_return",
    "volume_change",
    "SP500_return",
    "NASDAQ_return",
    "SOX_return",
    "VIX_change",
    "USDKRW_return",
    "US10Y_change"]

X_train = Train[shouldtrain]
y_train = Train["target"]

X_valid = Valid[shouldtrain]
y_valid = Valid["target"]

X_test = Test[shouldtrain]
y_test = Test["target"]


print(X_test.head(10))
print(y_test.head(10))




#window 만들기 (일단 흐름을 줘야함ㅁㅁ) ex window 30 -> 30일치 데이터 뭉텅이로 보내기
# src/dataset.py
def make_sequencesX(X, window_size=30):
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X.iloc[i:i + window_size].values)
    return np.array(X_seq) 

def make_sequencesy(y ,window_size=30):
    y_seq = []
    for i in range(len(y) - window_size):
        y_seq.append(y.iloc[i + window_size])
    return np.array(y_seq) 


X_train = make_sequencesX(X_train , window_size=30)
y_train = make_sequencesy(y_train , window_size=30)

X_valid = make_sequencesX(X_valid , window_size=30)
y_valid = make_sequencesy(y_valid , window_size=30)

X_test = make_sequencesX(X_test , window_size=30)
y_test = make_sequencesy(y_test , window_size=30)



#저장하기
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)

np.save("data/processed/X_valid.npy", X_valid)
np.save("data/processed/y_valid.npy", y_valid)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)