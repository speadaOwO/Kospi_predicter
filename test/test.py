name = "김시후"
score = [70,85,90,65,100]
print(score[-1])
print(score[:3])
score[4] = 0
import pandas as pd
df = pd.read_csv("./hehe/valo2.csv")
df.head(10)
print(df["Kills"],print["Assists"])
df.sort_values(by=["Kills","Assists"],ascending=[True,False])