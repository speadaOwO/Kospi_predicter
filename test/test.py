import pykrx as pk

from pykrx import stock

data = stock.get_market_trading_value_by_date(
    "20220101",
    "20220110",
    "KOSPI"
)

print(data)