# pylint: disable=missing-docstring

#Importing requests module for API part
import requests


# TODO: start by defining a `portfolio` using a dict!

portfolio = {
    
    "aapl": {
        "vol":10,
        "strike":154.12
    },    
    "goog":{
        "vol":2,
        "strike":812.56 
    },
    "tsla":{
        "vol":12,
        "strike":342.12 
    },
    "fb":{
        "vol":18,
        "strike":209.0
    }
}

print(f"Volume of TSLA stocks : {portfolio['tsla']['vol']}\n")
print(f"Strike of GOOG : {portfolio['goog']['strike']}\n")

market = {
    
    "aapl":198.84,
    "goog":1217.93,
    "tsla":267.66,
    "fb":179.06
}

def portfolio_pnl(portfolio, market):
    pnl=0
    for stock,strike in market.items():
        pnl += portfolio[stock]['vol'] * (strike - portfolio[stock]['strike'])
    return pnl

print(f"P&L : {portfolio_pnl(portfolio,market)}\n")


'''######

API Part

######'''
url = "https://api.iextrading.com/1.0/tops/last?symbols="

for stock in portfolio.keys():
    if stock != list(portfolio)[-1]:
        url += stock.upper()+","
    else:
        url += stock.upper()

# print(url)

real_time_market = requests.get(url).json()

real_time_dict = {}
for index, stock in enumerate(real_time_market):
    real_time_dict[stock['symbol'].lower()]=stock['price'] 

print(f"{real_time_dict}\n")
print(f"P&L with real market : {portfolio_pnl(portfolio,real_time_dict)}\n")




