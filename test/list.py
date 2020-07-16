
aapl = [ 10, 154.12 ]
goog = [  2, 812.56 ]
tsla = [ 12, 342.12 ]
fb   = [ 18, 209.0  ]

portfolio = [ aapl, goog, tsla, fb ]



print(f"fb array : {fb}\n")

print(f"portfolio with index 3 : {portfolio[3]}\n")

print(f"volume of fb stocks : {portfolio[3][0]}\n")

#P&L Computation
#            AAPL     GOOG    TSLA      FB
market = [ 198.84, 1217.93, 267.66, 179.06 ]

print('''\n\n\n
---------------
P&L Computation
---------------
\n\n\n\n''')

def portfolio_pnl(portfolio, market):
    
    pnl = 0
    for index, strike in enumerate(portfolio):
        portfolio[index].append(market[index]) #this will store the current_price of stock at 3rd position into the nested array
        pnl += strike[0]*(strike[2]-strike[1]) #compiling pnl (sum of stock volume * (stock current price - stock's "buy" price)). strike correspond to each strike list inside portfolio.
    print(f"New portfolio with current price at 3rd position : {portfolio}\n\n") #printing new nested portfolio array
    return pnl #returning compiled pnl

print(f"P&L : {portfolio_pnl(portfolio,market)}\n") #printing P&L by calling function portfolio_pnl with initial portfolio nested array and market array as arguments
