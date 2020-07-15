# pylint: disable=missing-docstring
import math

# TODO: 1st exercise: Define the `forward_price` function
def forward_price(spot, interest_rate, time):
    return round(spot * math.exp(interest_rate*time),2)

# TODO: 2nd exercise: Define the `short_pnl` function
def short_pnl(positions, mtm):
    pnl = 0
    for i in range(len(positions)):
        pnl += mtm[i] - positions[i]
    return pnl
        