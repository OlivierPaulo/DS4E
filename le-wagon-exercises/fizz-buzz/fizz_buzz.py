# Start with some pseudo-code!
# Players generally sit in a circle. 
# The player designated to go first says the number “1”, 
# and each player thenceforth counts one number in turn. 
# However, any number divisible by three is replaced by the word fizz and any divisible by five by the word buzz.
# Numbers divisible by both become fizzbuzz. 
# A player who hesitates or makes a mistake is eliminated from the game.

#We want to do a loop between 1 and 100 and display fizz for x%3 == 0, buzz for x % 5 == 5, and fizzbuzz if x%3==0 and if x%5 == 0

for i in range (1,101):
    if i%3 ==0 and i%5 ==0:
        print("fizzbuzz")
    elif i%3 == 0:
        print("fizz")
    elif i%5 == 0:
        print("buzz")
    else:
        print(i)
