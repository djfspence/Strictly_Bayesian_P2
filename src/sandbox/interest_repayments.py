import numpy as np

# exploring the relationship between house price, interest rate

P_base = 1000000
i_base = 2.7 / 100.0
n = 25

R = P_base * i_base / (1.0 - (1.0 / (1.0 + i_base)**n))

print R

balance = P_base
i = i_base

for year in range(25):

    balance = (balance * (1.0 + i)) - R
    print year, "%.0f" % balance


for percent_interest in range(1,16):

    i = percent_interest/100.0

    P = R * (1.0 - (1.0 / (1.0 + i)**n)) / i

    print percent_interest, "%.0f" % P, "%.2f" % (1.0 * P/(1.0 * P_base))


print 'With inflation'

j = 0.02
a = 1.0 + i_base
b = 1.0 + j
d = a/b

Ri = P_base * (a**n) * (d - 1.0) / (((d**n) - 1.0) * b**n)

Rim = Ri/12

print "%.0f" % Ri, "%.0f" % Rim

for percent_interest in range(1,16):

    i = percent_interest / 100.0

    a = 1.0 + i

    P = Ri * (d**n - 1.0) * b**n / (a**n * (d-1.0))

    print percent_interest, "%.0f" % P, "%.2f" % (1.0 * P / (1.0 * P_base))
