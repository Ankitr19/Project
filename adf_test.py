import os
import statistics
from scipy import stats
from pandas import Series
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller

def Error_intercept(X, Y):
    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate the sums.

    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sx2 = np.sum(X ** 2)
    Sxy = np.sum(X * Y)
    Sy2 = np.sum(Y ** 2)

    # Calculate re-used expressions.
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2

    # Calculate my, by, ry, s2, smy and sby.
    my = num / den
    by = (Sx2 * Sy - Sx * Sxy) / den
    ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

    diff = Y - by - my * X

    s2 = np.sum(diff * diff) / (n - 2)
    smy = np.sqrt(n * s2 / den)
    sby = np.sqrt(Sx2 * s2 / den)
    return sby


data_from_regression = open("stats_from_linear_regression.txt", "w")
series1 = Series.from_csv('ICICBank.csv',header=0)
series2 = Series.from_csv('HDFCBank.csv',header=0)
X1_icic = series1.values
X1_hdfc = series2.values
X1_residual = []
X1_intercept_diff = []

slope1, intercept1, r_value1, p_value1, std_error1 = stats.linregress(X1_icic, X1_hdfc)
for i in range(len(X1_hdfc)):
    X1_residual.append(X1_hdfc[i] - (X1_icic[i]*slope1 + intercept1))
    X1_intercept_diff.append((X1_hdfc[i] - X1_icic[i]*slope1))

standard_error_of_residuals1 = statistics.stdev(X1_residual)
standard_error_of_intercept1  = Error_intercept(X1_icic, X1_hdfc)
result1 = adfuller(X1_residual)
print(result1[1])


sl1 = "The slope when X is ICIC Bank and Y is HDFC Bank is "+str(slope1)
int1 = "The intercept when X is ICIC Bank and Y is HDFC Bank is "+str(intercept1)
ser1 = "The standard error of residuals when X is ICIC Bank and Y is HDFC Bank is "+str(standard_error_of_residuals1)
sei1 = "The standard error of intercepts when X is ICIC Bank and Y is HDFC Bank is "+str(standard_error_of_intercept1)
er1 = "The error ratio when X is ICIC Bank and Y is HDFC Bank is "+str(standard_error_of_intercept1/standard_error_of_residuals1)

data_from_regression.write(sl1+'\n')
data_from_regression.write(int1+'\n')
data_from_regression.write(ser1+'\n')
data_from_regression.write(sei1+'\n')
data_from_regression.write(er1+'\n\n')


X2_icic = series1.values
X2_hdfc = series2.values
slope2, intercept2, r_value2, p_value2, std_error2 = stats.linregress(X2_hdfc, X2_icic)

X2_residual = []
for i in range(len(X2_icic)):
    X2_residual.append(X2_icic[i] - (X2_hdfc[i]*slope2 + intercept2))

#print(X2_residual)
#print(len(X2_residual))
standard_error_of_residuals2 = statistics.stdev(X2_residual)
standard_error_of_intercept2 = Error_intercept(X2_hdfc, X2_icic)
result2 = adfuller(X2_residual)

print(result2[1])


sl2 = "The slope when X is HDFC Bank and Y is ICIC Bank is "+str(slope2)
int2 = "The intercept when X is HDFC Bank and Y is ICIC Bank is "+str(intercept2)
ser2 = "The standard error of residuals when X is hdfc bank and Y is icic bank is "+str(standard_error_of_residuals2)
sei2 = "The standard error of intercept when X is hdfc bank and Y is icic bank is "+str(standard_error_of_intercept2)
er2 = "The error ratio when when X is hdfc bank and Y is icic bank is "+str(standard_error_of_intercept2/standard_error_of_residuals2)
data_from_regression.write(sl2+'\n')
data_from_regression.write(int2+'\n')
data_from_regression.write(ser2+'\n')
data_from_regression.write(sei2+'\n')
data_from_regression.write(er2+'\n\n')


data_from_regression.close()
#print(X1_hdfc)
#print(X2_hdfc)

'''
X11, X12 = X1[0:split1], X1[split1:]
mean11, mean12 = X11.mean(), X12.mean()
var11, var12 = X11.var(), X12.var()
print('mean11=%f, mean12=%f' % (mean11, mean12))
print('variance11=%f, variance12=%f' % (var11, var12))
'''