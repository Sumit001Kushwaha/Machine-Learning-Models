import pandas as pd
import numpy as np

data = pd.read_csv("advertising.csv")
features = data.columns[:-1]
target = data.columns[-1]
data[features] = (data[features] - data[features].mean()) / data[features].std()
N = len(data)

def errorCal(slopes,intercept,points):
    slopes = np.array(slopes).reshape(-1,1)
    pred = np.matmul(points[features],slopes) + intercept
    vals =  pred - points[target].values.reshape(-1,1)
    err = np.mean(vals**2)
    return err

def gradientCal(slopes,intercept,points):
    X = points[features].values
    y = points[target].values.reshape(-1,1)
    pred = np.matmul(X,np.array(slopes).reshape(-1,1))+intercept
    err = y-pred
    d_m = (-2/N) * np.matmul(X.T,err).flatten()
    d_b = (-2/N) * np.sum(err)
    return d_m,d_b

learning_rate = 0.01
cycles = 1000
slopes = [0,0,0]
intercept = 0
print(errorCal(slopes,intercept,data))

for _ in range(cycles):
    m,b = gradientCal(slopes,intercept,data)
    slopes-=learning_rate*m
    intercept-=learning_rate*b
    if _%100==0:
        print(errorCal(slopes,intercept,data))
print(errorCal(slopes,intercept,data))