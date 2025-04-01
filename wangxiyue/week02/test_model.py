import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from wangxiyue.week02.logistic_iris import predict_iris, initData

if __name__ == '__main__':

    # 花萼长度、花萼宽度、花瓣长度、花瓣宽度
    x1, y1 = load_iris(return_X_y=True)
    xTrain, xTest, yTrain, yTest =  train_test_split(x1, y1, test_size=0.5, shuffle=True)


    for i in range(len(xTest)):
        preY = predict_iris(xTest[i])
        print(f'predict  = {preY} , true  = {yTest[i]}')
        if preY != yTest[i] :
            print(f"FALSE  :  true is {yTest[i]} , but predict is {preY}   " )




