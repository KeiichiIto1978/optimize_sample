"""
概要
simplex法によるパラメータフィッティングを10回行う
各試行で求まった予測値y'(n)の平均値を推論結果とする。
最小となるxtolとmaxiterをハイパーパラメータとして決定するestimatorを作成し、ランダムサーチで最適化する

参考情報
https://qiita.com/_takoika/items/89a7e42dd0dc964d0e29
"""

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

# 答え
a_ = 5.0
b_ = 2.0
c_ = 10.0

# データの定義
dataX = np.arange(100, dtype=float)
dataY = a_ * dataX * dataX + b_ * dataX + c_

# 評価関数の実体
def func(params, X, Y):
    predictY = params[0] * X * X + params[1] * X + params[2]
    mse = ((predictY - Y)**2).mean()
    #print(mse)
    return mse

class SimplexEstimator(BaseEstimator):

    def __init__(self, maxiter=10, xtol=1.0):
        self.maxiter = maxiter
        self.xtol = xtol
        self.param_list = np.zeros((10, 3))

    def fit(self, x, y):
        np.random.seed(0)
        arg = (dataX, dataY, )
        for i in range(10):
           param = np.random.uniform(-10.0, 10.0, 3)   
           self.param_list[i] = optimize.fmin(func, param, args=arg, xtol=self.xtol, maxiter=self.maxiter, disp=False, full_output=False)
        return self 

    def predict(self, x):
        predictY = np.zeros(x.size, dtype=float)
        for i in range(10):
            predictY = predictY + self.param_list[i][0] * x * x + self.param_list[i][1] * x + self.param_list[i][2]
        predictY = predictY / 10.0
        return predictY 

    def score(self, x, y):
        predictY = self.predict(x)
        mse = ((predictY - y)**2).mean()
        #print(mse)
        return -mse

    def get_params(self, deep=True):
        return {'maxiter': self.maxiter, 'xtol': self.xtol}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def print(self):
        print (self.param_list)
        return self

param_dist = {'maxiter': [1, 10, 100, 200, 300, 400, 500, 1000, 2500, 5000, 10000],
              'xtol':[1.0,0.5, 0.25, 1e-1,1e-2,1e-3,1e-4,1e-5, 0.0]}

simplex_estimator = SimplexEstimator()
random_search = RandomizedSearchCV( estimator=simplex_estimator,
                                    param_distributions=param_dist,
                                    cv=3,              #CV
                                    n_iter=20,          #interation num
                                    n_jobs=1,           #num of core
                                    verbose=1,          
                                    random_state=1)

random_search.fit(dataX,dataY)
random_search_best = random_search.best_estimator_ #best estimator
print("Best Model Parameter: ",random_search.best_params_)
print("Best Model Score: ",random_search.best_score_)

random_search_best.print()
