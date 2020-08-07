"""
概要
scikitlearn_sample.pyでやった事をoptunaを使ってやってみる。
なお、魔法のコマンドを使ってoptunaをインストールする必要がある。
conda install optuna 
私は問題無かったけど、↓じゃないとダメだと情報もあった
conda install -c conda-forge optuna

参考情報いろいろ
https://qiita.com/studio_haneya/items/2dc3ba9d7cafa36ddffa
https://nine-num-98.blogspot.com/2020/03/ai-hyper-opt-02.html
https://rin-effort.com/2019/12/30/machine-learning-7/
https://cpp-learning.com/optuna-pytorch/
"""

import optuna
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator #これでやるのか？

# 答え
a_ = 5.0
b_ = 2.0
c_ = 10.0

# scipyの評価関数
def func(params, X, Y):
    predictY = params[0] * X * X + params[1] * X + params[2]
    mse = ((predictY - Y)**2).mean()
    #print(mse)
    return mse

# データ生成と配列の確保
dataX = np.arange(100, dtype=float)
dataY = a_ * dataX * dataX + b_ * dataX + c_
param_list = np.zeros((10, 3))    

def objective(trial):
        
    # パラメータの定義
    maxiter_ = trial.suggest_int('max_depth', 1, 10000)
    xtol_ = trial.suggest_uniform('xtol', 0.0, 1.0)
    #maxiter_ = trial.suggest_categorical('maxiter', [1, 10, 100, 200, 300, 400, 500, 1000, 2500, 5000, 10000])
    #xtol_ = trial.suggest_categorical('xtol', [1.0,0.5, 0.25, 1e-1,1e-2,1e-3,1e-4,1e-5, 0.0])

    # 学習処理に該当する操作
    np.random.seed(0)
    arg = (dataX, dataY, )
    for i in range(10):
        param = np.random.uniform(-10.0, 10.0, 3)   
        param_list[i] = optimize.fmin(func, param, args=arg, xtol=xtol_, maxiter=maxiter_, disp=False, full_output=False)

    # 評価値計算処理
    predictY = np.zeros(dataX.size, dtype=float)
    for i in range(10):
        predictY = predictY + param_list[i][0] * dataX * dataX + param_list[i][1] * dataX + param_list[i][2]
    mse = ((predictY / 10 - dataY)**2).mean()

    # 本当はここで、最良モデルを保存する処理を記述する

    return mse


# メインの処理
study = optuna.create_study() # 最大化問題を解きたい時は、"direction='maximize'"とかするらしい
study.optimize(objective, n_trials=2500)

# 結果の出力
print('Best trial:')
trial = study.best_trial
 
print('  Value: {}'.format(trial.value))
 
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))