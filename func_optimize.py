"""
概要
独自に実装した関数の最大(最小)を求めるためのサンプル
勾配を利用しない最適化手法で、関数の最大(最小)を求める

問題設定
dataX,dataYを利用して、2次関数の係数a,b,cを最適化する。
予測値とdataYの2乗誤差が最小

参考情報1
http://www.turbare.net/transl/scipy-lecture-notes/advanced/mathematical_optimization/index.html

参考情報2
最適化対象に複数戻り値がある場合や説明変量以外の引数がある場合の対策
https://qiita.com/koshian2/items/79a9f97c126cc1bb86e8
"""

import numpy as np
from scipy import optimize


# 答え
a = 5.0
b = 2.0
c = 10.0

# データの定義
dataX = np.arange(100, dtype=float)
dataY = a * dataX * dataX + b * dataX + c
arg = (dataX, dataY, )
print("答え=[{} {} {}]\n".format(a, b, c))

def func(params, X, Y):
    predictY = params[0] * X * X + params[1] * X + params[2]
    mse = ((predictY - Y)**2).mean()
    #print(mse)
    return mse

def cbf(Xi):
    # 調整中のパラメータを出力する
    print(Xi)

# シンプレックス法でパラメータ最適化を行う
# この設定だと、最適解が求まらない場合もある
# 他にも色々な手法があるので、参考情報1を参照
# 初期値をランダムで決定する
print("Simplex法での最適化を10試行実行")
for i in range(10):
   #print("".format(i))
   param = np.random.uniform(-10.0, 10.0, 3)

   # 最適化処理の実行
   # ちなみに最大値を求めたい時に、「func」の代わりに「lambda:x -func(x)」を設定する 
   #res = optimize.fmin(func=func , x0=param, args=arg, xtol=0.0, maxiter=1000, disp=False, full_output=True,callback=cbf)
   res = optimize.fmin(func=func , x0=param, args=arg, xtol=0.0, maxiter=250, disp=False, full_output=True)
   print("試行{}, 最良MSE={}, 最適値={}".format(i, res[1], res[0]))





