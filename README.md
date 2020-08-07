# optimize_sample



### 概要

パラメータ最適化処理を実装するためのサンプルコード


### func_optimize.py

scipyを使って関数の最小値(最大値)を求めるサンプルコード
物理現象をシミュレーションするモデル式があって、測定データとシミュレーション結果の誤差が最小となるようなモデルパラメータを推定するなどが活用事例として考えられる。

サンプルでは、生成データから2次関数ax^2+bx+cの係数a,b,cをsimplex法で計算する



### automl.py

scikit-learnを使って、機械学習モデル(Estimator)のハイパーパラメータを最適化するサンプルコード
func_optimize.pyの処理を機械学習処理、simplex法実行時の引数をハイパーパラメータと仮定し、ランダムサーチで最適パラメータの探索を行う。(蛇足ですが、10試行平均を取るような処理を行っています。)



