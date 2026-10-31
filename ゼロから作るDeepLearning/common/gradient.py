# 数値微分と勾配法（04-learning.ipynb から移したもの）
import numpy as np


# 数値微分
# hを0にすることはできないため、厳密には誤差が生まれてしまう → 中心差分を計算することで解消
# 偏微分の場合には、微分対象の変数だけを残した関数を定義し直して、数値微分をすることで計算（勾配などは別）
def numerical_diff(f, x):
    h = 1e-4 # 0.0001程度であれば、十分良い結果を得ることができる
    return (f(x+h) - f(x-h)) / (2*h)


# 勾配
# 勾配が示す方向は、各場所において関数の値を最も減らす方向を指す（最小ではなく極小を指す）
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状で、要素がすべて0の配列を作成

    for idx in np.ndindex(x.shape): # 何次元でも全要素の位置を順番に回す
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
