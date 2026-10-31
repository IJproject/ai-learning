# 活性化関数（03-neural-network.ipynb から移したもの）
import numpy as np


# ステップ関数（階段関数）
# 入力が配列になるケースも想定した実装
def step_function(x):
    y = x > 0
    return y.astype(int) # 各要素の値を Boolean から Integer に変換する


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU関数
def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    # axis=-1: 最後の軸（各データのクラス方向）ごとに計算する。ミニバッチ (N, クラス数) では行ごと
    # keepdims=True: (N, 1) の形で残し、a や exp_a との引き算・割り算が行ごとに効くようにする
    c = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y
